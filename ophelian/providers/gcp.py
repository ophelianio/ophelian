"""GCP provider — orchestrates a pipeline against GCP infrastructure.

Mirror of :mod:`ophelian.providers.aws`. The provider stitches together
three pieces:

* a :class:`~ophelian.envs.gcp.GCPConfig` (validated at the env level),
* a :class:`~ophelian.providers.gcp_drivers.CloudDriver` that knows how
  to run a single step on GCP (GCE or Local), and
* an :class:`~ophelian.stores.base.ArtifactStore` — almost always
  :class:`~ophelian.stores.gcs.GCSArtifactStore`.

It owns the same state and lifecycle as :class:`AWSProvider`:

* assign a stable run id (or resume an existing one);
* fan out steps in topological order, persisting per-step artifacts;
* on :class:`~ophelian.runtime.spot.SpotInterruption` (= GCE
  preemption), write a checkpoint so the next ``pipe.run`` can resume;
* on any other failure, mark the run failed and tear down the driver.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, cast

from ophelian.core.nodes import (
    Data,
    Deploy,
    Eval,
    Pipeline,
    PipelineResult,
    StepResult,
    Train,
    Tune,
)
from ophelian.observability import bind_run, set_run_id
from ophelian.observability.summary import emit_run_summary
from ophelian.providers.base import Provider
from ophelian.providers.gcp_drivers import (
    GCEDriver,
    LocalGCPDriver,
    StepOutcome,
    StepRequest,
    persist_run_checkpoint,
    random_run_id,
    train_checkpoint_key,
)
from ophelian.runtime.spot import SpotInterruption, load_checkpoint

if TYPE_CHECKING:  # pragma: no cover
    from ophelian.core.compiler import ExecutionPlan
    from ophelian.envs.gcp import GCPConfig
    from ophelian.providers.gcp_drivers import LocalGCPDriver as _LocalGCPDriver  # noqa: F401
    from ophelian.stores.base import ArtifactStore

logger = logging.getLogger("ophelian.providers.gcp")


class GCPProvider(Provider):
    """Execute pipelines on GCP infrastructure."""

    name = "gcp"

    def __init__(
        self,
        *,
        config: GCPConfig,
        driver: Any | None = None,
        store: ArtifactStore | None = None,
        run_id: str | None = None,
    ) -> None:
        self._config = config
        self._store: ArtifactStore = store if store is not None else self._build_default_store()
        self._driver = driver if driver is not None else self._build_default_driver()
        self._run_id = run_id or config.resume_run_id or random_run_id()
        self._last_run_id: str | None = None
        self._last_checkpoint_uri: str | None = None

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _build_default_driver(self) -> Any:
        if self._config.gcp_backend == "gce":
            return GCEDriver(self._config)
        if self._config.gcp_backend == "gke":
            raise NotImplementedError(
                "The GKE backend is on the post-1.0 roadmap — v1.0 ships "
                "GCE only. Use `gcp_backend='gce'` (the default) until the "
                "GKE driver lands."
            )
        raise ValueError(f"Unknown GCP backend: {self._config.gcp_backend!r}")

    def _build_default_store(self) -> ArtifactStore:
        from ophelian.stores.gcs import GCSArtifactStore

        bucket = self._config.artifact_bucket
        if not bucket:
            bucket = f"ophelian-artifacts-{self._config.project}"
            self._config = self._config.model_copy(update={"artifact_bucket": bucket})
        return GCSArtifactStore(
            bucket=bucket,
            prefix=self._config.artifact_prefix,
            project=self._config.project,
            location=self._config.region,
        )

    def with_resume(self, run_id: str) -> GCPProvider:
        """Return a new provider configured to resume *run_id*."""
        new_config = self._config.with_resume(run_id)
        return GCPProvider(
            config=new_config,
            driver=self._driver,
            store=self._store,
            run_id=run_id,
        )

    def _train_checkpoint_uri(self, run_id: str, step_name: str) -> str | None:
        from ophelian.stores.gcs import GCSArtifactStore

        if not isinstance(self._store, GCSArtifactStore):
            return None
        sub = train_checkpoint_key(run_id, step_name)
        full = f"{self._store.prefix}/{sub}" if self._store.prefix else sub
        return f"gs://{self._store.bucket}/{full}"

    # ------------------------------------------------------------------
    # Public surface
    # ------------------------------------------------------------------

    @property
    def config(self) -> GCPConfig:
        return self._config

    @property
    def driver(self) -> Any:
        return self._driver

    @property
    def store(self) -> ArtifactStore:
        return self._store

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def last_checkpoint_uri(self) -> str | None:
        return self._last_checkpoint_uri

    def describe(self) -> str:
        return (
            f"gcp[{self._config.gcp_backend}]"
            f"(region={self._config.region}, machine_type={self._config.machine_type}, "
            f"spot={self._config.spot})"
        )

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(self, pipeline: Pipeline, plan: ExecutionPlan) -> PipelineResult:
        with bind_run(self._run_id):
            return self._execute_bound(pipeline, plan)

    def _execute_bound(
        self, pipeline: Pipeline, plan: ExecutionPlan
    ) -> PipelineResult:
        run_id = self._run_id
        self._last_run_id = run_id
        logger.info(
            "Starting GCP run %s for pipeline %s (driver=%s)",
            run_id,
            pipeline.name,
            self._driver.name,
        )

        artifact_index: dict[str, dict[str, str]] = {}
        info_index: dict[str, dict[str, Any]] = {}
        completed: list[str] = []
        results: list[StepResult] = []

        resume_target: str | None = None
        if self._config.resume_run_id:
            checkpoint = load_checkpoint(self._store, self._config.resume_run_id)
            if checkpoint is not None:
                logger.info(
                    "Resuming run %s — %d completed step(s); restarting at %r",
                    checkpoint.run_id,
                    len(checkpoint.completed_steps),
                    checkpoint.in_flight_step,
                )
                completed = list(checkpoint.completed_steps)
                artifact_index = {k: dict(v) for k, v in checkpoint.artifacts.items()}
                resume_target = checkpoint.in_flight_step
                run_id = checkpoint.run_id
                self._run_id = run_id
                self._last_run_id = run_id
                set_run_id(run_id)
                for step_name in completed:
                    results.append(
                        StepResult(
                            name=step_name,
                            kind=_kind_for_step(plan, step_name),
                            status="success",
                            artifacts=artifact_index.get(step_name, {}),
                            info={"resumed": True, "run_id": run_id},
                        )
                    )

        try:
            for step in plan.steps:
                if step.name in completed and step.name != resume_target:
                    continue
                step_resume_from: str | None = None
                if step.name == resume_target and step.kind == "train":
                    step_resume_from = self._train_checkpoint_uri(run_id, step.name)
                request = StepRequest(
                    run_id=run_id,
                    pipeline_name=pipeline.name,
                    step_name=step.name,
                    kind=step.kind,
                    node=cast("Data | Train | Tune | Eval | Deploy", step.node),
                    upstream_artifacts={
                        dep: artifact_index.get(dep, {}) for dep in step.depends_on
                    },
                    upstream_info={
                        dep: info_index.get(dep, {}) for dep in step.depends_on
                    },
                    resume=step.name == resume_target,
                    resume_from=step_resume_from,
                )
                _step_t0 = time.monotonic()
                try:
                    outcome: StepOutcome = self._driver.execute(request, self._store)
                except SpotInterruption as exc:
                    logger.warning(
                        "Preemption at step %s — checkpointing run %s",
                        step.name,
                        run_id,
                    )
                    self._last_checkpoint_uri = persist_run_checkpoint(
                        self._store,
                        run_id=run_id,
                        pipeline=pipeline.name,
                        completed=completed,
                        in_flight=step.name,
                        artifacts=artifact_index,
                        info={
                            "reason": str(exc),
                            "deadline": exc.deadline,
                            "machine_type": self._config.machine_type,
                        },
                    )
                    failure = StepResult(
                        name=step.name,
                        kind=step.kind,
                        status="failed",
                        error=(
                            f"preempted: resume with GCP(... resume_run_id={run_id!r})"
                        ),
                        info={
                            "checkpoint_uri": self._last_checkpoint_uri,
                            "run_id": run_id,
                            "resumable": True,
                        },
                        duration_seconds=time.monotonic() - _step_t0,
                    )
                    results.append(failure)
                    return PipelineResult(pipeline=pipeline.name, steps=results)
                except Exception as exc:
                    logger.exception("Step %s failed on GCP driver", step.name)
                    results.append(
                        StepResult(
                            name=step.name,
                            kind=step.kind,
                            status="failed",
                            error=str(exc),
                            info={"run_id": run_id},
                            duration_seconds=time.monotonic() - _step_t0,
                        )
                    )
                    return PipelineResult(pipeline=pipeline.name, steps=results)

                step_result = outcome.result.model_copy(
                    update={"duration_seconds": time.monotonic() - _step_t0}
                )
                results.append(step_result)
                artifact_index[step.name] = dict(step_result.artifacts)
                info_index[step.name] = dict(step_result.info)
                if step_result.status == "success":
                    if step.name not in completed:
                        completed.append(step.name)
                else:
                    return PipelineResult(pipeline=pipeline.name, steps=results)

            return PipelineResult(pipeline=pipeline.name, steps=results)
        finally:
            try:
                self._driver.teardown()
            except Exception:  # pragma: no cover
                logger.warning("Driver teardown raised", exc_info=True)
            try:
                emit_run_summary(
                    provider=self.name,
                    run_id=run_id,
                    pipeline=pipeline.name,
                    steps=results,
                    description=self.describe(),
                    hourly_usd=getattr(self, "_router_quote_hourly_usd", None),
                )
            except Exception:  # pragma: no cover - summary is best-effort
                logger.debug("Summary emission failed", exc_info=True)

    def cleanup(self) -> None:
        import contextlib

        with contextlib.suppress(Exception):
            self._driver.teardown()

    def cleanup_deploys(self) -> None:
        import contextlib

        cleanup = getattr(self._driver, "cleanup_deploys", None)
        if cleanup is None:
            return
        with contextlib.suppress(Exception):
            cleanup()


def _kind_for_step(plan: ExecutionPlan, step_name: str) -> str:
    for step in plan.steps:
        if step.name == step_name:
            return step.kind
    return "unknown"


__all__ = ["GCPProvider", "LocalGCPDriver"]
