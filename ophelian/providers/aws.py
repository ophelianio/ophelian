"""AWS provider — orchestrates a pipeline against AWS infrastructure.

The provider stitches together three pieces:

* a :class:`~ophelian.envs.aws.AWSConfig` (validated at the env level),
* a :class:`~ophelian.providers.aws_drivers.CloudDriver` that knows how to
  run a single step on AWS (EC2, EKS or the in-process Local driver tests
  use), and
* an :class:`~ophelian.stores.base.ArtifactStore` — almost always
  :class:`~ophelian.stores.s3.S3ArtifactStore` — that holds the dataset,
  model and checkpoint artifacts.

It owns just enough state to:

* assign a stable run id (or resume an existing one when the user passes
  ``resume_run_id`` in the env);
* fan out steps in topological order, persisting per-step artifacts so the
  next step can find them;
* on :class:`~ophelian.runtime.spot.SpotInterruption`, write a checkpoint
  the next ``pipe.run`` invocation can resume from;
* on any other failure, mark the run failed and tear down the driver.

Everything heavy (AWS API calls, container execution) lives in the driver
modules; this file is the deterministic glue.
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
from ophelian.providers.aws_drivers import (
    CloudDriver,
    EC2Driver,
    EKSDriver,
    StepOutcome,
    StepRequest,
    persist_run_checkpoint,
    random_run_id,
    train_checkpoint_key,
)
from ophelian.providers.base import Provider
from ophelian.runtime.spot import (
    SpotInterruption,
    load_checkpoint,
)

if TYPE_CHECKING:  # pragma: no cover
    from ophelian.core.compiler import ExecutionPlan
    from ophelian.envs.aws import AWSConfig
    from ophelian.stores.base import ArtifactStore

logger = logging.getLogger("ophelian.providers.aws")


class AWSProvider(Provider):
    """Execute pipelines on AWS infrastructure.

    Most users construct this through :func:`ophelian.envs.AWS` rather than
    directly. The ``driver`` and ``store`` constructor arguments exist so
    tests can plug in deterministic substitutes (typically a
    :class:`LocalDriver` over a moto-mocked S3 store).
    """

    name = "aws"

    def __init__(
        self,
        *,
        config: AWSConfig,
        driver: CloudDriver | None = None,
        store: ArtifactStore | None = None,
        run_id: str | None = None,
    ) -> None:
        self._config = config
        # Order matters: ``_build_default_store`` may auto-provision a
        # bucket and write the resolved name back into ``self._config``.
        # The driver has to be built AFTER that so its captured config
        # sees the resolved ``artifact_bucket`` — otherwise the EC2
        # user-data script would export an empty
        # ``OPHELIAN_ARTIFACT_BUCKET`` and result polling would fail.
        self._store: ArtifactStore = store if store is not None else self._build_default_store()
        self._driver: CloudDriver = driver if driver is not None else self._build_default_driver()
        self._run_id = run_id or config.resume_run_id or random_run_id()
        self._last_run_id: str | None = None
        self._last_checkpoint_uri: str | None = None

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _build_default_driver(self) -> CloudDriver:
        if self._config.aws_backend == "ec2":
            return EC2Driver(self._config)
        if self._config.aws_backend == "eks":
            return EKSDriver(self._config)
        raise ValueError(f"Unknown AWS backend: {self._config.aws_backend!r}")

    def _build_default_store(self) -> ArtifactStore:
        """Build (and if needed, provision) the S3 artifact store.

        ``AWS(region=..., instance=..., spot=True)`` must work as the
        task spec promises — without forcing the caller to also supply
        ``artifact_bucket=``. When the user omits the bucket we derive a
        deterministic per-account/per-region name
        (``ophelian-artifacts-{account_id}-{region}``) via
        ``sts:GetCallerIdentity`` and create it lazily if it doesn't
        already exist. The resolved bucket is also written back into
        ``self._config`` so the EC2 user-data script (which serialises
        ``config.artifact_bucket`` directly into ``OPHELIAN_ARTIFACT_BUCKET``)
        sees the same value.
        """
        from ophelian.stores.s3 import S3ArtifactStore

        bucket = self._config.artifact_bucket
        if not bucket:
            bucket = _ensure_default_bucket(self._config.region)
            self._config = self._config.model_copy(update={"artifact_bucket": bucket})
        return S3ArtifactStore(
            bucket=bucket,
            prefix=self._config.artifact_prefix,
            region=self._config.region,
        )

    def with_resume(self, run_id: str) -> AWSProvider:
        """Return a new provider configured to resume *run_id*.

        Mirrors :meth:`AWSConfig.with_resume` at the env/provider layer
        so users can write::

            env = AWS(region="us-east-1", instance="m5.large", spot=True).with_resume("run-abc")
            pipeline.run(env=env)

        which is the public surface documented in the README/changelog.
        Driver and store are reused as-is so that any spun-up clients
        (and our internal ``_run_id``) stay coherent.
        """
        new_config = self._config.with_resume(run_id)
        return AWSProvider(
            config=new_config,
            driver=self._driver,
            store=self._store,
            run_id=run_id,
        )

    def _train_checkpoint_uri(self, run_id: str, step_name: str) -> str | None:
        """``s3://bucket/[prefix/]runs/{run_id}/checkpoints/{step}`` URI
        the worker uploads its in-flight checkpoint to on SIGTERM.

        Returns ``None`` for non-S3 stores (the in-process executor does
        not need an S3 round trip — it can read the local checkpoint dir
        directly), which is what the standalone tests exercise.
        """
        from ophelian.stores.s3 import S3ArtifactStore

        if not isinstance(self._store, S3ArtifactStore):
            return None
        sub = train_checkpoint_key(run_id, step_name)
        full = f"{self._store.prefix}/{sub}" if self._store.prefix else sub
        return f"s3://{self._store.bucket}/{full}"

    # ------------------------------------------------------------------
    # Public surface
    # ------------------------------------------------------------------

    @property
    def config(self) -> AWSConfig:
        return self._config

    @property
    def driver(self) -> CloudDriver:
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
            f"aws[{self._config.aws_backend}]"
            f"(region={self._config.region}, instance={self._config.instance}, "
            f"spot={self._config.spot})"
        )

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(self, pipeline: Pipeline, plan: ExecutionPlan) -> PipelineResult:
        with bind_run(self._run_id):
            return self._execute_bound(pipeline, plan)

    def _execute_bound(self, pipeline: Pipeline, plan: ExecutionPlan) -> PipelineResult:
        run_id = self._run_id
        self._last_run_id = run_id
        logger.info(
            "Starting AWS run %s for pipeline %s (driver=%s)",
            run_id,
            pipeline.name,
            self._driver.name,
        )

        artifact_index: dict[str, dict[str, str]] = {}
        # Parallel index of upstream ``info`` dicts so deploy steps can
        # read the trainer's framework / model_path without re-fetching
        # ``result.json`` from S3.
        info_index: dict[str, dict[str, Any]] = {}
        completed: list[str] = []
        results: list[StepResult] = []

        # Resume support — load the prior checkpoint if one exists.
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
                # Materialise StepResult stubs for every step we already finished
                # so the returned PipelineResult is complete.
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
                # On retry of an interrupted Train step, hand the worker
                # the URI of the checkpoint snapshot the previous attempt
                # uploaded on SIGTERM so the adapter can resume training
                # rather than restart it.
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
                    upstream_info={dep: info_index.get(dep, {}) for dep in step.depends_on},
                    resume=step.name == resume_target,
                    resume_from=step_resume_from,
                    context=getattr(pipeline, "context", None),
                )
                _step_t0 = time.monotonic()
                try:
                    outcome: StepOutcome = self._driver.execute(request, self._store)
                except SpotInterruption as exc:
                    logger.warning(
                        "Spot interruption at step %s — checkpointing run %s",
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
                            "instance": self._config.instance,
                        },
                    )
                    failure = StepResult(
                        name=step.name,
                        kind=step.kind,
                        status="failed",
                        error=f"spot-interrupted: resume with AWS(... resume_run_id={run_id!r})",
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
                    logger.exception("Step %s failed on AWS driver", step.name)
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
            except Exception:  # pragma: no cover - defensive
                logger.warning("Driver teardown raised", exc_info=True)
            try:
                from ophelian.observability.summary import emit_run_summary

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
        """Tear down any leftover driver state (no-op when already torn down)."""
        import contextlib

        with contextlib.suppress(Exception):  # pragma: no cover
            self._driver.teardown()

    def cleanup_deploys(self) -> None:
        """Terminate every long-lived Deploy worker tracked by the driver.

        ``Pipeline.run(env=AWS(...))`` intentionally leaves Deploy
        instances running so the FastAPI endpoint stays reachable. Call
        this when you're done serving — typically from a ``finally``
        block in your script. No-op for drivers that don't expose the
        method (e.g. EKS, where deletion is part of ``teardown()``).
        """
        import contextlib

        cleanup = getattr(self._driver, "cleanup_deploys", None)
        if cleanup is None:
            return
        with contextlib.suppress(Exception):  # pragma: no cover
            cleanup()


def _kind_for_step(plan: ExecutionPlan, step_name: str) -> str:
    for step in plan.steps:
        if step.name == step_name:
            return step.kind
    return "unknown"


def _ensure_default_bucket(region: str) -> str:
    """Resolve and create-if-missing a default per-account bucket.

    Naming follows ``ophelian-artifacts-{account_id}-{region}`` so it's
    deterministic across runs (idempotent on the second invocation) and
    avoids name collisions in the global S3 namespace. ``us-east-1``
    must NOT be passed as a ``LocationConstraint`` to ``create_bucket``
    (S3 quirk); every other region must.
    """
    import boto3
    from botocore.exceptions import ClientError

    sts = boto3.client("sts", region_name=region)
    try:
        account_id = sts.get_caller_identity()["Account"]
    except ClientError as exc:
        raise ValueError(
            "Could not auto-derive a default S3 bucket because "
            f"sts:GetCallerIdentity failed: {exc}. Either configure AWS "
            "credentials or pass `artifact_bucket=` to AWS(...) explicitly."
        ) from exc
    bucket = f"ophelian-artifacts-{account_id}-{region}"
    s3 = boto3.client("s3", region_name=region)
    try:
        s3.head_bucket(Bucket=bucket)
        return bucket
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code")
        if code not in {"404", "NoSuchBucket", "NotFound"}:
            # 403 likely means it exists but in another account / we
            # lack ListBucket. Raise so the user picks an explicit name.
            raise ValueError(
                f"Default bucket {bucket!r} exists but is inaccessible ({code}). "
                "Pass `artifact_bucket=<your-bucket>` to AWS(...) instead."
            ) from exc
    create_kwargs: dict[str, object] = {"Bucket": bucket}
    if region != "us-east-1":
        create_kwargs["CreateBucketConfiguration"] = {"LocationConstraint": region}
    try:
        s3.create_bucket(**create_kwargs)
    except ClientError as exc:
        raise ValueError(
            f"Could not auto-provision default bucket {bucket!r}: {exc}. "
            "Either grant s3:CreateBucket or pass `artifact_bucket=` explicitly."
        ) from exc
    logger.info("Provisioned default Ophelian S3 bucket %s in %s", bucket, region)
    return bucket


__all__ = ["AWSProvider"]
