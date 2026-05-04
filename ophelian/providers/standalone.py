"""Standalone provider — runs each pipeline step locally.

The Standalone provider has two execution modes:

* ``inprocess`` (default fallback): every node runs in the current Python
  process. Fast, dependency-free, ideal for unit tests, notebooks and
  environments without a Docker daemon.

* ``container``: every node runs inside a Docker container produced from a
  small ``ophelian-runtime`` image. ``Train``/``Eval``/``Tune``/``Data`` use
  one-shot containers that mount a per-step workspace and read/write
  artifacts on the host filesystem. ``Deploy`` runs a detached container
  publishing its port so callers can hit ``/health`` and ``/predict`` on
  the bound host port.

Mode selection
--------------

Pass ``container=True`` to require containerised execution (raises
:class:`~ophelian.providers.docker_engine.DockerUnavailableError` if the
daemon is not reachable). Pass ``container=False`` to force in-process.
The default (``container="auto"``) probes the daemon and picks
``container`` if reachable, otherwise ``inprocess``.

For tests, inject a :class:`FakeDockerEngine` via ``docker_engine=...`` —
the provider then issues real Docker commands against the fake and tests
can assert on the recorded actions without a daemon.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from ophelian import __version__ as _OPHELIAN_VERSION
from ophelian.core.nodes import (
    Data,
    Deploy,
    Eval,
    PipelineResult,
    StepResult,
    Train,
    Tune,
)
from ophelian.observability import bind_run
from ophelian.providers.base import Provider
from ophelian.providers.docker_engine import (
    ContainerHandle,
    DockerEngine,
    DockerUnavailableError,
    FakeDockerEngine,
    RealDockerEngine,
)

if TYPE_CHECKING:
    from fastapi import FastAPI

    from ophelian.core.compiler import ExecutionPlan, PlannedStep
    from ophelian.core.nodes import Pipeline


logger = logging.getLogger("ophelian.providers.standalone")

ContainerMode = Literal["auto", True, False]

_DEFAULT_RUNTIME_IMAGE_TAG = f"ophelian-runtime:{_OPHELIAN_VERSION}"
_DEFAULT_RUNTIME_EXTRAS: tuple[str, ...] = ("sklearn",)


def _runtime_dockerfile(extras: tuple[str, ...]) -> str:
    """Dockerfile for the reusable runtime image used by all step containers.

    ``extras`` controls which adapter dependencies are installed alongside
    the framework. Defaults to just ``sklearn`` to keep the image small;
    callers running container-mode pipelines for ``pytorch``/``xgboost``/
    ``huggingface`` should pass the matching extras (or ``("all",)``) so
    the in-container step runner can import the relevant adapter.
    """
    extras_spec = ",".join(extras) if extras else ""
    install_target = f".[{extras_spec}]" if extras_spec else "."
    return (
        "FROM python:3.12-slim\n"
        "ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1\n"
        "WORKDIR /opt/ophelian\n"
        "COPY . /opt/ophelian\n"
        f"RUN pip install --upgrade pip && pip install '{install_target}' uvicorn fastapi\n"
        "WORKDIR /work\n"
    )


# Re-export for backwards compatibility / external callers.
__all__ = [
    "ContainerHandle",
    "DockerEngine",
    "DockerUnavailableError",
    "FakeDockerEngine",
    "RealDockerEngine",
    "StandaloneProvider",
]


class StandaloneProvider(Provider):
    """Execute pipelines locally — in-process or in Docker containers.

    Parameters
    ----------
    local:
        Must be ``True`` for v0.1; remote standalone clusters land later.
    workspace:
        Directory used for per-step artifacts. Defaults to a fresh temp dir
        under ``$OPHELIAN_HOME`` (or the OS temp dir).
    container:
        ``True`` to force Docker-backed execution, ``False`` to force
        in-process, ``"auto"`` (default) to pick container if a Docker
        daemon is reachable.
    docker_engine:
        Override the engine used for container mode. Tests typically pass a
        :class:`FakeDockerEngine`. Defaults to :class:`RealDockerEngine`.
    runtime_image:
        Tag of the container image used to run steps. Built on first use
        from the source tree if it does not exist.
    runtime_extras:
        Tuple of pyproject extras installed into the runtime image
        alongside ``ophelian`` itself. Defaults to ``("sklearn",)``;
        pipelines that train PyTorch, XGBoost or HuggingFace models in
        container mode should pass the matching extras (e.g.
        ``("sklearn", "pytorch")``) or ``("all",)``.
    project_root:
        Path used as the build context when the runtime image needs to be
        built. Defaults to the repository root inferred from this file.
    serve_deploys:
        For in-process mode only — start a uvicorn subprocess after
        building each Deploy app. Container mode always serves Deploy
        steps via a detached container.
    """

    name = "standalone"

    def __init__(
        self,
        *,
        local: bool = True,
        workspace: str | os.PathLike[str] | None = None,
        container: ContainerMode = "auto",
        docker_engine: DockerEngine | None = None,
        runtime_image: str = _DEFAULT_RUNTIME_IMAGE_TAG,
        runtime_extras: tuple[str, ...] = _DEFAULT_RUNTIME_EXTRAS,
        project_root: str | os.PathLike[str] | None = None,
        serve_deploys: bool = False,
    ) -> None:
        if not local:
            raise ValueError(
                "StandaloneProvider only supports `local=True` in v0.1; "
                "remote standalone clusters are coming in a later release."
            )
        self._workspace = Path(workspace) if workspace else self._default_workspace()
        self._workspace.mkdir(parents=True, exist_ok=True)
        self._engine: DockerEngine = (
            docker_engine if docker_engine is not None else RealDockerEngine()
        )
        self._runtime_image = runtime_image
        self._runtime_extras = tuple(runtime_extras)
        self._project_root = Path(project_root) if project_root else _infer_project_root()
        self._image_built = False
        self._mode = self._select_mode(container)
        self._serve_deploys = serve_deploys
        self.apps: dict[str, FastAPI] = {}
        self._serving: dict[str, subprocess.Popen[bytes]] = {}
        self._containers: dict[str, ContainerHandle] = {}

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _default_workspace() -> Path:
        root = Path(os.environ.get("OPHELIAN_HOME", tempfile.gettempdir())) / "ophelian"
        root.mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(prefix="run-", dir=root))

    def _select_mode(self, requested: ContainerMode) -> str:
        if requested is True:
            if not self._engine.ping():
                raise DockerUnavailableError(
                    "container=True was requested but the Docker engine is not reachable."
                )
            return "container"
        if requested is False:
            return "inprocess"
        # auto
        try:
            return "container" if self._engine.ping() else "inprocess"
        except Exception:  # pragma: no cover - defensive
            return "inprocess"

    @property
    def workspace(self) -> Path:
        return self._workspace

    @property
    def engine(self) -> DockerEngine:
        return self._engine

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def runtime_image(self) -> str:
        return self._runtime_image

    @property
    def containers(self) -> dict[str, ContainerHandle]:
        return self._containers

    # ------------------------------------------------------------------
    # Pipeline execution
    # ------------------------------------------------------------------

    def execute(self, pipeline: Pipeline, plan: ExecutionPlan) -> PipelineResult:
        with bind_run(getattr(self, "_run_id", None) or pipeline.name):
            return self._execute_bound(pipeline, plan)

    def _execute_bound(self, pipeline: Pipeline, plan: ExecutionPlan) -> PipelineResult:
        # In container mode, pre-extend the runtime extras with whatever
        # frameworks the pipeline actually uses so the image we build can
        # `import` the relevant adapter dependencies.
        if self._mode == "container":
            self._extend_runtime_extras_from_plan(plan)
        from ophelian.observability.events import (
            StepCompleted,
            StepFailed,
            StepStarted,
        )
        from ophelian.observability.events import (
            emit as emit_lifecycle,
        )
        from ophelian.observability.otel import (
            ATTR_STATUS,
            record_step_outcome,
            step_span,
        )

        results: list[StepResult] = []
        artifact_index: dict[str, dict[str, str]] = {}
        run_id_for_span = getattr(self, "_run_id", None) or pipeline.name
        # Propagate the pipeline's opaque context dict through every
        # emitted step event so subscribers see consistent labels.
        # Stash it on the instance too so ``_run_step_in_container``
        # can encode it into the worker's ``step.json``.
        pipeline_context = getattr(pipeline, "context", None)
        self._current_pipeline_context = pipeline_context
        try:
            for step in plan.steps:
                _step_t0 = time.monotonic()
                # We open the step span ourselves but suppress its
                # built-in metric emission (record_step_outcome is
                # called manually below) because we want the metric
                # tagged with the actual StepResult.status — including
                # the case where the handler raised but we converted
                # it into a failed StepResult rather than re-raising.
                with step_span(
                    step_name=step.name,
                    step_kind=step.kind,
                    run_id=str(run_id_for_span),
                    provider=self.name,
                    emit_metric=False,
                ) as _otel_step_span:
                    # Lifecycle: step_started — inside the step span
                    # so the OTel mirror attaches to it.
                    emit_lifecycle(
                        StepStarted(
                            source=step.name,
                            run_id=str(run_id_for_span),
                            context=pipeline_context,
                            step_name=step.name,
                            step_kind=step.kind,
                            provider=self.name,
                        )
                    )
                    try:
                        step_result = self._execute_step(step, artifact_index)
                    except Exception as exc:
                        logger.exception("Step %s failed", step.name)
                        step_result = StepResult(
                            name=step.name,
                            kind=step.kind,
                            status="failed",
                            error=str(exc),
                        )
                    # Stamp the authoritative ``StepResult.status`` on
                    # the span before it closes so downstream traces
                    # reflect handler-converted failures, not just
                    # exception-escape semantics.
                    if _otel_step_span is not None and hasattr(
                        _otel_step_span, "set_attribute"
                    ):
                        _otel_step_span.set_attribute(
                            ATTR_STATUS, step_result.status
                        )
                    if step_result.duration_seconds is None:
                        step_result = step_result.model_copy(
                            update={"duration_seconds": time.monotonic() - _step_t0}
                        )
                    # Lifecycle: step_completed / step_failed — emitted
                    # before the step span closes so OTel can mirror.
                    if step_result.status == "failed":
                        emit_lifecycle(
                            StepFailed(
                                source=step.name,
                                run_id=str(run_id_for_span),
                                context=pipeline_context,
                                step_name=step.name,
                                step_kind=step.kind,
                                provider=self.name,
                                duration_seconds=step_result.duration_seconds,
                                error=step_result.error or "unknown",
                            )
                        )
                    else:
                        emit_lifecycle(
                            StepCompleted(
                                source=step.name,
                                run_id=str(run_id_for_span),
                                context=pipeline_context,
                                step_name=step.name,
                                step_kind=step.kind,
                                provider=self.name,
                                duration_seconds=step_result.duration_seconds or 0.0,
                                status=step_result.status,
                            )
                        )
                # Authoritative metric emission (step_span suppressed
                # its built-in emission via ``emit_metric=False``).
                record_step_outcome(
                    step_name=step.name,
                    step_kind=step.kind,
                    status=step_result.status,
                    duration_seconds=step_result.duration_seconds or 0.0,
                )
                results.append(step_result)
                artifact_index[step.name] = step_result.artifacts
                if step_result.status == "failed":
                    break
            return PipelineResult(pipeline=pipeline.name, steps=results)
        finally:
            try:
                from ophelian.observability.summary import emit_run_summary

                emit_run_summary(
                    provider=self.name,
                    run_id=getattr(self, "_run_id", None) or pipeline.name,
                    pipeline=pipeline.name,
                    steps=results,
                    description=f"standalone[{self._mode}]",
                    hourly_usd=getattr(self, "_router_quote_hourly_usd", None),
                )
            except Exception:  # pragma: no cover - summary is best-effort
                logger.debug("Summary emission failed", exc_info=True)

    def _execute_step(
        self,
        step: PlannedStep,
        artifacts: Mapping[str, Mapping[str, str]],
    ) -> StepResult:
        node = step.node
        step_dir = self._workspace / step.name
        step_dir.mkdir(parents=True, exist_ok=True)

        if self._mode == "container":
            if isinstance(node, Deploy):
                return self._handle_deploy_container(node, step_dir, artifacts)
            return self._run_step_in_container(node, step_dir, artifacts)

        if isinstance(node, Data):
            return self._handle_data(node, step_dir)
        if isinstance(node, Train):
            return self._handle_train(node, step_dir, artifacts)
        if isinstance(node, Tune):
            return self._handle_tune(node, step_dir, artifacts)
        if isinstance(node, Eval):
            return self._handle_eval(node, step_dir, artifacts)
        if isinstance(node, Deploy):
            return self._handle_deploy(node, step_dir, artifacts)
        raise TypeError(f"StandaloneProvider does not know how to execute {type(node).__name__}")

    # ------------------------------------------------------------------
    # In-process handlers
    # ------------------------------------------------------------------

    def _handle_data(self, node: Data, step_dir: Path) -> StepResult:
        manifest = {
            "source": node.source,
            "format": node.format,
            "split": node.split,
            "options": node.options,
        }
        manifest_path = step_dir / "data.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
        from ophelian.data import materialize

        dataset = materialize(node)
        dataset_path = step_dir / "dataset.json"
        dataset_path.write_text(json.dumps(dataset, default=str))
        return StepResult(
            name=node.name,
            kind=node.kind,
            status="success",
            artifacts={"manifest": str(manifest_path), "dataset": str(dataset_path)},
            info={"resolved_source": node.source, "n_samples": len(dataset.get("X", []))},
        )

    def _handle_train(
        self,
        node: Train,
        step_dir: Path,
        artifacts: Mapping[str, Mapping[str, str]],
        *,
        resume_from: Path | None = None,
    ) -> StepResult:
        data_artifacts = artifacts.get(node.data, {})
        dataset = self._load_dataset(data_artifacts)
        from ophelian.models import registry

        adapter = registry.get(node.framework)()
        # ``checkpoint_dir`` is wired into every Train run so adapters
        # that emit periodic checkpoints (e.g. PyTorch) have a stable
        # filesystem location to write into. The EC2 worker's SIGTERM
        # handler uploads this directory to S3 on spot reclamation, and
        # the next attempt downloads it as ``resume_from``.
        checkpoint_dir = step_dir / "checkpoint"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model = adapter.train(
            model=node.model,
            data=dataset,
            hyperparameters=node.hyperparameters,
            epochs=node.epochs,
            batch_size=node.batch_size,
            resume_from=resume_from,
            checkpoint_dir=checkpoint_dir,
        )
        model_dir = step_dir / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = adapter.save(model, model_dir)
        descriptor = {
            "framework": node.framework,
            "model": node.model,
            "hyperparameters": node.hyperparameters,
            "artifact": str(artifact_path),
            "resumed": resume_from is not None,
        }
        (model_dir / "ophelian.json").write_text(json.dumps(descriptor, indent=2, default=str))
        out_artifacts: dict[str, str] = {"model": str(model_dir), "artifact": str(artifact_path)}
        # Drift baseline capture (Task #30). Snapshots the training
        # feature + prediction distributions so a downstream Deploy
        # node can attach drift monitors without manual plumbing.
        if node.capture_baseline:
            baseline_path = _capture_training_baseline(
                node=node,
                adapter=adapter,
                model=model,
                dataset=dataset,
                step_dir=step_dir,
            )
            if baseline_path is not None:
                out_artifacts["baseline"] = str(baseline_path)
        return StepResult(
            name=node.name,
            kind=node.kind,
            status="success",
            artifacts=out_artifacts,
            info={
                "framework": node.framework,
                "mode": self._mode,
                "resumed_from": str(resume_from) if resume_from else "",
            },
        )

    def _handle_tune(
        self,
        node: Tune,
        step_dir: Path,
        artifacts: Mapping[str, Mapping[str, str]],
    ) -> StepResult:
        train_artifacts = artifacts.get(node.train, {})
        report = {
            "strategy": node.strategy,
            "max_trials": node.max_trials,
            "metric": node.metric,
            "direction": node.direction,
            "search_space": node.search_space,
            "base_model": train_artifacts.get("model"),
        }
        best_dir = step_dir / "best_model"
        best_dir.mkdir(parents=True, exist_ok=True)
        if train_artifacts.get("model"):
            for child in Path(train_artifacts["model"]).iterdir():
                shutil.copy2(child, best_dir / child.name)
        (best_dir / "trial.json").write_text(json.dumps(report, indent=2, default=str))
        return StepResult(
            name=node.name,
            kind=node.kind,
            status="success",
            artifacts={"best_model": str(best_dir)},
            info={"mode": self._mode},
        )

    def _handle_eval(
        self,
        node: Eval,
        step_dir: Path,
        artifacts: Mapping[str, Mapping[str, str]],
    ) -> StepResult:
        train_artifacts = artifacts.get(node.model, {})
        data_artifacts = artifacts.get(node.data, {})
        descriptor_path = Path(train_artifacts.get("model", "")) / "ophelian.json"
        if not descriptor_path.exists():
            raise FileNotFoundError(
                f"Eval step {node.name!r} cannot find a trained model under {train_artifacts!r}"
            )
        descriptor = json.loads(descriptor_path.read_text())
        from ophelian.models import registry

        adapter = registry.get(descriptor["framework"])()
        model = adapter.load(Path(train_artifacts["model"]))
        dataset = self._load_dataset(data_artifacts)
        predictions = adapter.predict(model, dataset["X"])
        metrics = _compute_metrics(node.metrics, dataset["y"], predictions)
        report_path = step_dir / "report.json"
        report_path.write_text(json.dumps(metrics, indent=2))
        return StepResult(
            name=node.name,
            kind=node.kind,
            status="success",
            metrics=metrics,
            artifacts={"report": str(report_path)},
            info={"framework": descriptor["framework"], "mode": self._mode},
        )

    def _handle_deploy(
        self,
        node: Deploy,
        step_dir: Path,
        artifacts: Mapping[str, Mapping[str, str]],
    ) -> StepResult:
        del step_dir
        train_artifacts = artifacts.get(node.model, {})
        model_path = train_artifacts.get("model")
        if not model_path:
            raise ValueError(
                f"Deploy step {node.name!r} has no upstream model artifact from {node.model!r}"
            )
        descriptor_path = Path(model_path) / "ophelian.json"
        descriptor = json.loads(descriptor_path.read_text())
        from ophelian.runtime.fastapi_runtime import build_app

        # Resolve the drift baseline (Task #30) before constructing the
        # app so monitor attachment is part of app construction rather
        # than a follow-up step external code might forget to invoke.
        baseline_path: str | None = None
        if node.drift_baseline == "auto":
            baseline_path = train_artifacts.get("baseline")
            if baseline_path is None:
                raise ValueError(
                    f"Deploy step {node.name!r} requested drift_baseline='auto' "
                    f"but upstream Train step {node.model!r} did not publish a "
                    f"'baseline' artifact — set capture_baseline=True on the Train node."
                )
        elif node.drift_baseline:
            baseline_path = node.drift_baseline

        app = build_app(
            framework=descriptor["framework"],
            model_path=model_path,
            drift_baseline_path=baseline_path,
            drift_endpoint=node.name if baseline_path else None,
            drift_window_size=node.drift_window_size,
            drift_stride=node.drift_stride,
            drift_test=node.drift_test,
            drift_threshold=node.drift_threshold,
        )
        self.apps[node.name] = app

        url = f"http://localhost:{node.port}"
        info: dict[str, Any] = {
            "framework": descriptor["framework"],
            "mode": self._mode,
            "predict": f"{url}/predict",
            "health": f"{url}/health",
            "replicas": node.replicas,
        }
        artifacts_out = {"endpoint_url": url, "model": str(model_path)}
        if self._serve_deploys:  # pragma: no cover - exercised manually
            self._serving[node.name] = self._spawn_uvicorn(
                node, descriptor["framework"], model_path
            )
            info["served"] = True
        else:
            info["served"] = False
            info["hint"] = (
                "App built but not served — call provider.serve(step_name) "
                "or instantiate StandaloneProvider(serve_deploys=True)."
            )
        return StepResult(
            name=node.name,
            kind=node.kind,
            status="success",
            artifacts=artifacts_out,
            info=info,
        )

    # ------------------------------------------------------------------
    # Container handlers
    # ------------------------------------------------------------------

    def _extend_runtime_extras_from_plan(self, plan: ExecutionPlan) -> None:
        """Add the framework of each Train step to the runtime extras.

        Keeps the default image small (sklearn-only) but expands it on
        demand so pipelines using ``pytorch``/``xgboost``/``huggingface``
        get an image that can actually import their adapter.
        """
        wanted: set[str] = set(self._runtime_extras)
        for step in plan.steps:
            framework = getattr(step.node, "framework", None)
            if framework:
                wanted.add(framework)
        new_extras = tuple(sorted(wanted))
        if new_extras != self._runtime_extras:
            # Force a rebuild on first run after the extras change.
            self._image_built = False
            self._runtime_extras = new_extras

    def _ensure_runtime_image(self) -> None:
        if self._image_built:
            return
        # Use the project root as build context so the Dockerfile can install
        # the local ophelian package (`pip install .`) without any registry.
        self._engine.build_image(
            context=self._project_root,
            dockerfile=_runtime_dockerfile(self._runtime_extras),
            tag=self._runtime_image,
        )
        self._image_built = True

    def _run_step_in_container(
        self,
        node: Any,
        step_dir: Path,
        artifacts: Mapping[str, Mapping[str, str]],
    ) -> StepResult:
        """Run a Data/Train/Tune/Eval node inside a container."""
        self._ensure_runtime_image()
        kind = node.kind

        # Prepare upstream artifact references — translate host paths under the
        # workspace into the in-container `/work-*` mounts so the step runner
        # can find them.
        rewritten_artifacts: dict[str, dict[str, str]] = {}
        extra_volumes: dict[Path, str] = {}
        for upstream_name, upstream_artifacts in artifacts.items():
            mount = f"/work-{upstream_name}"
            extra_volumes[self._workspace / upstream_name] = mount
            rewritten_artifacts[upstream_name] = {
                k: _rewrite_path(v, self._workspace / upstream_name, mount)
                for k, v in upstream_artifacts.items()
            }

        spec: dict[str, Any] = {
            "kind": kind,
            "node": node.model_dump(mode="json"),
            "artifacts": rewritten_artifacts,
        }
        # Propagate the owning pipeline's context dict into the
        # container's step spec so the in-container ``step_runner``
        # tags the lifecycle events it emits with the same labels as
        # in-process execution. ``_current_pipeline_context`` is set
        # by ``StandaloneProvider.run`` for the duration of the run.
        pipeline_context = getattr(self, "_current_pipeline_context", None)
        if pipeline_context:
            spec["context"] = dict(pipeline_context)
        spec_path = step_dir / "step.json"
        spec_path.write_text(json.dumps(spec, default=str))

        volumes: dict[Path, str] = {step_dir: "/work", **extra_volumes}
        handle = self._engine.run_container(
            image=self._runtime_image,
            command=["python", "-m", "ophelian.runtime.step_runner", "/work/step.json"],
            volumes=volumes,
            environment={"PYTHONUNBUFFERED": "1"},
            name=f"ophelian-{node.name}",
        )
        if handle.exit_code != 0:
            return StepResult(
                name=node.name,
                kind=kind,
                status="failed",
                error=f"container exited with {handle.exit_code}: {handle.logs.strip()[-500:]}",
            )
        result_path = step_dir / "result.json"
        if not result_path.exists():
            return StepResult(
                name=node.name,
                kind=kind,
                status="failed",
                error="container did not produce result.json",
            )
        payload = json.loads(result_path.read_text())
        # Translate paths in the result back to host paths so downstream steps
        # find their inputs through the host-side workspace.
        host_artifacts = {
            k: _rewrite_path(v, "/work", str(step_dir))
            for k, v in (payload.get("artifacts") or {}).items()
        }
        info = dict(payload.get("info") or {})
        info["mode"] = "container"
        info["container_id"] = handle.container_id
        return StepResult(
            name=payload["name"],
            kind=payload["kind"],
            status=payload["status"],
            metrics=payload.get("metrics") or {},
            artifacts=host_artifacts,
            info=info,
            error=payload.get("error"),
        )

    def _handle_deploy_container(
        self,
        node: Deploy,
        step_dir: Path,
        artifacts: Mapping[str, Mapping[str, str]],
    ) -> StepResult:
        self._ensure_runtime_image()
        train_artifacts = artifacts.get(node.model, {})
        model_path_host = train_artifacts.get("model")
        if not model_path_host:
            raise ValueError(
                f"Deploy step {node.name!r} has no upstream model artifact from {node.model!r}"
            )
        descriptor_path = Path(model_path_host) / "ophelian.json"
        descriptor = json.loads(descriptor_path.read_text())

        host_port = _pick_free_port()
        container_port = node.port
        container_model_dir = "/model"
        # Drift baseline (Task #30) — resolve and stage the same way
        # the in-process deploy path does so the container deploy
        # has parity. We mount the baseline file into the container
        # at a stable path and pass drift config via env vars
        # consumed by ``app_from_env``.
        host_baseline: str | None = None
        if node.drift_baseline == "auto":
            host_baseline = train_artifacts.get("baseline")
            if host_baseline is None:
                raise ValueError(
                    f"Deploy step {node.name!r} requested drift_baseline='auto' "
                    f"but upstream Train step {node.model!r} did not publish a "
                    f"'baseline' artifact — set capture_baseline=True on the Train node."
                )
        elif node.drift_baseline:
            host_baseline = node.drift_baseline
        # ``model_path_host`` already points to the per-step ``model/`` dir
        # produced by Train (it contains ``ophelian.json`` and the artifact
        # file). Mount THAT directly at /model so the in-container adapter
        # finds files where it expects them — not the parent step dir.
        host_model_dir = Path(model_path_host)

        container_name = f"ophelian-deploy-{node.name}-{host_port}"
        handle = self._engine.run_container(
            image=self._runtime_image,
            command=[
                "python",
                "-m",
                "uvicorn",
                "--factory",
                "ophelian.runtime.fastapi_runtime:app_from_env",
                "--host",
                "0.0.0.0",
                "--port",
                str(container_port),
            ],
            volumes=(
                {host_model_dir: container_model_dir, Path(host_baseline): "/baseline.json"}
                if host_baseline
                else {host_model_dir: container_model_dir}
            ),
            environment=_deploy_container_env(
                framework=descriptor["framework"],
                model_dir=container_model_dir,
                node=node,
                baseline_in_container="/baseline.json" if host_baseline else None,
            ),
            ports={host_port: container_port},
            detach=True,
            name=container_name,
        )
        self._containers[node.name] = handle

        url = f"http://localhost:{host_port}"
        info: dict[str, Any] = {
            "framework": descriptor["framework"],
            "mode": "container",
            "predict": f"{url}/predict",
            "health": f"{url}/health",
            "host_port": host_port,
            "container_port": container_port,
            "container_id": handle.container_id,
            "container_name": container_name,
            "replicas": node.replicas,
            "served": True,
        }
        artifacts_out = {
            "endpoint_url": url,
            "model": str(model_path_host),
            "container_id": handle.container_id,
        }
        # Best-effort readiness wait — only when running against a real daemon.
        if isinstance(self._engine, RealDockerEngine):  # pragma: no cover - real docker only
            _wait_for_http("127.0.0.1", host_port, timeout=20.0)
        del step_dir
        return StepResult(
            name=node.name,
            kind=node.kind,
            status="success",
            artifacts=artifacts_out,
            info=info,
        )

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------

    def serve(
        self, step_name: str, *, host: str = "0.0.0.0", port: int = 8000
    ) -> None:  # pragma: no cover
        import uvicorn

        app = self.apps[step_name]
        uvicorn.run(app, host=host, port=port)

    def _spawn_uvicorn(
        self, node: Deploy, framework: str, model_path: str
    ) -> subprocess.Popen[bytes]:  # pragma: no cover
        cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            "--factory",
            "ophelian.runtime.fastapi_runtime:build_app",
            "--host",
            "0.0.0.0",
            "--port",
            str(node.port),
        ]
        env = {
            **os.environ,
            "OPHELIAN_FRAMEWORK": framework,
            "OPHELIAN_MODEL_PATH": model_path,
        }
        return subprocess.Popen(cmd, env=env)

    def _load_dataset(self, data_artifacts: Mapping[str, str]) -> dict[str, Any]:
        dataset_path = data_artifacts.get("dataset")
        if not dataset_path:
            raise ValueError(
                "Train/Eval step is missing an upstream Data artifact — make sure the "
                "Data node ran successfully before this step."
            )
        payload: dict[str, Any] = json.loads(Path(dataset_path).read_text())
        return payload

    def cleanup(self) -> None:
        """Remove the workspace directory and stop any containers we own."""
        for proc in self._serving.values():  # pragma: no cover
            proc.terminate()
        self._serving.clear()
        for handle in self._containers.values():
            try:
                self._engine.stop(handle.container_id)
            finally:
                self._engine.remove(handle.container_id, force=True)
        self._containers.clear()
        shutil.rmtree(self._workspace, ignore_errors=True)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _infer_project_root() -> Path:
    here = Path(__file__).resolve()
    # ophelian/providers/standalone.py → repo root is two levels up.
    return here.parent.parent.parent


def _rewrite_path(value: str, src_prefix: Path | str, dst_prefix: Path | str) -> str:
    src = str(src_prefix)
    dst = str(dst_prefix)
    if value.startswith(src):
        return dst + value[len(src) :]
    return value


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_for_http(host: str, port: int, *, timeout: float) -> bool:  # pragma: no cover
    """Wait for a TCP port to accept connections; used after starting Deploy."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(0.5)
    return False


def _deploy_container_env(
    *,
    framework: str,
    model_dir: str,
    node: Deploy,
    baseline_in_container: str | None,
) -> dict[str, str]:
    """Env contract consumed by ``app_from_env`` inside the deploy container.

    Mirrors the in-process ``build_app`` drift parameters so opt-in
    drift behaviour is identical across deploy modes (Task #30
    parity finding).
    """
    env: dict[str, str] = {
        "OPHELIAN_FRAMEWORK": framework,
        "OPHELIAN_MODEL_PATH": model_dir,
    }
    if baseline_in_container:
        env["OPHELIAN_DRIFT_BASELINE_PATH"] = baseline_in_container
        env["OPHELIAN_DRIFT_ENDPOINT"] = node.name
        env["OPHELIAN_DRIFT_WINDOW_SIZE"] = str(node.drift_window_size)
        if node.drift_stride is not None:
            env["OPHELIAN_DRIFT_STRIDE"] = str(node.drift_stride)
        env["OPHELIAN_DRIFT_TEST"] = node.drift_test
        env["OPHELIAN_DRIFT_THRESHOLD"] = str(node.drift_threshold)
    return env


def _capture_training_baseline(
    *,
    node: Train,
    adapter: Any,
    model: Any,
    dataset: Mapping[str, Any],
    step_dir: Path,
) -> Path | None:
    """Snapshot the training distribution as a drift baseline (Task #30).

    Best-effort: if the dataset shape is unfamiliar (rows that are
    not dicts/lists or features without names) the function returns
    ``None`` and training still succeeds — drift is opt-in and a
    missing baseline simply means no monitor will attach later.
    """
    from ophelian.observability.drift import Baseline

    rows = dataset.get("X")
    if not rows:
        return None
    feature_samples: dict[str, list[Any]] = {}
    if isinstance(rows[0], Mapping):
        keys = list(rows[0].keys())
        for k in keys:
            feature_samples[k] = [row.get(k) for row in rows if isinstance(row, Mapping)]
    elif isinstance(rows[0], (list, tuple)):
        width = len(rows[0])
        names = (
            list(node.baseline_features)
            if node.baseline_features
            else [f"f{i}" for i in range(width)]
        )
        for i, name in enumerate(names[:width]):
            feature_samples[name] = [row[i] for row in rows if i < len(row)]
    else:
        return None
    if node.baseline_features is not None:
        feature_samples = {
            k: v for k, v in feature_samples.items() if k in set(node.baseline_features)
        }
    prediction_samples: list[Any] | None = None
    try:
        preds = adapter.predict(model, rows)
        if isinstance(preds, (list, tuple)):
            prediction_samples = [p for p in preds if not isinstance(p, (list, tuple, Mapping))]
            if not prediction_samples:
                prediction_samples = None
    except Exception:  # pragma: no cover - defensive: prediction baseline is optional
        prediction_samples = None
    baseline = Baseline.from_samples(
        feature_samples=feature_samples or None,
        prediction_samples=prediction_samples,
        metadata={
            "framework": node.framework,
            "model": node.model,
            "n_samples": len(rows),
            "captured_by": "standalone._handle_train",
        },
    )
    out = step_dir / "baseline.json"
    return baseline.save(out)


def _compute_metrics(
    metric_names: tuple[str, ...],
    y_true: list[Any],
    y_pred: list[Any],
) -> dict[str, float]:
    out: dict[str, float] = {}
    for metric in metric_names:
        key = metric.lower()
        if key == "accuracy":
            if not y_true:
                out[metric] = 0.0
            else:
                correct = sum(1 for a, b in zip(y_true, y_pred, strict=False) if a == b)
                out[metric] = correct / len(y_true)
        else:
            try:  # pragma: no cover
                from sklearn import metrics as sk_metrics

                fn = getattr(sk_metrics, f"{key}_score", None)
                out[metric] = float(fn(y_true, y_pred)) if fn else 0.0
            except ImportError:  # pragma: no cover
                out[metric] = 0.0
    return out
