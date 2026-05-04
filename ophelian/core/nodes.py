"""Declarative DAG nodes that make up a Pipeline.

Every node is an immutable Pydantic v2 model. Pipelines are *descriptions*, not
imperative programs: they can be inspected, serialised and compiled to an
execution plan before anything actually runs. This is what enables the same
Pipeline to be executed locally with Docker today and on a cloud provider
tomorrow without changing the user's code.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

if TYPE_CHECKING:
    from ophelian.providers.base import Provider


_FROZEN = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)


class Node(BaseModel):
    """Base class for every DAG node."""

    model_config = _FROZEN

    name: str = Field(default_factory=lambda: f"node-{uuid4().hex[:8]}")
    """Human-friendly node identifier — also acts as the DAG handle."""

    depends_on: tuple[str, ...] = Field(default_factory=tuple)
    """Names of upstream nodes this one depends on."""

    @property
    def kind(self) -> str:
        """Short string used by the compiler / renderer to identify the node."""
        return type(self).__name__.lower()


class Data(Node):
    """Describes a dataset input to the pipeline."""

    source: str
    """URI or local path the runtime knows how to resolve (s3://..., file://..., hf://...)."""

    format: Literal[
        "parquet",
        "csv",
        "json",
        "jsonl",
        "image-folder",
        "huggingface",
        "inline",
        "synthetic",
    ] = "parquet"
    split: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)


class Train(Node):
    """Train a model with a specific framework adapter."""

    framework: Literal["pytorch", "huggingface", "sklearn", "xgboost"]
    model: str
    """Model identifier — adapter-specific (e.g. `resnet18`, `bert-base-uncased`, `RandomForestClassifier`)."""

    data: str
    """Name of the upstream `Data` node feeding this trainer."""

    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    epochs: int | None = None
    batch_size: int | None = None
    output: str = "model"
    """Logical artifact name the trained model is written to."""

    capture_baseline: bool = False
    """Snapshot the training feature + prediction distributions as a
    drift-detection baseline artifact (Task #30). When ``True`` the
    standalone provider writes ``baseline.json`` next to the model and
    publishes it as the ``baseline`` artifact key — a downstream
    :class:`Deploy` node with ``drift_baseline="auto"`` will pick it
    up and attach drift monitors to the served endpoint automatically.
    """

    baseline_features: tuple[str, ...] | None = None
    """Optional explicit feature subset to capture in the baseline.
    ``None`` (default) captures every column the trainer sees."""

    @field_validator("framework")
    @classmethod
    def _normalise_framework(cls, value: str) -> str:
        return value.lower()

    @model_validator(mode="after")
    def _wire_data_dependency(self) -> Train:
        if self.data not in self.depends_on:
            object.__setattr__(self, "depends_on", (*self.depends_on, self.data))
        return self


class Tune(Node):
    """Hyperparameter search over a base `Train` step."""

    train: str
    """Name of the upstream `Train` node to optimise."""

    search_space: dict[str, Any]
    strategy: Literal["grid", "random", "bayes"] = "random"
    max_trials: int = Field(default=10, ge=1)
    metric: str = "loss"
    direction: Literal["minimize", "maximize"] = "minimize"

    @model_validator(mode="after")
    def _wire_train_dependency(self) -> Tune:
        if self.train not in self.depends_on:
            object.__setattr__(self, "depends_on", (*self.depends_on, self.train))
        return self


class Eval(Node):
    """Evaluate a trained model against a dataset."""

    model: str
    """Name of the upstream `Train` (or `Tune`) node."""

    data: str
    """Name of the upstream `Data` node holding the eval split."""

    metrics: tuple[str, ...] = ("accuracy",)

    @field_validator("metrics")
    @classmethod
    def _no_empty_metrics(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("Eval requires at least one metric")
        return value

    @model_validator(mode="after")
    def _wire_dependencies(self) -> Eval:
        deps = list(self.depends_on)
        for name in (self.model, self.data):
            if name not in deps:
                deps.append(name)
        object.__setattr__(self, "depends_on", tuple(deps))
        return self


class Deploy(Node):
    """Serve a trained model behind a FastAPI endpoint."""

    model: str
    """Name of the upstream `Train` (or `Tune`) node to deploy."""

    runtime: Literal["fastapi"] = "fastapi"
    port: int = Field(default=8000, ge=1, le=65535)
    replicas: int = Field(default=1, ge=1)
    autoscale: bool = False

    drift_baseline: str | None = None
    """Drift baseline source (Task #30). ``None`` (default) disables
    drift hooks. ``"auto"`` resolves to the ``baseline`` artifact
    produced by the upstream :class:`Train` step (which must have
    ``capture_baseline=True``). Any other string is treated as an
    explicit filesystem path to a ``baseline.json``.
    """

    drift_window_size: int = Field(default=200, ge=2)
    drift_stride: int | None = None
    drift_test: Literal["ks", "chi2", "psi"] = "ks"
    drift_threshold: float = Field(default=0.05, gt=0.0)

    @model_validator(mode="after")
    def _wire_model_dependency(self) -> Deploy:
        if self.model not in self.depends_on:
            object.__setattr__(self, "depends_on", (*self.depends_on, self.model))
        return self


class StepResult(BaseModel):
    """Outcome of executing a single node."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    name: str
    kind: str
    status: Literal["success", "skipped", "failed"]
    artifacts: dict[str, str] = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)
    info: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    duration_seconds: float | None = None


class PipelineResult(BaseModel):
    """Aggregate result returned by `Pipeline.run`."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    pipeline: str
    steps: list[StepResult] = Field(default_factory=list)

    @property
    def succeeded(self) -> bool:
        return all(step.status != "failed" for step in self.steps)

    def step(self, name: str) -> StepResult:
        for step in self.steps:
            if step.name == name:
                return step
        raise KeyError(f"No step named {name!r} in pipeline result")


class Pipeline(BaseModel):
    """Top-level container that composes nodes into a DAG."""

    model_config = _FROZEN

    name: str = Field(default_factory=lambda: f"pipeline-{uuid4().hex[:8]}")
    steps: tuple[Node, ...]
    description: str | None = None
    context: dict[str, Any] | None = None
    """Opaque dict propagated to every lifecycle event emitted during
    this pipeline's run. Multi-tenant callers stash ``tenant_id`` and
    similar labels here so subscribers can route handlers without
    additional plumbing. The cost ledger task also reads from this
    dict to attribute spend rows. ``None`` (the default) means the
    field is simply absent from emitted events.
    """

    def __init__(self, steps: Sequence[Node] | None = None, **data: Any) -> None:
        if steps is not None and "steps" not in data:
            data["steps"] = tuple(steps)
        super().__init__(**data)

    @field_validator("steps")
    @classmethod
    def _validate_steps(cls, value: tuple[Node, ...]) -> tuple[Node, ...]:
        if not value:
            raise ValueError("A Pipeline must contain at least one step")
        seen: set[str] = set()
        for node in value:
            if node.name in seen:
                raise ValueError(f"Duplicate node name in pipeline: {node.name!r}")
            seen.add(node.name)
        for node in value:
            for dep in node.depends_on:
                if dep not in seen:
                    raise ValueError(f"Node {node.name!r} depends on unknown node {dep!r}")
        return value

    def run(
        self,
        env: Provider,
        *,
        dry_run: bool = False,
    ) -> PipelineResult:
        """Compile the pipeline and execute it on the given provider."""
        from ophelian.core.compiler import GraphCompiler
        from ophelian.observability.events import (
            PipelineCompleted,
            PipelineStarted,
        )
        from ophelian.observability.events import (
            emit as emit_lifecycle,
        )
        from ophelian.observability.otel import (
            ATTR_RUN_ID,
            ATTR_STATUS,
            pipeline_span,
            record_pipeline_outcome,
        )

        plan = GraphCompiler().compile(self)
        if dry_run:
            plan.render()
            return PipelineResult(pipeline=self.name)

        env_class = getattr(env, "name", "unknown") or "unknown"
        provider_name = env_class
        region = (
            getattr(env, "_region", None)
            or getattr(getattr(env, "_config", None), "region", None)
            or getattr(getattr(env, "_config", None), "location", None)
            or "local"
        )
        # We open the span ourselves but suppress its built-in metric
        # emission; we want the ``ophelian.pipeline.runs`` counter
        # tagged with the authoritative ``PipelineResult.succeeded``
        # rather than mere exception-escape semantics.
        status: str = "failed"
        result_for_ledger: PipelineResult | None = None
        try:
            with pipeline_span(
                pipeline_name=self.name,
                run_id=getattr(env, "_run_id", None),
                env_class=env_class,
                provider=provider_name,
                region=str(region) if region is not None else None,
                emit_metric=False,
            ) as span:
                # Lifecycle: pipeline_started — fired INSIDE the
                # pipeline span so the OTel mirror lands on it.
                emit_lifecycle(
                    PipelineStarted(
                        source=self.name,
                        run_id=str(getattr(env, "_run_id", None) or ""),
                        context=self.context,
                        pipeline=self.name,
                        provider=provider_name,
                        env_class=env_class,
                    )
                )
                try:
                    result = env.execute(self, plan)
                    result_for_ledger = result
                    status = "success" if result.succeeded else "failed"
                    if span is not None and hasattr(span, "set_attribute"):
                        final_run_id = getattr(env, "_run_id", None)
                        if final_run_id:
                            span.set_attribute(ATTR_RUN_ID, str(final_run_id))
                        span.set_attribute(ATTR_STATUS, status)
                    return result
                finally:
                    # Lifecycle: pipeline_completed — fired in finally
                    # so it runs on both success and exception paths,
                    # while still inside the pipeline span.
                    emit_lifecycle(
                        PipelineCompleted(
                            source=self.name,
                            run_id=str(getattr(env, "_run_id", None) or ""),
                            context=self.context,
                            pipeline=self.name,
                            provider=provider_name,
                            status=status,
                        )
                    )
        finally:
            record_pipeline_outcome(
                pipeline_name=self.name,
                status=status,
                provider=provider_name,
            )
            # Cost ledger row (Task #28). Best-effort: never raises.
            try:
                from ophelian.observability.ledger import emit_run_row

                if result_for_ledger is not None:
                    hours = sum(
                        (s.duration_seconds or 0.0) for s in result_for_ledger.steps
                    ) / 3600.0
                else:
                    hours = 0.0
                emit_run_row(
                    pipeline_name=self.name,
                    run_id=getattr(env, "_run_id", None)
                    or getattr(env, "_last_run_id", None),
                    env_class=env_class,
                    provider=provider_name,
                    region=str(region) if region is not None else None,
                    gpu_type=getattr(getattr(env, "_config", None), "gpu_type", None),
                    instance=getattr(getattr(env, "_config", None), "instance", None)
                    or getattr(getattr(env, "_config", None), "vm_size", None),
                    hours=hours,
                    hourly_usd=getattr(env, "_router_quote_hourly_usd", None),
                    status=status,
                    context=self.context,
                )
            except Exception:  # pragma: no cover - ledger must never break runs
                pass

    def dry_run(self) -> None:
        """Pretty-print the execution plan without running anything."""
        from ophelian.core.compiler import GraphCompiler

        GraphCompiler().compile(self).render()
