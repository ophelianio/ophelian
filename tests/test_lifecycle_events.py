"""Tests for the lifecycle events bus (Task #29).

Covers:

* the bus contract (subscribe, emit, unsubscribe, handler isolation
  under exceptions, OTel span-event mirroring);
* the canonical event hierarchy (every documented type is
  constructible and dispatched);
* end-to-end wiring into ``Pipeline.run`` (pipeline_started,
  pipeline_completed), the standalone executor (step_started,
  step_completed, step_failed), the FastAPI runtime (model_loaded,
  model_unloaded, inference_failed), and the spot probe
  (spot_interruption_received);
* propagation of ``Pipeline.context`` through every emitted event.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar

import pytest
from fastapi.testclient import TestClient
from ophelian import Data, Pipeline, Standalone
from ophelian.models.base import ModelAdapter, register_adapter
from ophelian.observability.events import (
    InferenceFailed,
    LifecycleEvent,
    ModelLoaded,
    ModelSwapped,
    ModelUnloaded,
    PipelineCompleted,
    PipelineStarted,
    SpotInterruptionReceived,
    StepCompleted,
    StepFailed,
    StepStarted,
    _clear_subscribers_for_tests,
    emit,
    on_event,
)

# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture()
def otel(_otel_in_memory_providers: dict[str, Any]) -> Iterator[dict[str, Any]]:
    _otel_in_memory_providers["spans"].clear()
    yield _otel_in_memory_providers


@pytest.fixture()
def captured() -> Iterator[list[LifecycleEvent]]:
    """Yield a list that captures every event emitted during the test.

    Subscribers are cleared at both setup and teardown so handlers
    registered by one test never bleed into another.
    """
    _clear_subscribers_for_tests()
    events: list[LifecycleEvent] = []
    on_event(events.append)
    try:
        yield events
    finally:
        _clear_subscribers_for_tests()


# ----------------------------------------------------------------------
# Bus contract
# ----------------------------------------------------------------------


def test_subscribe_emit_unsubscribe(captured: list[LifecycleEvent]) -> None:
    emit(
        ModelSwapped(
            source="t",
            framework="x",
            previous_model_path="/a",
            new_model_path="/b",
        )
    )
    assert len(captured) == 1
    assert isinstance(captured[0], ModelSwapped)


def test_unsubscribe_stops_delivery() -> None:
    _clear_subscribers_for_tests()
    seen: list[LifecycleEvent] = []
    unsub = on_event(seen.append)
    emit(
        ModelSwapped(
            source="t",
            framework="x",
            previous_model_path="/a",
            new_model_path="/b",
        )
    )
    unsub()
    emit(
        ModelSwapped(
            source="t",
            framework="x",
            previous_model_path="/c",
            new_model_path="/d",
        )
    )
    assert len(seen) == 1


def test_handler_exception_does_not_propagate(captured: list[LifecycleEvent]) -> None:
    """Subscriber failures are logged at WARN and never break the
    emitting code path nor sibling subscribers."""

    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _Capture(level=logging.WARNING)
    target = logging.getLogger("ophelian.observability.events")
    prior_level = target.level
    target.addHandler(handler)
    target.setLevel(logging.WARNING)
    try:

        def boom(_: LifecycleEvent) -> None:
            raise RuntimeError("subscriber boom")

        on_event(boom)
        emit(
            ModelSwapped(
                source="t",
                framework="x",
                previous_model_path="/a",
                new_model_path="/b",
            )
        )
    finally:
        target.removeHandler(handler)
        target.setLevel(prior_level)
    # Other subscribers still received the event.
    assert len(captured) == 1
    # Failure logged at WARN with handler identity.
    assert any("Lifecycle handler" in r.getMessage() for r in records)


def test_otel_span_event_mirroring(otel: dict[str, Any], captured: list[LifecycleEvent]) -> None:
    from ophelian.observability.otel import pipeline_span

    with pipeline_span(pipeline_name="lc-mirror", run_id="r1"):
        emit(
            ModelLoaded(
                source="t",
                run_id="r1",
                framework="sklearn",
                model_path="/m",
            )
        )
    spans = otel["spans"].get_finished_spans()
    pipeline = [s for s in spans if "pipeline" in s.name]
    assert pipeline, [s.name for s in spans]
    events = pipeline[-1].events
    assert any(e.name == "ophelian.model.loaded" for e in events), [e.name for e in events]


# ----------------------------------------------------------------------
# Canonical contract: every documented event class is constructible and
# is delivered to subscribers without runtime support.
# ----------------------------------------------------------------------


_EVENT_FACTORIES: list[tuple[str, Any]] = [
    ("model_loaded", lambda: ModelLoaded(source="x", framework="sklearn", model_path="/m")),
    ("model_unloaded", lambda: ModelUnloaded(source="x", framework="sklearn", model_path="/m")),
    (
        "model_swapped",
        lambda: ModelSwapped(
            source="x",
            framework="sklearn",
            previous_model_path="/a",
            new_model_path="/b",
        ),
    ),
    (
        "inference_failed",
        lambda: InferenceFailed(
            source="x", framework="sklearn", route="/predict", method="POST", error="boom"
        ),
    ),
    (
        "pipeline_started",
        lambda: PipelineStarted(
            source="p", pipeline="p", provider="standalone", env_class="standalone"
        ),
    ),
    (
        "pipeline_completed",
        lambda: PipelineCompleted(
            source="p", pipeline="p", provider="standalone", status="success"
        ),
    ),
    (
        "step_started",
        lambda: StepStarted(source="s", step_name="s", step_kind="data", provider="standalone"),
    ),
    (
        "step_completed",
        lambda: StepCompleted(
            source="s",
            step_name="s",
            step_kind="data",
            provider="standalone",
            duration_seconds=0.1,
            status="success",
        ),
    ),
    (
        "step_failed",
        lambda: StepFailed(
            source="s",
            step_name="s",
            step_kind="data",
            provider="standalone",
            duration_seconds=0.1,
            error="boom",
        ),
    ),
    ("spot_interruption_received", lambda: SpotInterruptionReceived(source="spot")),
]


@pytest.mark.parametrize("name,factory", _EVENT_FACTORIES, ids=[n for n, _ in _EVENT_FACTORIES])
def test_event_class_constructible_and_dispatched(
    captured: list[LifecycleEvent], name: str, factory: Any
) -> None:
    del name
    ev = factory()
    emit(ev)
    assert captured == [ev]


# ----------------------------------------------------------------------
# Pipeline + step lifecycle wiring (in-process standalone executor)
# ----------------------------------------------------------------------


def _toy_pipeline(
    pipe_name: str = "lifecycle-test",
    *,
    context: dict[str, Any] | None = None,
) -> Pipeline:
    kwargs: dict[str, Any] = {"name": pipe_name}
    if context is not None:
        kwargs["context"] = context
    return Pipeline(
        [
            Data(
                name="ds",
                source="memory://toy",
                format="inline",
                options={"X": [[1.0, 2.0]], "y": [0, 1]},
            )
        ],
        **kwargs,
    )


def test_pipeline_started_and_completed_fire(captured: list[LifecycleEvent]) -> None:
    _toy_pipeline().run(env=Standalone(local=True, container=False))
    started = [e for e in captured if isinstance(e, PipelineStarted)]
    completed = [e for e in captured if isinstance(e, PipelineCompleted)]
    assert len(started) == 1, [type(e).__name__ for e in captured]
    assert len(completed) == 1
    assert completed[0].status == "success"
    assert completed[0].pipeline == "lifecycle-test"


def test_step_started_and_completed_fire(captured: list[LifecycleEvent]) -> None:
    _toy_pipeline().run(env=Standalone(local=True, container=False))
    starts = [e for e in captured if isinstance(e, StepStarted)]
    completes = [e for e in captured if isinstance(e, StepCompleted)]
    assert len(starts) == 1
    assert len(completes) == 1
    assert starts[0].step_name == "ds"
    assert completes[0].step_kind == "data"
    assert completes[0].duration_seconds >= 0.0


def test_step_failed_fires_when_handler_raises(
    captured: list[LifecycleEvent], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Force the in-process Data handler to raise; verify the
    executor catches it, builds a failed StepResult, and emits
    StepFailed (not StepCompleted)."""
    from ophelian.providers.standalone import StandaloneProvider

    def boom(self: Any, *_a: Any, **_kw: Any) -> Any:
        raise RuntimeError("synthetic step failure")

    monkeypatch.setattr(StandaloneProvider, "_handle_data", boom)
    _toy_pipeline("lifecycle-fail").run(env=Standalone(local=True, container=False))
    failed = [e for e in captured if isinstance(e, StepFailed)]
    completed = [e for e in captured if isinstance(e, StepCompleted)]
    assert len(failed) == 1
    assert "synthetic step failure" in failed[0].error
    assert completed == []


def test_context_propagated_from_pipeline_to_every_event(
    captured: list[LifecycleEvent],
) -> None:
    ctx = {"tenant_id": "acme", "trace_id": "xyz"}
    _toy_pipeline(context=ctx).run(env=Standalone(local=True, container=False))
    propagating = [
        e
        for e in captured
        if isinstance(e, (PipelineStarted, PipelineCompleted, StepStarted, StepCompleted))
    ]
    assert propagating, [type(e).__name__ for e in captured]
    for e in propagating:
        assert e.context == ctx, type(e).__name__


# ----------------------------------------------------------------------
# FastAPI runtime wiring
# ----------------------------------------------------------------------


@register_adapter
class _LcLoadAdapter(ModelAdapter):
    """Synthetic adapter for model_loaded / model_unloaded tests."""

    framework: ClassVar[str] = "_lifecycle_test"

    def train(self, **_: Any) -> Any:  # pragma: no cover - unused
        return object()

    def save(self, model: Any, path: Path) -> Path:  # pragma: no cover - unused
        path.mkdir(parents=True, exist_ok=True)
        return path

    def load(self, path: Path) -> Any:
        return {"path": str(path)}

    def predict(self, model: Any, inputs: Any) -> Any:
        return {"out": inputs}


@register_adapter
class _LcRaiseAdapter(ModelAdapter):
    """Synthetic adapter whose ``predict`` raises — used to drive
    the inference_failed wiring."""

    framework: ClassVar[str] = "_lifecycle_raise"

    def train(self, **_: Any) -> Any:  # pragma: no cover - unused
        return object()

    def save(self, model: Any, path: Path) -> Path:  # pragma: no cover - unused
        path.mkdir(parents=True, exist_ok=True)
        return path

    def load(self, path: Path) -> Any:
        return object()

    def predict(self, model: Any, inputs: Any) -> Any:
        raise RuntimeError("synthetic predict failure")


def _model_dir(tmp_path: Path) -> Path:
    p = tmp_path / "model"
    p.mkdir(parents=True, exist_ok=True)
    return p


def test_model_loaded_fires_on_build_app(tmp_path: Path, captured: list[LifecycleEvent]) -> None:
    from ophelian.runtime.fastapi_runtime import build_app

    build_app(framework="_lifecycle_test", model_path=_model_dir(tmp_path))
    loaded = [e for e in captured if isinstance(e, ModelLoaded)]
    assert len(loaded) == 1
    assert loaded[0].framework == "_lifecycle_test"


def test_model_unloaded_fires_on_app_shutdown(
    tmp_path: Path, captured: list[LifecycleEvent]
) -> None:
    from ophelian.runtime.fastapi_runtime import build_app

    app = build_app(framework="_lifecycle_test", model_path=_model_dir(tmp_path))
    with TestClient(app) as client:
        client.get("/health")
    # TestClient.__exit__ triggers FastAPI shutdown handlers.
    unloaded = [e for e in captured if isinstance(e, ModelUnloaded)]
    assert len(unloaded) == 1
    assert unloaded[0].framework == "_lifecycle_test"


def test_inference_failed_fires_on_predict_exception(
    tmp_path: Path, captured: list[LifecycleEvent]
) -> None:
    from ophelian.runtime.fastapi_runtime import build_app

    app = build_app(framework="_lifecycle_raise", model_path=_model_dir(tmp_path))
    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.post("/predict", json={"inputs": [[1.0]]})
        assert resp.status_code == 500
    failed = [e for e in captured if isinstance(e, InferenceFailed)]
    assert len(failed) == 1
    assert failed[0].route == "/predict"
    assert failed[0].method == "POST"
    assert "synthetic predict failure" in failed[0].error


# ----------------------------------------------------------------------
# Spot probe wiring
# ----------------------------------------------------------------------


def test_spot_interruption_received_fires_on_rising_edge(
    captured: list[LifecycleEvent],
) -> None:
    from ophelian.runtime.spot import SpotInterruptionMonitor

    monitor = SpotInterruptionMonitor(probe=lambda: True)
    assert monitor.check() is True
    spot = [e for e in captured if isinstance(e, SpotInterruptionReceived)]
    assert len(spot) == 1
    # Subsequent polls must NOT re-emit — the monitor latches.
    monitor.check()
    monitor.check()
    spot_again = [e for e in captured if isinstance(e, SpotInterruptionReceived)]
    assert len(spot_again) == 1


# ----------------------------------------------------------------------
# Context propagation across cloud / container step-spec encoders
# ----------------------------------------------------------------------


def _make_data_request(context: dict[str, Any] | None) -> Any:
    """Build a minimal :class:`StepRequest` for encoder round-trip tests."""
    from ophelian.providers.aws_drivers import StepRequest

    data = Data(
        name="ds",
        source="memory://toy",
        format="inline",
        options={"X": [[1.0, 2.0]], "y": [0, 1]},
    )
    return StepRequest(
        run_id="run-123",
        pipeline_name="cloud-test",
        step_name="ds",
        kind="data",
        node=data,
        context=context,
    )


@pytest.mark.parametrize(
    "encoder_module",
    [
        "ophelian.providers.aws_drivers",
        "ophelian.providers.gcp_drivers",
        "ophelian.providers.azure_drivers",
    ],
)
def test_step_spec_encoders_roundtrip_context(encoder_module: str) -> None:
    """Every cloud driver's ``_encode_step_spec`` must persist
    ``request.context`` so the worker's ``step_runner`` can tag
    lifecycle events with it."""
    import base64
    import importlib
    import json as _json

    mod = importlib.import_module(encoder_module)
    ctx = {"tenant_id": "acme", "trace_id": "xyz"}
    request = _make_data_request(ctx)
    encoded = mod._encode_step_spec(request)
    decoded = _json.loads(base64.b64decode(encoded).decode("utf-8"))
    assert decoded["context"] == ctx, encoder_module


@pytest.mark.parametrize(
    "encoder_module",
    [
        "ophelian.providers.aws_drivers",
        "ophelian.providers.gcp_drivers",
        "ophelian.providers.azure_drivers",
    ],
)
def test_step_spec_encoders_omit_or_null_context_when_missing(
    encoder_module: str,
) -> None:
    """When the pipeline has no context, the encoded spec must either
    omit the key or set it to ``None`` — never crash and never invent
    data — so older specs/workers stay compatible."""
    import base64
    import importlib
    import json as _json

    mod = importlib.import_module(encoder_module)
    request = _make_data_request(None)
    decoded = _json.loads(base64.b64decode(mod._encode_step_spec(request)).decode("utf-8"))
    assert decoded.get("context") in (None,), encoder_module


def test_step_runner_emits_events_with_context(
    captured: list[LifecycleEvent], tmp_path: Path
) -> None:
    """End-to-end: write a step spec containing ``context`` to disk,
    invoke ``step_runner.run``, and assert every emitted lifecycle
    event carries the same context dict."""
    import json as _json

    from ophelian.runtime import step_runner

    ctx = {"tenant_id": "acme", "env": "prod"}
    spec = {
        "kind": "data",
        "node": {
            "name": "ds",
            "source": "memory://toy",
            "format": "inline",
            "options": {"X": [[1.0, 2.0]], "y": [0, 1]},
        },
        "artifacts": {},
        "run_id": "run-runner",
        "context": ctx,
    }
    spec_path = tmp_path / "step.json"
    spec_path.write_text(_json.dumps(spec))
    rc = step_runner.run(spec_path)
    assert rc == 0
    step_events = [e for e in captured if isinstance(e, (StepStarted, StepCompleted, StepFailed))]
    assert step_events, [type(e).__name__ for e in captured]
    for e in step_events:
        assert e.context == ctx, type(e).__name__


def test_standalone_provider_threads_context_to_container_spec(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The container path of :class:`StandaloneProvider` must write
    ``context`` into ``step.json`` so the worker's ``step_runner``
    sees it (verified end-to-end above)."""
    import json as _json

    from ophelian.providers import standalone as _standalone

    captured_specs: list[dict[str, Any]] = []
    real_writer = Path.write_text

    def _spy_write_text(self: Path, data: str, *args: Any, **kwargs: Any) -> int:
        if self.name == "step.json":
            captured_specs.append(_json.loads(data))
        return real_writer(self, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", _spy_write_text)

    from ophelian.providers.docker_engine import FakeDockerEngine

    provider = _standalone.StandaloneProvider(
        local=True,
        container=True,
        workspace=tmp_path,
        docker_engine=FakeDockerEngine(),
    )
    provider._current_pipeline_context = {"tenant_id": "acme"}

    node = Data(
        name="ds",
        source="memory://toy",
        format="inline",
        options={"X": [[1.0, 2.0]], "y": [0, 1]},
    )
    # Skip the runtime-image build — we don't need a real image to
    # write step.json (the only thing this test inspects).
    monkeypatch.setattr(provider, "_ensure_runtime_image", lambda: None)

    step_dir = tmp_path / "ds"
    step_dir.mkdir(parents=True, exist_ok=True)
    # We don't care about a successful container run here — the test
    # only verifies the spec written before the container is invoked.
    import contextlib as _contextlib

    with _contextlib.suppress(Exception):
        provider._run_step_in_container(node, step_dir, {})

    assert captured_specs, "step.json was not written"
    assert captured_specs[-1].get("context") == {"tenant_id": "acme"}
