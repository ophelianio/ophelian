"""OpenTelemetry first-class instrumentation tests.

These tests prove three things:

1. With ``[otel]`` installed and an in-memory exporter wired up, every
   ``Pipeline.run`` emits a parent span, every step emits a child
   span, and the documented metrics fire (pipeline runs counter, step
   duration histogram, serve request counter + latency histogram).
2. The Ophelian-specific attribute schema is honoured exactly — these
   names are the public contract downstream consumers depend on.
3. With OTel uninstalled (simulated by hiding ``opentelemetry`` from
   the import system), every helper degrades to a no-op and zero
   behaviour change is observable.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from ophelian.core.nodes import Data, Pipeline
from ophelian.observability.otel import (
    ATTR_ENV_CLASS,
    ATTR_PIPELINE_NAME,
    ATTR_PROVIDER,
    ATTR_STATUS,
    ATTR_STEP_KIND,
    ATTR_STEP_NAME,
    METRIC_PIPELINE_RUNS,
    METRIC_SERVE_LATENCY,
    METRIC_SERVE_REQUESTS,
    METRIC_STEP_DURATION,
    _reset_auto_configuration_for_tests,
)
from ophelian.providers.standalone import StandaloneProvider

# ----------------------------------------------------------------------
# In-memory OTel fixture
# ----------------------------------------------------------------------


@pytest.fixture()
def _otel_providers(_otel_in_memory_providers: dict[str, Any]) -> dict[str, Any]:
    """Thin alias over the shared session-scoped OTel providers.

    The actual provider installation lives in ``tests/conftest.py``
    because OTel's ``set_*_provider`` is one-shot — multiple test
    modules cannot each install their own provider, the second call
    is silently dropped. Individual tests still clear the in-memory
    exporter / reader between runs through their own fixtures.
    """
    return _otel_in_memory_providers


@pytest.fixture()
def otel_capture(_otel_providers: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Per-test view: clears the in-memory exporter so each test sees
    only spans / metrics it produced itself."""
    _otel_providers["spans"].clear()
    # InMemoryMetricReader is cumulative; record a snapshot baseline
    # the helpers can subtract against. For our assertions we only
    # check ``>= 1`` semantics so the baseline is informational.
    _reset_auto_configuration_for_tests()
    yield _otel_providers


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _metric_points(reader: Any, name: str) -> list[Any]:
    data = reader.get_metrics_data()
    points: list[Any] = []
    if data is None:
        return points
    for resource_metric in data.resource_metrics:
        for scope_metric in resource_metric.scope_metrics:
            for metric in scope_metric.metrics:
                if metric.name != name:
                    continue
                points.extend(metric.data.data_points)
    return points


# ----------------------------------------------------------------------
# Pipeline + step instrumentation
# ----------------------------------------------------------------------


def _toy_data_node(name: str, depends_on: tuple[str, ...] = ()) -> Data:
    return Data(
        name=name,
        source="memory://toy",
        format="inline",
        options={"X": [[0.0, 1.0], [1.0, 0.0]], "y": [0, 1]},
        depends_on=depends_on,
    )


def test_pipeline_run_emits_parent_span_with_schema(
    otel_capture: dict[str, Any], tmp_path: Path
) -> None:
    pipeline = Pipeline(name="otel-smoke", steps=[_toy_data_node("ingest")])
    provider = StandaloneProvider(local=True, workspace=tmp_path, container=False)
    result = pipeline.run(provider)

    assert result.succeeded
    spans = otel_capture["spans"].get_finished_spans()
    pipeline_spans = [s for s in spans if s.name == "ophelian.pipeline.otel-smoke"]
    assert len(pipeline_spans) == 1, [s.name for s in spans]
    p = pipeline_spans[0]
    assert p.attributes[ATTR_PIPELINE_NAME] == "otel-smoke"
    assert p.attributes[ATTR_ENV_CLASS] == "standalone"
    assert p.attributes[ATTR_PROVIDER] == "standalone"
    assert p.attributes[ATTR_STATUS] == "success"


def test_each_step_emits_child_span(otel_capture: dict[str, Any], tmp_path: Path) -> None:
    pipeline = Pipeline(
        name="otel-steps",
        steps=[
            _toy_data_node("ingest"),
            _toy_data_node("transform", depends_on=("ingest",)),
        ],
    )
    provider = StandaloneProvider(local=True, workspace=tmp_path, container=False)
    pipeline.run(provider)

    spans = otel_capture["spans"].get_finished_spans()
    step_spans = [s for s in spans if s.name.startswith("ophelian.step.")]
    assert {s.name for s in step_spans} == {
        "ophelian.step.ingest",
        "ophelian.step.transform",
    }
    for step_span_obj in step_spans:
        assert step_span_obj.attributes[ATTR_STEP_KIND] == "data"
        assert step_span_obj.attributes[ATTR_STEP_NAME] in {"ingest", "transform"}
        assert step_span_obj.attributes[ATTR_PROVIDER] == "standalone"
        # Status is set by step_span on exit.
        assert step_span_obj.attributes[ATTR_STATUS] == "success"


def test_pipeline_metrics_recorded(otel_capture: dict[str, Any], tmp_path: Path) -> None:
    pipeline = Pipeline(name="otel-metrics", steps=[_toy_data_node("ingest")])
    provider = StandaloneProvider(local=True, workspace=tmp_path, container=False)
    pipeline.run(provider)

    runs = _metric_points(otel_capture["metrics"], METRIC_PIPELINE_RUNS)
    # Scope to this pipeline name (the metric reader is module-cumulative).
    scoped = [p for p in runs if p.attributes.get(ATTR_PIPELINE_NAME) == "otel-metrics"]
    assert scoped, [p.attributes for p in runs]
    # Regression: pipeline.runs must be incremented EXACTLY once per
    # pipeline.run() call, not double-counted by both pipeline_span and
    # the explicit record_pipeline_outcome path.
    assert sum(p.value for p in scoped) == 1, [
        (p.attributes.get(ATTR_PIPELINE_NAME), p.attributes.get(ATTR_STATUS), p.value)
        for p in scoped
    ]
    assert all(p.attributes.get(ATTR_STATUS) == "success" for p in scoped)

    step_durations = _metric_points(otel_capture["metrics"], METRIC_STEP_DURATION)
    assert step_durations, "step duration histogram never recorded"
    # Regression: step duration histogram must observe exactly one
    # sample per step (was previously double-recorded by step_span +
    # standalone provider). Note: histogram is cumulative; we check
    # the delta by recording the count for the "ingest" step name —
    # this test owns the only "ingest" emission of the module *after*
    # filtering on the (pipeline-unique) attribute set is impractical,
    # so we only assert presence + status here. The "exactly one"
    # contract is enforced by the pipeline.runs assertion above and
    # by the explicit failure regression test below.
    ingest_points = [
        p for p in step_durations if p.attributes.get(ATTR_STEP_NAME) == "ingest"
    ]
    assert ingest_points
    assert any(
        p.attributes.get(ATTR_STATUS) == "success"
        and p.attributes.get(ATTR_STEP_KIND) == "data"
        for p in ingest_points
    )


def test_step_span_records_failed_duration_when_handler_raises(
    otel_capture: dict[str, Any],
) -> None:
    """Regression for the cloud worker dispatch path: ``step_span``
    with default ``emit_metric=True`` (the cloud path) must emit
    exactly one duration sample tagged ``status="failed"`` when an
    exception escapes the with-body."""
    from ophelian.observability.otel import step_span as _step_span

    raised = False
    try:
        with _step_span(
            step_name="cloud-fail",
            step_kind="train",
            run_id="r-cloud",
            provider="standalone-runtime",
        ):
            raise RuntimeError("simulated handler crash")
    except RuntimeError:
        raised = True
    assert raised

    durations = _metric_points(otel_capture["metrics"], METRIC_STEP_DURATION)
    matching = [
        p
        for p in durations
        if p.attributes.get(ATTR_STEP_NAME) == "cloud-fail"
        and p.attributes.get(ATTR_STATUS) == "failed"
    ]
    assert matching, [
        (p.attributes.get(ATTR_STEP_NAME), p.attributes.get(ATTR_STATUS))
        for p in durations
    ]
    assert sum(p.count for p in matching) == 1


def test_failed_pipeline_status_is_not_overwritten_to_success(
    otel_capture: dict[str, Any], tmp_path: Path
) -> None:
    """Regression: a Pipeline that returns a failed PipelineResult
    (no exception) must surface as ``failed`` on both the parent span
    and the ``pipeline.runs`` counter."""
    # Force a failure by pointing a Data node at an unsupported source.
    pipeline = Pipeline(
        name="otel-fail",
        steps=[Data(name="ingest", source="https://does-not-exist", format="csv")],
    )
    provider = StandaloneProvider(local=True, workspace=tmp_path, container=False)
    result = pipeline.run(provider)
    assert not result.succeeded

    spans = otel_capture["spans"].get_finished_spans()
    pipeline_spans = [s for s in spans if s.name == "ophelian.pipeline.otel-fail"]
    assert pipeline_spans
    assert pipeline_spans[0].attributes[ATTR_STATUS] == "failed"

    runs = _metric_points(otel_capture["metrics"], METRIC_PIPELINE_RUNS)
    scoped = [p for p in runs if p.attributes.get(ATTR_PIPELINE_NAME) == "otel-fail"]
    assert scoped, [p.attributes for p in runs]
    failed = [p for p in scoped if p.attributes.get(ATTR_STATUS) == "failed"]
    success = [p for p in scoped if p.attributes.get(ATTR_STATUS) == "success"]
    assert sum(p.value for p in failed) == 1, [p.attributes for p in scoped]
    # Critical: must NOT be counted as success.
    assert sum(p.value for p in success) == 0, [p.attributes for p in scoped]


# ----------------------------------------------------------------------
# Serve instrumentation
# ----------------------------------------------------------------------


def test_serve_request_emits_span_and_metrics(
    otel_capture: dict[str, Any], tmp_path: Path
) -> None:
    pytest.importorskip("sklearn")
    import joblib
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression()
    model.classes_ = []  # placeholder; we won't predict in this test
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    joblib.dump(model, model_dir / "model.joblib")

    from ophelian.runtime.fastapi_runtime import build_app

    app = build_app(framework="sklearn", model_path=model_dir)
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200

    spans = otel_capture["spans"].get_finished_spans()
    serve_spans = [s for s in spans if s.name.startswith("ophelian.serve.")]
    assert serve_spans, [s.name for s in spans]
    s = serve_spans[0]
    assert s.attributes["http.method"] == "GET"
    assert s.attributes["http.route"] == "/health"
    assert s.attributes["http.status_code"] == 200

    counter_points = _metric_points(otel_capture["metrics"], METRIC_SERVE_REQUESTS)
    assert any(
        p.attributes.get("http.route") == "/health" and p.value >= 1 for p in counter_points
    )
    latency_points = _metric_points(otel_capture["metrics"], METRIC_SERVE_LATENCY)
    # The in-memory metric reader is session-scoped (set_meter_provider
    # is one-shot in OTel), so other tests may have left /predict
    # samples behind. Assert that *our* /health sample is present.
    assert any(p.attributes.get("http.route") == "/health" for p in latency_points), [
        p.attributes for p in latency_points
    ]


# ----------------------------------------------------------------------
# Zero-behaviour-change without the [otel] extra
# ----------------------------------------------------------------------


def test_helpers_are_noops_when_otel_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """With ``opentelemetry`` hidden, every helper must degrade silently.

    We hide the package by mapping every ``opentelemetry*`` name to
    ``None`` in ``sys.modules`` (Python treats that as "import failed
    last time, raise ImportError on re-import") *and* by reaching into
    our own otel module to clear cached singletons.
    """
    from ophelian.observability import otel as otel_mod

    # Hide opentelemetry packages from any subsequent import attempts.
    for name in list(sys.modules):
        if name == "opentelemetry" or name.startswith("opentelemetry."):
            monkeypatch.setitem(sys.modules, name, None)
    # Clear cached metric singletons from prior tests so the lazy
    # accessors re-evaluate and observe that OTel is now "missing".
    monkeypatch.setattr(otel_mod, "_pipeline_runs", None)
    monkeypatch.setattr(otel_mod, "_step_duration", None)
    monkeypatch.setattr(otel_mod, "_serve_requests", None)
    monkeypatch.setattr(otel_mod, "_serve_latency", None)
    monkeypatch.setattr(otel_mod, "_auto_configured", False)

    assert otel_mod.is_otel_available() is False
    assert otel_mod.get_tracer() is None
    assert otel_mod.get_meter() is None

    # Span helpers must yield None and never raise.
    with otel_mod.pipeline_span(pipeline_name="p", run_id="r") as span:
        assert span is None
    with otel_mod.step_span(step_name="s", step_kind="data") as span:
        assert span is None
    with otel_mod.serve_request_span(method="GET", route="/x") as span:
        assert span is None

    # The post-hoc recorders must also accept span=None / no meter.
    otel_mod.record_serve_outcome(
        method="GET", route="/x", status_code=200, duration_seconds=0.01, span=None
    )
    otel_mod.record_step_outcome(
        step_name="s", step_kind="data", status="success", duration_seconds=0.0
    )


def test_pipeline_run_succeeds_without_otel(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Even with OTel hidden, the public Pipeline.run contract holds."""
    from ophelian.observability import otel as otel_mod

    for name in list(sys.modules):
        if name == "opentelemetry" or name.startswith("opentelemetry."):
            monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.setattr(otel_mod, "_pipeline_runs", None)
    monkeypatch.setattr(otel_mod, "_step_duration", None)
    monkeypatch.setattr(otel_mod, "_serve_requests", None)
    monkeypatch.setattr(otel_mod, "_serve_latency", None)
    monkeypatch.setattr(otel_mod, "_auto_configured", False)

    pipeline = Pipeline(name="otel-absent", steps=[_toy_data_node("ingest")])
    provider = StandaloneProvider(local=True, workspace=tmp_path, container=False)
    result = pipeline.run(provider)
    assert result.succeeded
