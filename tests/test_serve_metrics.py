"""Per-endpoint serve metrics tests (Task #27).

These tests prove the full operator-grade metric set fires correctly:

* Inference duration is recorded **separately** from total HTTP latency.
* The in-flight gauge goes up during a request and back to zero after.
* HTTP status class is bucketed correctly (``2xx`` / ``5xx``).
* Token counters fire only when the adapter exposes a recognisable
  token-usage shape — and stay silent for plain prediction outputs.
* The opt-in ``/metrics`` endpoint serves Prometheus text when
  ``enable_prometheus=True``, and 404s otherwise.

The fixture installs in-memory OTel providers exactly once for the
module — see ``test_otel_instrumentation.py`` for why ``set_*_provider``
cannot be re-invoked per test.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar

import pytest
from fastapi.testclient import TestClient
from ophelian.models.base import ModelAdapter, register_adapter
from ophelian.observability.otel import (
    ATTR_FRAMEWORK,
    ATTR_HTTP_STATUS_CLASS,
    METRIC_SERVE_IDLE_SECONDS,
    METRIC_SERVE_INFERENCE_DURATION,
    METRIC_SERVE_INFLIGHT,
    METRIC_SERVE_LATENCY,
    METRIC_SERVE_REQUESTS,
    METRIC_SERVE_TOKENS_IN,
    METRIC_SERVE_TOKENS_OUT,
    _reset_auto_configuration_for_tests,
)

# ----------------------------------------------------------------------
# In-memory OTel fixture (mirrors test_otel_instrumentation.py)
# ----------------------------------------------------------------------


@pytest.fixture()
def otel(_otel_in_memory_providers: dict[str, Any]) -> Iterator[dict[str, Any]]:
    _otel_in_memory_providers["spans"].clear()
    _reset_auto_configuration_for_tests()
    yield _otel_in_memory_providers


def _metric_points(reader: Any, name: str) -> list[Any]:
    data = reader.get_metrics_data()
    points: list[Any] = []
    if data is None:
        return points
    for resource_metric in data.resource_metrics:
        for scope_metric in resource_metric.scope_metrics:
            for metric in scope_metric.metrics:
                if metric.name == name:
                    points.extend(metric.data.data_points)
    return points


# ----------------------------------------------------------------------
# Synthetic adapters — keep tests independent of sklearn/torch installs
# ----------------------------------------------------------------------


@register_adapter
class _SlowAdapter(ModelAdapter):
    """Adapter whose ``predict`` sleeps long enough to dominate latency."""

    framework: ClassVar[str] = "_serve_test_slow"

    def train(self, **_: Any) -> Any:  # pragma: no cover - unused
        return object()

    def save(self, model: Any, path: Path) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        marker = path / "marker"
        marker.write_text("ok")
        return marker

    def load(self, path: Path) -> Any:
        return {"loaded_from": str(path)}

    def predict(self, model: Any, payload: Any) -> Any:
        del model, payload
        time.sleep(0.05)  # 50ms — easily distinguishable from HTTP overhead
        return [0.0]


@register_adapter
class _OpenAIShapeAdapter(ModelAdapter):
    """Adapter that returns an OpenAI-compatible ``usage`` block."""

    framework: ClassVar[str] = "_serve_test_openai"

    def train(self, **_: Any) -> Any:  # pragma: no cover - unused
        return object()

    def save(self, model: Any, path: Path) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        marker = path / "marker"
        marker.write_text("ok")
        return marker

    def load(self, path: Path) -> Any:
        return {"loaded_from": str(path)}

    def predict(self, model: Any, payload: Any) -> Any:
        del model, payload
        return {
            "choices": [{"text": "hola"}],
            "usage": {"prompt_tokens": 13, "completion_tokens": 7},
        }


@register_adapter
class _PlainDictAdapter(ModelAdapter):
    """Returns a dict with no token-usage shape — must NOT trigger token metrics."""

    framework: ClassVar[str] = "_serve_test_plain"

    def train(self, **_: Any) -> Any:  # pragma: no cover - unused
        return object()

    def save(self, model: Any, path: Path) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        marker = path / "marker"
        marker.write_text("ok")
        return marker

    def load(self, path: Path) -> Any:
        return {"loaded_from": str(path)}

    def predict(self, model: Any, payload: Any) -> Any:
        del model, payload
        return {"prediction": [1.0, 2.0, 3.0]}


@register_adapter
class _RaisingAdapter(ModelAdapter):
    """Predict raises — drives the 5xx status_class branch."""

    framework: ClassVar[str] = "_serve_test_raising"

    def train(self, **_: Any) -> Any:  # pragma: no cover - unused
        return object()

    def save(self, model: Any, path: Path) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        marker = path / "marker"
        marker.write_text("ok")
        return marker

    def load(self, path: Path) -> Any:
        return {"loaded_from": str(path)}

    def predict(self, model: Any, payload: Any) -> Any:
        del model, payload
        raise RuntimeError("synthetic predict failure")


def _model_dir(tmp_path: Path) -> Path:
    d = tmp_path / "model"
    d.mkdir()
    return d


# ----------------------------------------------------------------------
# Inference duration vs HTTP latency
# ----------------------------------------------------------------------


def test_inference_duration_recorded_separately_from_http_latency(
    otel: dict[str, Any], tmp_path: Path
) -> None:
    from ophelian.runtime.fastapi_runtime import build_app

    app = build_app(framework="_serve_test_slow", model_path=_model_dir(tmp_path))
    with TestClient(app) as client:
        response = client.post("/predict", json={"inputs": [[0.0]]})
    assert response.status_code == 200

    inf_points = _metric_points(otel["metrics"], METRIC_SERVE_INFERENCE_DURATION)
    inf_predict = [p for p in inf_points if p.attributes.get("http.route") == "/predict"]
    assert inf_predict, [p.attributes for p in inf_points]
    # Slow adapter sleeps 50ms; the recorded sample must reflect that.
    assert inf_predict[0].sum >= 0.04, inf_predict[0].sum

    lat_points = _metric_points(otel["metrics"], METRIC_SERVE_LATENCY)
    lat_predict = [p for p in lat_points if p.attributes.get("http.route") == "/predict"]
    assert lat_predict
    # Total HTTP latency must be >= inference duration (it includes
    # serialization + framework overhead). This is the contract the
    # whole "two histograms" feature exists to expose.
    assert lat_predict[0].sum >= inf_predict[0].sum


def test_health_endpoint_records_no_inference_duration(
    otel: dict[str, Any], tmp_path: Path
) -> None:
    """Endpoints that never invoke an adapter must NOT emit an
    inference-duration sample (else the histogram is meaningless)."""
    from ophelian.runtime.fastapi_runtime import build_app

    app = build_app(framework="_serve_test_slow", model_path=_model_dir(tmp_path))
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200

    inf_points = _metric_points(otel["metrics"], METRIC_SERVE_INFERENCE_DURATION)
    health_points = [p for p in inf_points if p.attributes.get("http.route") == "/health"]
    assert health_points == []


# ----------------------------------------------------------------------
# In-flight gauge
# ----------------------------------------------------------------------


def test_inflight_gauge_returns_to_zero_after_request(otel: dict[str, Any], tmp_path: Path) -> None:
    from ophelian.runtime.fastapi_runtime import build_app

    app = build_app(framework="_serve_test_slow", model_path=_model_dir(tmp_path))
    with TestClient(app) as client:
        for _ in range(3):
            assert client.post("/predict", json={"inputs": [[0.0]]}).status_code == 200

    inflight = _metric_points(otel["metrics"], METRIC_SERVE_INFLIGHT)
    predict_inflight = [p for p in inflight if p.attributes.get("http.route") == "/predict"]
    assert predict_inflight, [p.attributes for p in inflight]
    # Sum across all data points for /predict must be 0 — every inc
    # has been matched by a dec.
    assert sum(p.value for p in predict_inflight) == 0


# ----------------------------------------------------------------------
# Status class
# ----------------------------------------------------------------------


def test_status_class_5xx_for_raising_predict(otel: dict[str, Any], tmp_path: Path) -> None:
    from ophelian.runtime.fastapi_runtime import build_app

    app = build_app(framework="_serve_test_raising", model_path=_model_dir(tmp_path))
    # raise_server_exceptions=False so the test client returns the 500
    # response object instead of re-raising the adapter exception.
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/predict", json={"inputs": [[0.0]]})
    assert response.status_code == 500

    counts = _metric_points(otel["metrics"], METRIC_SERVE_REQUESTS)
    five_xx = [
        p
        for p in counts
        if p.attributes.get("http.route") == "/predict"
        and p.attributes.get(ATTR_HTTP_STATUS_CLASS) == "5xx"
    ]
    assert sum(p.value for p in five_xx) == 1, [
        (p.attributes.get("http.route"), p.attributes.get(ATTR_HTTP_STATUS_CLASS), p.value)
        for p in counts
    ]


def test_status_class_2xx_for_healthy_request(otel: dict[str, Any], tmp_path: Path) -> None:
    from ophelian.runtime.fastapi_runtime import build_app

    app = build_app(framework="_serve_test_slow", model_path=_model_dir(tmp_path))
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200

    counts = _metric_points(otel["metrics"], METRIC_SERVE_REQUESTS)
    two_xx = [
        p
        for p in counts
        if p.attributes.get("http.route") == "/health"
        and p.attributes.get(ATTR_HTTP_STATUS_CLASS) == "2xx"
    ]
    assert sum(p.value for p in two_xx) >= 1


# ----------------------------------------------------------------------
# Token usage detection
# ----------------------------------------------------------------------


def test_tokens_counters_fire_for_openai_shape(otel: dict[str, Any], tmp_path: Path) -> None:
    from ophelian.runtime.fastapi_runtime import build_app

    app = build_app(framework="_serve_test_openai", model_path=_model_dir(tmp_path))
    with TestClient(app) as client:
        response = client.post("/predict", json={"inputs": "hello"})
    assert response.status_code == 200

    tokens_in = _metric_points(otel["metrics"], METRIC_SERVE_TOKENS_IN)
    tokens_out = _metric_points(otel["metrics"], METRIC_SERVE_TOKENS_OUT)
    in_predict = [p for p in tokens_in if p.attributes.get("http.route") == "/predict"]
    out_predict = [p for p in tokens_out if p.attributes.get("http.route") == "/predict"]
    assert sum(p.value for p in in_predict) == 13
    assert sum(p.value for p in out_predict) == 7
    # Framework must propagate so consumers can fan out by adapter.
    assert any(p.attributes.get(ATTR_FRAMEWORK) == "_serve_test_openai" for p in in_predict)


def test_tokens_counters_silent_for_plain_predict(otel: dict[str, Any], tmp_path: Path) -> None:
    """Adapters that return a plain prediction (no usage shape) must
    NOT contribute to the token counters — guessing would be worse
    than having no signal."""
    from ophelian.runtime.fastapi_runtime import build_app

    app = build_app(framework="_serve_test_plain", model_path=_model_dir(tmp_path))
    with TestClient(app) as client:
        assert client.post("/predict", json={"inputs": [[1.0]]}).status_code == 200

    tokens_in = _metric_points(otel["metrics"], METRIC_SERVE_TOKENS_IN)
    in_plain_route = [
        p
        for p in tokens_in
        if p.attributes.get("http.route") == "/predict"
        and p.attributes.get(ATTR_FRAMEWORK) == "_serve_test_plain"
    ]
    assert in_plain_route == [], [p.attributes for p in tokens_in]


# ----------------------------------------------------------------------
# Idle time
# ----------------------------------------------------------------------


def test_idle_seconds_counter_increments_between_requests(
    otel: dict[str, Any], tmp_path: Path
) -> None:
    from ophelian.runtime.fastapi_runtime import build_app

    app = build_app(framework="_serve_test_plain", model_path=_model_dir(tmp_path))
    with TestClient(app) as client:
        client.post("/predict", json={"inputs": [[1.0]]})
        time.sleep(0.05)  # observable idle window
        client.post("/predict", json={"inputs": [[1.0]]})

    idle_points = _metric_points(otel["metrics"], METRIC_SERVE_IDLE_SECONDS)
    predict_idle = [p for p in idle_points if p.attributes.get("http.route") == "/predict"]
    # Second request goes 0 -> 1 in-flight after the first finishes,
    # so the counter must capture the ~50ms gap.
    assert sum(p.value for p in predict_idle) >= 0.04


# ----------------------------------------------------------------------
# /metrics Prometheus endpoint
# ----------------------------------------------------------------------


def test_metrics_endpoint_404_when_prometheus_disabled(tmp_path: Path) -> None:
    from ophelian.runtime.fastapi_runtime import build_app

    app = build_app(framework="_serve_test_plain", model_path=_model_dir(tmp_path))
    with TestClient(app) as client:
        assert client.get("/metrics").status_code == 404


def test_metrics_endpoint_serves_prometheus_text_when_enabled(
    otel: dict[str, Any], tmp_path: Path
) -> None:
    """Full integration: Ophelian metric → OTel MeterProvider →
    PrometheusMetricReader → prometheus_client.REGISTRY → ``/metrics``.

    The session-scoped conftest attaches a ``PrometheusMetricReader``
    to the meter provider precisely so this test can prove the bridge
    is wired correctly end-to-end.
    """
    pytest.importorskip("prometheus_client")
    pytest.importorskip("opentelemetry.exporter.prometheus")
    from ophelian.runtime.fastapi_runtime import build_app

    app = build_app(
        framework="_serve_test_plain",
        model_path=_model_dir(tmp_path),
        enable_prometheus=True,
    )
    with TestClient(app) as client:
        # Drive at least one request so something has been recorded.
        assert client.post("/predict", json={"inputs": [[1.0]]}).status_code == 200
        response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    body = response.text
    assert "# HELP" in body and "# TYPE" in body
    # OTel sanitises ``ophelian.serve.requests`` into
    # ``ophelian_serve_requests`` for Prometheus; counters get a
    # ``_total`` suffix per Prometheus conventions. We accept either
    # to stay resilient to upstream naming policy tweaks.
    assert "ophelian_serve_requests" in body or "ophelian_serve_latency" in body, body[:2000]


def test_queue_depth_observer_reports_registered_value(
    otel: dict[str, Any], tmp_path: Path
) -> None:
    """`register_queue_depth_observer` must wire a callback that the
    OTel meter polls on each collection cycle, populating
    ``ophelian.serve.queue.depth`` with the current value."""
    del tmp_path
    from ophelian.observability.otel import (
        METRIC_SERVE_QUEUE_DEPTH,
        register_queue_depth_observer,
    )

    state = {"depth": 7}
    register_queue_depth_observer(lambda: state["depth"], route="/predict")

    points = _metric_points(otel["metrics"], METRIC_SERVE_QUEUE_DEPTH)
    predict = [p for p in points if p.attributes.get("http.route") == "/predict"]
    assert predict, [p.attributes for p in points]
    assert predict[-1].value == 7
