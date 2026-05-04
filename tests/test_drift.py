"""Tests for the drift detection hooks (Task #30).

Covers the four contractual outcomes called out in the task brief:

* drift correctly detected on a synthetic distribution shift,
* no false positive on a stable stream,
* baseline save / load round-trip preserves enough information for
  the detector to reach the same verdict,
* a ``drift_detected`` lifecycle event reaches the bus, with the
  documented payload (metric, value, threshold, monitor type, model
  identifier) and is mirrored onto the active OTel span.

Also exercises the FastAPI middleware integration end-to-end so the
serve path actually feeds the monitors.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar

import pytest
from fastapi.testclient import TestClient
from ophelian.models.base import ModelAdapter, register_adapter
from ophelian.observability.drift import (
    Baseline,
    DataDriftMonitor,
    DriftDetected,
    DriftResult,
    PredictionDriftMonitor,
    ScipyDetector,
    attach_drift_monitors,
)
from ophelian.observability.events import (
    LifecycleEvent,
    _clear_subscribers_for_tests,
    on_event,
)


@pytest.fixture()
def captured() -> Iterator[list[LifecycleEvent]]:
    _clear_subscribers_for_tests()
    events: list[LifecycleEvent] = []
    on_event(events.append)
    try:
        yield events
    finally:
        _clear_subscribers_for_tests()


@pytest.fixture()
def otel(_otel_in_memory_providers: dict[str, Any]) -> Iterator[dict[str, Any]]:
    _otel_in_memory_providers["spans"].clear()
    yield _otel_in_memory_providers


# ----------------------------------------------------------------------
# Detector / statistical correctness
# ----------------------------------------------------------------------


def test_synthetic_shift_detected_on_numeric_feature(
    captured: list[LifecycleEvent],
) -> None:
    rng = random.Random(0)
    reference = [rng.gauss(0.0, 1.0) for _ in range(500)]
    baseline = Baseline.from_samples(feature_samples={"x": reference})
    monitor = DataDriftMonitor(
        baseline=baseline, model_id="m@v1", window_size=200, detector=ScipyDetector(test="ks")
    )
    # Feed a clearly-shifted distribution (mean +3) one record at a
    # time so we also cover the per-record observe path.
    for _ in range(200):
        monitor.observe({"x": rng.gauss(3.0, 1.0)})
    drifts = [e for e in captured if isinstance(e, DriftDetected)]
    assert drifts, "KS test should flag a clear mean shift"
    ev = drifts[0]
    assert ev.monitor == "data"
    assert ev.model_id == "m@v1"
    assert ev.feature == "x"
    assert ev.metric == "ks_p_value"
    assert ev.direction == "below"
    assert ev.value < ev.threshold
    assert ev.sample_size == 200
    # Default endpoint is None — wired-up deployments populate it
    # from the Deploy node name (see the integration test below).
    assert ev.endpoint is None


def test_endpoint_identifier_is_propagated_through_event(
    captured: list[LifecycleEvent],
) -> None:
    """``DriftDetected.endpoint`` is part of the public payload so
    consumers running the same model behind multiple endpoints can
    route alerts/rollback per surface."""
    rng = random.Random(42)
    baseline = Baseline.from_samples(
        feature_samples={"x": [rng.gauss(0.0, 1.0) for _ in range(400)]}
    )
    monitor = DataDriftMonitor(
        baseline=baseline,
        model_id="m@v1",
        endpoint="checkout-scoring",
        window_size=100,
    )
    for _ in range(100):
        monitor.observe({"x": rng.gauss(4.0, 1.0)})
    drifts = [e for e in captured if isinstance(e, DriftDetected)]
    assert drifts and drifts[0].endpoint == "checkout-scoring"


def test_stable_stream_does_not_false_positive(
    captured: list[LifecycleEvent],
) -> None:
    rng = random.Random(1)
    reference = [rng.gauss(0.0, 1.0) for _ in range(1000)]
    baseline = Baseline.from_samples(feature_samples={"x": reference})
    monitor = DataDriftMonitor(
        baseline=baseline,
        model_id="m@v1",
        window_size=200,
        detector=ScipyDetector(test="ks", threshold=0.05),
    )
    # Five full windows from the SAME distribution. With p<0.05 the
    # expected false-positive rate is ~5% per window so 5 windows
    # could legitimately trip once; we use a fresh RNG seed and
    # assert the strong "zero false positives" property which holds
    # for this seed. Drift events are only emitted on the rising
    # edge of a full window so the count is bounded by 5.
    for _ in range(5 * 200):
        monitor.observe({"x": rng.gauss(0.0, 1.0)})
    drifts = [e for e in captured if isinstance(e, DriftDetected)]
    assert drifts == [], f"unexpected drift on stable stream: {drifts}"


def test_categorical_chi2_detects_class_imbalance_shift(
    captured: list[LifecycleEvent],
) -> None:
    reference = ["A"] * 700 + ["B"] * 300
    baseline = Baseline.from_samples(feature_samples={"label": reference})
    monitor = DataDriftMonitor(
        baseline=baseline,
        model_id="cls@v1",
        window_size=100,
        detector=ScipyDetector(test="chi2", threshold=0.01),
    )
    # New traffic is 30/70 instead of 70/30 — chi2 should flag it.
    for _ in range(70):
        monitor.observe({"label": "B"})
    for _ in range(30):
        monitor.observe({"label": "A"})
    drifts = [e for e in captured if isinstance(e, DriftDetected)]
    assert drifts, "chi2 should flag an inverted class balance"
    assert drifts[0].metric == "chi2_p_value"


def test_psi_detects_distribution_shift(captured: list[LifecycleEvent]) -> None:
    rng = random.Random(2)
    reference = [rng.gauss(0.0, 1.0) for _ in range(500)]
    baseline = Baseline.from_samples(feature_samples={"x": reference})
    monitor = DataDriftMonitor(
        baseline=baseline,
        model_id="m@v1",
        window_size=200,
        detector=ScipyDetector(test="psi", threshold=0.2),
    )
    for _ in range(200):
        monitor.observe({"x": rng.gauss(2.5, 1.0)})
    drifts = [e for e in captured if isinstance(e, DriftDetected)]
    assert drifts, "PSI should flag a 2.5-sigma mean shift"
    ev = drifts[0]
    assert ev.metric == "psi"
    assert ev.direction == "above"
    assert ev.value > ev.threshold


# ----------------------------------------------------------------------
# Baseline persistence round-trip
# ----------------------------------------------------------------------


def test_baseline_round_trip_preserves_detector_verdict(
    tmp_path: Path, captured: list[LifecycleEvent]
) -> None:
    rng = random.Random(3)
    feature_samples = {
        "num": [rng.gauss(0.0, 1.0) for _ in range(400)],
        "cat": [rng.choice(["A", "B", "C"]) for _ in range(400)],
    }
    prediction_samples = [rng.random() for _ in range(400)]
    baseline = Baseline.from_samples(
        feature_samples=feature_samples,
        prediction_samples=prediction_samples,
        metadata={"trained_at": "2026-01-01"},
    )
    target = tmp_path / "baseline.json"
    baseline.save(target)
    loaded = Baseline.load(target)

    assert loaded.metadata == {"trained_at": "2026-01-01"}
    assert {f.name for f in loaded.features} == {"num", "cat"}
    assert loaded.predictions is not None
    assert len(loaded.predictions.samples) == 400
    cat = next(f for f in loaded.features if f.name == "cat")
    assert cat.kind == "categorical"
    assert sum(cat.counts) == 400

    # Same shifted stream against both baselines must reach the same
    # verdict — that's the contract round-trip really protects.
    shifted = [rng.gauss(3.0, 1.0) for _ in range(200)]
    detector = ScipyDetector(test="ks")
    fresh = detector.detect(reference=baseline.feature("num"), current=shifted)
    after = detector.detect(reference=loaded.feature("num"), current=shifted)
    assert fresh.drifted == after.drifted
    assert fresh.value == pytest.approx(after.value, rel=1e-9)


# ----------------------------------------------------------------------
# Lifecycle event mirroring + payload contract
# ----------------------------------------------------------------------


def test_drift_detected_mirrors_to_active_otel_span(
    otel: dict[str, Any], captured: list[LifecycleEvent]
) -> None:
    from ophelian.observability.otel import pipeline_span

    rng = random.Random(4)
    baseline = Baseline.from_samples(
        feature_samples={"x": [rng.gauss(0.0, 1.0) for _ in range(400)]}
    )
    monitor = DataDriftMonitor(baseline=baseline, model_id="m@v1", window_size=100)
    with pipeline_span(pipeline_name="drift-mirror", run_id="r1"):
        for _ in range(100):
            monitor.observe({"x": rng.gauss(4.0, 1.0)})

    drifts = [e for e in captured if isinstance(e, DriftDetected)]
    assert drifts, "expected drift to be detected"

    spans = otel["spans"].get_finished_spans()
    pipeline = [s for s in spans if "pipeline" in s.name]
    assert pipeline, [s.name for s in spans]
    events = pipeline[-1].events
    drift_events = [e for e in events if e.name == "ophelian.drift.detected"]
    assert drift_events, [e.name for e in events]
    attrs = drift_events[0].attributes or {}
    for key in ("monitor", "model_id", "metric", "value", "threshold", "direction"):
        assert key in attrs, f"missing attribute {key!r} in {dict(attrs)!r}"


# ----------------------------------------------------------------------
# Prediction-drift monitor
# ----------------------------------------------------------------------


def test_prediction_drift_monitor_emits_on_shifted_outputs(
    captured: list[LifecycleEvent],
) -> None:
    rng = random.Random(5)
    baseline = Baseline.from_samples(
        prediction_samples=[rng.random() for _ in range(500)]
    )
    monitor = PredictionDriftMonitor(
        baseline=baseline, model_id="m@v1", window_size=150
    )
    for _ in range(150):
        # All predictions clamped near 1.0 — a clear distribution shift
        # away from the uniform baseline.
        monitor.observe(rng.uniform(0.9, 1.0))
    drifts = [e for e in captured if isinstance(e, DriftDetected)]
    assert drifts
    assert drifts[0].monitor == "prediction"
    # PredictionDriftMonitor reports feature=None per the schema.
    assert drifts[0].feature is None


def test_prediction_monitor_no_baseline_is_silent_noop(
    captured: list[LifecycleEvent],
) -> None:
    monitor = PredictionDriftMonitor(
        baseline=Baseline.from_samples(feature_samples={"x": [1.0, 2.0, 3.0]}),
        model_id="m@v1",
        window_size=10,
    )
    for v in range(50):
        assert monitor.observe(float(v)) is None
    assert [e for e in captured if isinstance(e, DriftDetected)] == []


# ----------------------------------------------------------------------
# FastAPI integration
# ----------------------------------------------------------------------


@register_adapter
class _DriftAdapter(ModelAdapter):
    """Synthetic adapter whose predict echoes ``inputs['x']`` so we
    can drive both data- and prediction-drift monitors from the same
    HTTP traffic."""

    framework: ClassVar[str] = "_drift_test"

    def train(self, **_: Any) -> Any:  # pragma: no cover - unused
        return object()

    def save(self, model: Any, path: Path) -> Path:  # pragma: no cover - unused
        path.mkdir(parents=True, exist_ok=True)
        return path

    def load(self, path: Path) -> Any:
        return object()

    def predict(self, model: Any, inputs: Any) -> Any:
        if isinstance(inputs, dict):
            return float(inputs.get("x", 0.0)) + 5.0
        return 0.0


def _model_dir(tmp_path: Path) -> Path:
    p = tmp_path / "model"
    p.mkdir(parents=True, exist_ok=True)
    return p


def test_fastapi_serve_path_feeds_attached_monitors(
    tmp_path: Path, captured: list[LifecycleEvent]
) -> None:
    from ophelian.runtime.fastapi_runtime import build_app

    rng = random.Random(6)
    feature_baseline = [rng.gauss(0.0, 1.0) for _ in range(400)]
    prediction_baseline = [v + 5.0 for v in feature_baseline]
    baseline = Baseline.from_samples(
        feature_samples={"x": feature_baseline},
        prediction_samples=prediction_baseline,
    )

    app = build_app(framework="_drift_test", model_path=_model_dir(tmp_path))
    attach_drift_monitors(
        app,
        data_monitor=DataDriftMonitor(
            baseline=baseline, model_id="m@v1", window_size=100
        ),
        prediction_monitor=PredictionDriftMonitor(
            baseline=baseline, model_id="m@v1", window_size=100
        ),
    )

    with TestClient(app) as client:
        for _ in range(100):
            shifted = rng.gauss(4.0, 1.0)
            r = client.post("/predict", json={"inputs": {"x": shifted}})
            assert r.status_code == 200

    drifts = [e for e in captured if isinstance(e, DriftDetected)]
    monitors_seen = {e.monitor for e in drifts}
    assert "data" in monitors_seen, drifts
    assert "prediction" in monitors_seen, drifts


def test_fastapi_serve_path_unchanged_when_no_monitor_attached(
    tmp_path: Path, captured: list[LifecycleEvent]
) -> None:
    """Default behaviour for users who do not attach monitors must
    be a complete no-op — the contract calls this out explicitly."""
    from ophelian.runtime.fastapi_runtime import build_app

    app = build_app(framework="_drift_test", model_path=_model_dir(tmp_path))
    with TestClient(app) as client:
        for _ in range(20):
            assert client.post("/predict", json={"inputs": {"x": 0.5}}).status_code == 200
    assert [e for e in captured if isinstance(e, DriftDetected)] == []


# ----------------------------------------------------------------------
# Window / API edge cases
# ----------------------------------------------------------------------


def test_window_slides_with_stride_and_keeps_recent_samples() -> None:
    """The window must roll oldest-off / newest-on (true sliding)
    rather than reset after each evaluation. Detection cadence is
    governed by ``stride`` once the window has filled at least once
    so consecutive checks operate on overlapping but advancing
    samples (the standard online-drift-detection pattern)."""
    baseline = Baseline.from_samples(feature_samples={"x": [0.0] * 100})
    monitor = DataDriftMonitor(
        baseline=baseline, model_id="m", window_size=10, stride=2
    )
    # Fill the window: it triggers the first evaluation, then keeps
    # all 10 most-recent samples (does NOT reset to empty).
    for _ in range(10):
        monitor.observe({"x": 0.0})
    win = monitor._windows["x"]  # type: ignore[attr-defined]
    assert len(win.values) == 10
    # Two more observations satisfy ``stride`` and trigger another
    # evaluation but the window stays at capacity (oldest rolls off).
    monitor.observe({"x": 0.0})
    monitor.observe({"x": 0.0})
    assert len(win.values) == 10


def test_positional_list_batch_maps_columns_to_baseline_names(
    captured: list[LifecycleEvent],
) -> None:
    """sklearn / xgboost tabular adapters call ``predict(X)`` with X
    as ``list[list]``. The monitor must map those positional columns
    to the baseline's named features so input drift is actually
    evaluated for that very common shape."""
    rng = random.Random(11)
    baseline = Baseline.from_samples(
        feature_samples={
            "f0": [rng.gauss(0.0, 1.0) for _ in range(400)],
            "f1": [rng.gauss(0.0, 1.0) for _ in range(400)],
        }
    )
    monitor = DataDriftMonitor(
        baseline=baseline, model_id="m@v1", window_size=80, stride=10
    )
    # Send 100 positional rows (list[list]) of clearly-shifted
    # values — both columns should populate windows and trigger
    # drift detection.
    for _ in range(100):
        monitor.observe([[rng.gauss(5.0, 1.0), rng.gauss(5.0, 1.0)]])
    win = monitor._windows.get("f0")  # type: ignore[attr-defined]
    assert win is not None and len(win.values) > 0
    drifts = [e for e in captured if isinstance(e, DriftDetected)]
    assert drifts, "positional batch should populate windows and detect drift"
    assert {e.feature for e in drifts} == {"f0", "f1"}


def test_single_positional_row_observation() -> None:
    """A single positional row (``list|tuple``) is also a valid shape."""
    baseline = Baseline.from_samples(feature_samples={"f0": [0.0] * 50})
    monitor = DataDriftMonitor(
        baseline=baseline, model_id="m", window_size=10, stride=2
    )
    for _ in range(10):
        monitor.observe([0.5])
    assert len(monitor._windows["f0"].values) == 10  # type: ignore[attr-defined]


def test_unknown_features_are_ignored() -> None:
    """A request carrying a feature absent from the baseline must
    not raise nor pollute monitor state."""
    baseline = Baseline.from_samples(feature_samples={"known": [0.0, 1.0, 2.0]})
    monitor = DataDriftMonitor(baseline=baseline, model_id="m", window_size=4)
    for _ in range(20):
        monitor.observe({"unknown": 999.0})
    assert "unknown" not in monitor._windows  # type: ignore[attr-defined]


def test_batched_payload_decomposes_into_records() -> None:
    """A batch payload (dict[name -> list]) feeds N points per call."""
    baseline = Baseline.from_samples(feature_samples={"x": [0.0] * 100})
    monitor = DataDriftMonitor(
        baseline=baseline, model_id="m", window_size=20, stride=5
    )
    monitor.observe({"x": [0.0] * 19})  # one short of full
    win = monitor._windows["x"]  # type: ignore[attr-defined]
    assert len(win.values) == 19
    monitor.observe({"x": [0.0]})
    # Window slides — it stays at capacity, the oldest sample rolls
    # off as the newest arrives. (Tumbling clear-on-eval has been
    # explicitly retired in favour of true sliding semantics.)
    assert len(win.values) == 20


def test_window_size_must_be_at_least_two() -> None:
    with pytest.raises(ValueError):
        DataDriftMonitor(
            baseline=Baseline.from_samples(feature_samples={"x": [1.0, 2.0]}),
            model_id="m",
            window_size=1,
        )


def test_train_capture_baseline_then_deploy_attaches_monitors(
    tmp_path: Path, captured: list[LifecycleEvent]
) -> None:
    """End-to-end: a Train node with ``capture_baseline=True`` writes
    a ``baseline`` artifact; a Deploy node with ``drift_baseline="auto"``
    picks it up and the served endpoint emits ``drift_detected`` with
    the Deploy node name as ``endpoint`` once the live distribution
    diverges from the training distribution."""
    from ophelian import Data, Deploy, Pipeline, Standalone, Train

    rng = random.Random(7)
    train_x = [[rng.gauss(0.0, 1.0)] for _ in range(400)]
    train_y = [0 if row[0] < 0.0 else 1 for row in train_x]
    pipeline = Pipeline(
        [
            Data(
                name="ds",
                source="memory://drift",
                format="inline",
                options={"X": train_x, "y": train_y},
            ),
            Train(
                name="trainer",
                framework="sklearn",
                model="sklearn.linear_model.LogisticRegression",
                data="ds",
                capture_baseline=True,
            ),
            Deploy(
                name="serve-checkout",
                model="trainer",
                port=9101,
                drift_baseline="auto",
                drift_window_size=80,
                drift_stride=10,
                drift_threshold=0.05,
            ),
        ],
        name="drift-e2e",
    )
    provider = Standalone(local=True, container=False, workspace=tmp_path / "ws")
    result = pipeline.run(env=provider)
    assert result.succeeded, [s.error for s in result.steps if s.error]
    # Train step published the baseline artifact.
    assert "baseline" in result.step("trainer").artifacts
    # Deploy attached monitors via build_app.
    app = provider.apps["serve-checkout"]
    assert getattr(app.state, "ophelian_data_drift_monitor", None) is not None

    client = TestClient(app)
    # Drive the served endpoint with a clearly shifted distribution.
    for _ in range(120):
        shifted = [[rng.gauss(5.0, 0.5)]]
        assert client.post("/predict", json={"inputs": shifted}).status_code == 200

    drifts = [e for e in captured if isinstance(e, DriftDetected)]
    assert drifts, "expected drift to be detected on shifted serve traffic"
    assert all(e.endpoint == "serve-checkout" for e in drifts)


def test_container_deploy_propagates_drift_env_to_app_from_env(
    tmp_path: Path,
) -> None:
    """Container deploy mode must wire drift config via the env
    contract consumed by ``app_from_env`` so opt-in drift behaviour
    has parity with in-process deploy (Task #30 parity finding)."""
    from ophelian import Deploy
    from ophelian.providers.standalone import _deploy_container_env

    node = Deploy(
        name="serve-prod",
        model="trainer",
        port=8080,
        drift_baseline="auto",
        drift_window_size=120,
        drift_stride=15,
        drift_test="psi",
        drift_threshold=0.1,
    )
    env = _deploy_container_env(
        framework="sklearn",
        model_dir="/model",
        node=node,
        baseline_in_container="/baseline.json",
    )
    assert env["OPHELIAN_FRAMEWORK"] == "sklearn"
    assert env["OPHELIAN_MODEL_PATH"] == "/model"
    assert env["OPHELIAN_DRIFT_BASELINE_PATH"] == "/baseline.json"
    assert env["OPHELIAN_DRIFT_ENDPOINT"] == "serve-prod"
    assert env["OPHELIAN_DRIFT_WINDOW_SIZE"] == "120"
    assert env["OPHELIAN_DRIFT_STRIDE"] == "15"
    assert env["OPHELIAN_DRIFT_TEST"] == "psi"
    assert env["OPHELIAN_DRIFT_THRESHOLD"] == "0.1"

    # And app_from_env must consume that contract end-to-end.
    rng = random.Random(99)
    Baseline.from_samples(
        feature_samples={"x": [rng.gauss(0.0, 1.0) for _ in range(200)]},
        prediction_samples=[rng.random() for _ in range(200)],
    ).save(tmp_path / "baseline.json")
    import os as _os

    saved = {k: _os.environ.get(k) for k in env}
    try:
        _os.environ["OPHELIAN_FRAMEWORK"] = "_drift_test"
        _os.environ["OPHELIAN_MODEL_PATH"] = str(_model_dir(tmp_path))
        _os.environ["OPHELIAN_DRIFT_BASELINE_PATH"] = str(tmp_path / "baseline.json")
        _os.environ["OPHELIAN_DRIFT_ENDPOINT"] = "serve-prod"
        _os.environ["OPHELIAN_DRIFT_WINDOW_SIZE"] = "50"
        _os.environ["OPHELIAN_DRIFT_TEST"] = "ks"
        _os.environ["OPHELIAN_DRIFT_THRESHOLD"] = "0.05"
        from ophelian.runtime.fastapi_runtime import app_from_env

        app = app_from_env()
    finally:
        for k, v in saved.items():
            if v is None:
                _os.environ.pop(k, None)
            else:
                _os.environ[k] = v
    assert getattr(app.state, "ophelian_data_drift_monitor", None) is not None
    assert app.state.ophelian_data_drift_monitor.endpoint == "serve-prod"


def test_drift_result_payload_fields() -> None:
    res = DriftResult(
        metric="ks_p_value",
        value=0.01,
        threshold=0.05,
        drifted=True,
        direction="below",
        sample_size=200,
    )
    assert res.drifted
    assert res.direction == "below"
