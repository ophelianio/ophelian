"""Data and prediction drift detection hooks (Task #30).

Continuous AI workloads (CRM/ERP scoring, recommendations,
classifiers) silently degrade long before ground truth arrives —
leads close weeks later, transactions settle next quarter, campaigns
end after a fortnight. By the time accuracy visibly drops, the
served population has already drifted away from what the model was
trained on. This module ships two opt-in monitors that compare live
inference traffic against a per-model baseline distribution and emit
a ``drift_detected`` lifecycle event the moment a configurable
statistical test crosses threshold:

* :class:`DataDriftMonitor` — observes inference *inputs*.
* :class:`PredictionDriftMonitor` — observes model *outputs*.

Both monitors maintain a true sliding window per feature (the
oldest samples roll off as new ones arrive) and re-evaluate every
``stride`` new observations once the window has filled at least
once. When no monitor is attached, the serve path is unchanged.

Statistics are delegated to a small :class:`DriftDetector`
interface — Ophelian does not implement the tests itself. The
default implementation prefers `Evidently
<https://github.com/evidentlyai/evidently>`_ when installed (via
the ``[drift]`` extra) and falls back to the SciPy reference
implementations Evidently itself wraps. Either way, swapping the
backend is an implementation detail and does not change consumer
code.

End-to-end wiring with Train + Deploy
-------------------------------------
Setting ``capture_baseline=True`` on a :class:`~ophelian.core.nodes.Train`
node snapshots the training feature + prediction distributions as a
``baseline.json`` artifact. A downstream :class:`~ophelian.core.nodes.Deploy`
node with ``drift_baseline="auto"`` (or an explicit path) builds the
monitors automatically and the FastAPI runtime attaches them so the
serve path begins observing on the very first request. Manual wiring
is also supported via :func:`attach_drift_monitors`.
"""

from __future__ import annotations

import json
import threading
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Protocol

from ophelian.observability.events import LifecycleEvent
from ophelian.observability.events import emit as emit_lifecycle

if TYPE_CHECKING:
    from fastapi import FastAPI

TestName = Literal["ks", "chi2", "psi"]
MonitorKind = Literal["data", "prediction"]


# ----------------------------------------------------------------------
# Lifecycle event
# ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class DriftDetected(LifecycleEvent):
    """Emitted when a monitor's statistical test crosses threshold.

    Payload schema is part of the public contract — auto-rollback,
    canary controllers, and alerting integrations key off these
    fields by name. ``model_id`` identifies the served model;
    ``endpoint`` identifies the deployment surface (e.g. the Deploy
    node name or a logical ``service:route``) so consumers running
    the same model behind multiple endpoints can route correctly.
    """

    EVENT_NAME: ClassVar[str] = "ophelian.drift.detected"
    monitor: MonitorKind
    model_id: str
    endpoint: str | None
    feature: str | None
    metric: str
    value: float
    threshold: float
    direction: Literal["above", "below"]
    sample_size: int


# ----------------------------------------------------------------------
# Baseline
# ----------------------------------------------------------------------


def _is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _classify(samples: Sequence[Any]) -> Literal["numeric", "categorical"]:
    """Classify a feature as numeric or categorical from its samples."""
    has_any = False
    for v in samples:
        if v is None:
            continue
        has_any = True
        if not _is_numeric(v):
            return "categorical"
    return "numeric" if has_any else "categorical"


@dataclass(frozen=True, slots=True)
class FeatureBaseline:
    """Reference distribution snapshot for a single feature.

    For ``numeric`` features the raw reference samples are stored so
    KS / PSI can be computed against the live window without
    information loss. For ``categorical`` features the per-category
    counts are stored.
    """

    name: str
    kind: Literal["numeric", "categorical"]
    samples: tuple[float, ...] = ()
    categories: tuple[str, ...] = ()
    counts: tuple[int, ...] = ()

    @classmethod
    def from_samples(cls, name: str, samples: Sequence[Any]) -> FeatureBaseline:
        kind = _classify(samples)
        if kind == "numeric":
            cleaned = tuple(float(v) for v in samples if v is not None)
            return cls(name=name, kind="numeric", samples=cleaned)
        counts: dict[str, int] = {}
        for v in samples:
            if v is None:
                continue
            key = str(v)
            counts[key] = counts.get(key, 0) + 1
        cats = tuple(sorted(counts))
        return cls(
            name=name,
            kind="categorical",
            categories=cats,
            counts=tuple(counts[c] for c in cats),
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"name": self.name, "kind": self.kind}
        if self.kind == "numeric":
            out["samples"] = list(self.samples)
        else:
            out["categories"] = list(self.categories)
            out["counts"] = list(self.counts)
        return out

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> FeatureBaseline:
        kind = payload["kind"]
        if kind == "numeric":
            return cls(
                name=payload["name"],
                kind="numeric",
                samples=tuple(float(v) for v in payload.get("samples", [])),
            )
        return cls(
            name=payload["name"],
            kind="categorical",
            categories=tuple(payload.get("categories", [])),
            counts=tuple(int(c) for c in payload.get("counts", [])),
        )


@dataclass(frozen=True, slots=True)
class Baseline:
    """Reference distributions for inputs and predictions.

    Captured from a training run (``Baseline.from_samples``), saved
    as a small JSON artifact (``Baseline.save``) and reloaded by the
    serving endpoint (``Baseline.load``). The detector configuration
    is intentionally NOT stored here — the monitor owns it so the
    same baseline can power multiple monitors with different tests.
    """

    features: tuple[FeatureBaseline, ...] = ()
    predictions: FeatureBaseline | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    version: int = 1

    def feature(self, name: str) -> FeatureBaseline | None:
        for f in self.features:
            if f.name == name:
                return f
        return None

    @classmethod
    def from_samples(
        cls,
        *,
        feature_samples: Mapping[str, Sequence[Any]] | None = None,
        prediction_samples: Sequence[Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Baseline:
        feats: list[FeatureBaseline] = []
        if feature_samples:
            for name, samples in feature_samples.items():
                feats.append(FeatureBaseline.from_samples(name, list(samples)))
        preds: FeatureBaseline | None = None
        if prediction_samples is not None:
            preds = FeatureBaseline.from_samples("__prediction__", list(prediction_samples))
        return cls(
            features=tuple(feats),
            predictions=preds,
            metadata=dict(metadata or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "metadata": dict(self.metadata),
            "features": [f.to_dict() for f in self.features],
            "predictions": self.predictions.to_dict() if self.predictions else None,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Baseline:
        feats = tuple(FeatureBaseline.from_dict(f) for f in payload.get("features", []))
        preds_payload = payload.get("predictions")
        preds = FeatureBaseline.from_dict(preds_payload) if preds_payload else None
        return cls(
            features=feats,
            predictions=preds,
            metadata=dict(payload.get("metadata", {})),
            version=int(payload.get("version", 1)),
        )

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict()))
        return target

    @classmethod
    def load(cls, path: str | Path) -> Baseline:
        return cls.from_dict(json.loads(Path(path).read_text()))


# ----------------------------------------------------------------------
# Detector interface + library-backed implementations
# ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DriftResult:
    """Outcome of a single detector run for one feature."""

    metric: str
    value: float
    threshold: float
    drifted: bool
    direction: Literal["above", "below"]
    sample_size: int


class DriftDetector(Protocol):
    """Pluggable detector interface.

    The default implementation is :class:`ScipyDetector`. When
    Evidently is installed, :func:`default_detector` returns an
    :class:`EvidentlyDetector` instead. A consumer that prefers
    Alibi-Detect or any other library ships their own class
    implementing this protocol and passes it to the monitor — the
    public Ophelian surface (monitors + ``DriftDetected`` event)
    stays identical.
    """

    @property
    def test(self) -> TestName:  # pragma: no cover - protocol
        ...

    @property
    def threshold(self) -> float:  # pragma: no cover - protocol
        ...

    def detect(
        self, *, reference: FeatureBaseline, current: Sequence[Any]
    ) -> DriftResult:  # pragma: no cover - protocol
        ...


def _ks_p_value(reference: Sequence[float], current: Sequence[float]) -> float:
    """Two-sample Kolmogorov-Smirnov p-value via SciPy.

    SciPy is the reference statistics library Evidently and
    Alibi-Detect themselves wrap for KS — Ophelian does not
    re-implement the test, only the wrapping needed to feed the
    pluggable :class:`DriftDetector` interface.
    """
    from scipy.stats import ks_2samp  # type: ignore[import-untyped]

    return float(ks_2samp(list(reference), list(current)).pvalue)


def _chi2_p_value(reference_counts: Sequence[int], current_counts: Sequence[int]) -> float:
    """Chi-squared p-value for two count vectors via SciPy."""
    from scipy.stats import chi2_contingency

    table = [list(reference_counts), list(current_counts)]
    if all(sum(row) == 0 for row in table):
        return 1.0
    result = chi2_contingency(table)
    return float(result[1])


def _psi(reference: Sequence[float], current: Sequence[float], *, bins: int = 10) -> float:
    """Population Stability Index via NumPy histograms.

    PSI is canonical risk-modeling drift metric (also exposed by
    Evidently and Alibi-Detect). The threshold convention is the
    industry standard: < 0.1 no shift, 0.1 to 0.25 moderate,
    > 0.25 significant.
    """
    import numpy as np

    if not reference or not current:
        return 0.0
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    lo = float(min(ref.min(), cur.min()))
    hi = float(max(ref.max(), cur.max()))
    if hi <= lo:
        return 0.0
    edges = np.linspace(lo, hi, bins + 1)
    edges[-1] += 1e-12
    ref_hist, _ = np.histogram(ref, bins=edges)
    cur_hist, _ = np.histogram(cur, bins=edges)
    # Laplace-smooth so empty bins do not blow up the log term.
    ref_p = (ref_hist + 1e-6) / (ref_hist.sum() + bins * 1e-6)
    cur_p = (cur_hist + 1e-6) / (cur_hist.sum() + bins * 1e-6)
    return float(((cur_p - ref_p) * np.log(cur_p / ref_p)).sum())


@dataclass(frozen=True, slots=True)
class ScipyDetector:
    """SciPy-backed :class:`DriftDetector` implementation.

    * ``test="ks"``    — two-sample Kolmogorov-Smirnov on numeric
      features. Drift when ``p_value < threshold`` (default 0.05).
    * ``test="chi2"``  — Chi-squared on categorical features. Drift
      when ``p_value < threshold``.
    * ``test="psi"``   — Population Stability Index on numeric
      features. Drift when ``psi > threshold`` (default 0.2).

    The monitor auto-promotes ``ks`` / ``psi`` to ``chi2`` for
    categorical features so a single detector instance can cover
    both column types in :class:`DataDriftMonitor`.
    """

    test: TestName = "ks"
    threshold: float = 0.05

    def detect(
        self, *, reference: FeatureBaseline, current: Sequence[Any]
    ) -> DriftResult:
        if reference.kind == "categorical":
            return self._chi2(reference, current)
        if self.test == "psi":
            return self._psi(reference, current)
        return self._ks(reference, current)

    def _ks(self, reference: FeatureBaseline, current: Sequence[Any]) -> DriftResult:
        cur = [float(v) for v in current if v is not None]
        p = _ks_p_value(reference.samples, cur)
        return DriftResult(
            metric="ks_p_value",
            value=p,
            threshold=self.threshold,
            drifted=p < self.threshold,
            direction="below",
            sample_size=len(cur),
        )

    def _psi(self, reference: FeatureBaseline, current: Sequence[Any]) -> DriftResult:
        cur = [float(v) for v in current if v is not None]
        psi = _psi(reference.samples, cur)
        threshold = self.threshold if self.test == "psi" else 0.2
        return DriftResult(
            metric="psi",
            value=psi,
            threshold=threshold,
            drifted=psi > threshold,
            direction="above",
            sample_size=len(cur),
        )

    def _chi2(self, reference: FeatureBaseline, current: Sequence[Any]) -> DriftResult:
        cats = list(reference.categories)
        ref_counts = list(reference.counts)
        cur_counts = [0] * len(cats)
        index = {c: i for i, c in enumerate(cats)}
        extras = 0
        for v in current:
            if v is None:
                continue
            key = str(v)
            i = index.get(key)
            if i is None:
                extras += 1
            else:
                cur_counts[i] += 1
        if extras:
            cats.append("__other__")
            ref_counts.append(0)
            cur_counts.append(extras)
        p = _chi2_p_value(ref_counts, cur_counts)
        return DriftResult(
            metric="chi2_p_value",
            value=p,
            threshold=self.threshold,
            drifted=p < self.threshold,
            direction="below",
            sample_size=sum(cur_counts),
        )


@dataclass(frozen=True, slots=True)
class EvidentlyDetector:
    """Evidently-backed :class:`DriftDetector` (optional).

    Available when Ophelian is installed with the ``[drift]`` extra
    (``pip install ophelian[drift]``). Delegates per-column drift
    to ``evidently.metrics.ColumnDriftMetric`` so consumers who
    already standardise on Evidently get its richer test catalogue
    (Wasserstein, Jensen-Shannon, energy distance, etc.) without
    Ophelian re-implementing any statistics.

    Test name mapping:

    * ``ks``   → ``ks`` (numeric)
    * ``chi2`` → ``chisquare`` (categorical)
    * ``psi``  → ``psi`` (numeric)

    Categorical features always promote to ``chisquare`` regardless
    of ``test`` so a single detector covers both column types, the
    same way :class:`ScipyDetector` does.
    """

    test: TestName = "ks"
    threshold: float = 0.05

    @staticmethod
    def is_available() -> bool:
        # Probe the specific symbol we depend on rather than the
        # top-level package, so version skew (the symbol moves
        # across major Evidently releases) downgrades cleanly to
        # the SciPy backend instead of crashing at first call.
        try:
            from evidently.metrics import (  # type: ignore[import-untyped]
                ColumnDriftMetric,  # noqa: F401
            )
            from evidently.report import Report  # type: ignore[import-untyped]  # noqa: F401
        except ImportError:
            return False
        return True

    def detect(
        self, *, reference: FeatureBaseline, current: Sequence[Any]
    ) -> DriftResult:  # pragma: no cover - exercised only when Evidently is installed
        # Build two single-column DataFrames and ask Evidently for
        # the drift metric. The import is local so absence of the
        # optional dependency does not break ``ophelian.observability``.
        import pandas as pd
        from evidently.metrics import ColumnDriftMetric
        from evidently.report import Report

        if reference.kind == "numeric":
            ref_df = pd.DataFrame({"x": list(reference.samples)})
            stat_test = "psi" if self.test == "psi" else "ks"
        else:
            ref_values: list[str] = []
            for cat, count in zip(reference.categories, reference.counts, strict=True):
                ref_values.extend([cat] * count)
            ref_df = pd.DataFrame({"x": ref_values})
            stat_test = "chisquare"

        cur_df = pd.DataFrame({"x": [v for v in current if v is not None]})
        report = Report(
            metrics=[ColumnDriftMetric(column_name="x", stattest=stat_test, threshold=self.threshold)]
        )
        report.run(reference_data=ref_df, current_data=cur_df)
        result = report.as_dict()["metrics"][0]["result"]
        score = float(result.get("drift_score", 0.0))
        drifted = bool(result.get("drift_detected", False))
        if stat_test == "psi":
            metric = "psi"
            direction: Literal["above", "below"] = "above"
        elif stat_test == "chisquare":
            metric = "chi2_p_value"
            direction = "below"
        else:
            metric = "ks_p_value"
            direction = "below"
        return DriftResult(
            metric=metric,
            value=score,
            threshold=self.threshold,
            drifted=drifted,
            direction=direction,
            sample_size=len(cur_df),
        )


def default_detector(
    *, test: TestName = "ks", threshold: float = 0.05
) -> DriftDetector:
    """Return the preferred backend with the requested test/threshold.

    Picks :class:`EvidentlyDetector` when Evidently is installed;
    otherwise returns :class:`ScipyDetector`. Either way the
    underlying statistics come from a proven library — Ophelian
    does not implement the tests itself.

    Raises :class:`ImportError` (fail-fast, NOT swallowed by the
    serve-path ``suppress``) when neither backend can be loaded.
    Install ``ophelian[drift]`` to satisfy the dependency.
    """
    if EvidentlyDetector.is_available():  # pragma: no cover - depends on extras
        return EvidentlyDetector(test=test, threshold=threshold)
    try:
        import numpy  # noqa: F401
        import scipy  # type: ignore[import-untyped]  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised when extras missing
        raise ImportError(
            "Ophelian drift detection requires either Evidently or "
            "SciPy + NumPy. Install the optional extra: "
            "`pip install 'ophelian[drift]'`."
        ) from exc
    return ScipyDetector(test=test, threshold=threshold)


# ----------------------------------------------------------------------
# Coercion helpers
# ----------------------------------------------------------------------


def _records_from_inputs(
    inputs: Any, *, feature_names: Sequence[str] | None = None
) -> list[dict[str, Any]]:
    """Best-effort coercion of a ``/predict`` payload into row records.

    Accepts every shape an Ophelian model adapter can be called
    with on the FastAPI runtime:

    * ``dict[name -> value]``      — single named row
    * ``dict[name -> list]``       — column-major batch (all
      same length)
    * ``list[dict]``               — row-major batch with named
      fields
    * ``list[list|tuple]``         — positional batch (sklearn /
      xgboost tabular convention); columns are mapped to
      ``feature_names`` in order, falling back to ``f0..fn`` when
      no names are supplied
    * ``list|tuple``               — single positional row, mapped
      the same way

    Anything else is treated as opaque and silently dropped — the
    monitor will not raise on the hot path no matter what shape
    the model receives.
    """
    if inputs is None:
        return []
    if isinstance(inputs, Mapping):
        list_lengths = [len(v) for v in inputs.values() if isinstance(v, (list, tuple))]
        if list_lengths and len(list_lengths) == len(inputs) and len(set(list_lengths)) == 1:
            n = list_lengths[0]
            return [
                {k: list(v)[i] for k, v in inputs.items() if isinstance(v, (list, tuple))}
                for i in range(n)
            ]
        return [dict(inputs)]
    if isinstance(inputs, (list, tuple)):
        if not inputs:
            return []
        if all(isinstance(item, Mapping) for item in inputs):
            return [dict(item) for item in inputs]
        # Positional batch — list[list|tuple]. Map each row's
        # columns to feature names from the baseline so monitor
        # windows actually receive samples for tabular adapters
        # (sklearn, xgboost, etc.) which call ``predict(X)`` with
        # X as a positional matrix.
        if all(isinstance(item, (list, tuple)) for item in inputs):
            return _positional_records(list(inputs), feature_names)
        # Single positional row.
        if not any(isinstance(item, (list, tuple, Mapping)) for item in inputs):
            return _positional_records([list(inputs)], feature_names)
        return []
    return []


def _positional_records(
    rows: Sequence[Sequence[Any]], feature_names: Sequence[str] | None
) -> list[dict[str, Any]]:
    if not rows:
        return []
    width = max(len(row) for row in rows)
    if feature_names and len(feature_names) >= width:
        names = list(feature_names[:width])
    else:
        names = [f"f{i}" for i in range(width)]
    out: list[dict[str, Any]] = []
    for row in rows:
        rec: dict[str, Any] = {}
        for i, name in enumerate(names):
            if i < len(row):
                rec[name] = row[i]
        out.append(rec)
    return out


def _flatten_predictions(prediction: Any) -> list[Any]:
    if prediction is None:
        return []
    if isinstance(prediction, (list, tuple)):
        return [v for v in prediction if not isinstance(v, (list, tuple, Mapping))]
    if isinstance(prediction, Mapping):
        return []
    return [prediction]


# ----------------------------------------------------------------------
# Sliding window
# ----------------------------------------------------------------------


@dataclass(slots=True)
class _SlidingWindow:
    """Fixed-capacity FIFO with a stride-based "ready to evaluate" flag.

    ``size`` is the window capacity; once filled the oldest sample
    rolls off as new samples arrive so the window always reflects the
    most recent ``size`` observations (true sliding semantics).
    ``stride`` is the number of new observations that must be
    appended between consecutive evaluations — defaults to a tenth
    of ``size`` (minimum 1) so detection cadence stays bounded
    without going down to per-request.
    """

    size: int
    stride: int
    values: deque[Any] = field(init=False)
    _since_last_eval: int = 0
    _ever_filled: bool = False

    def __post_init__(self) -> None:
        self.values = deque(maxlen=self.size)

    def extend(self, items: Iterable[Any]) -> int:
        """Append items, return the number actually added (None skipped)."""
        n = 0
        for v in items:
            if v is None:
                continue
            self.values.append(v)
            n += 1
        if not self._ever_filled and len(self.values) >= self.size:
            self._ever_filled = True
        self._since_last_eval += n
        return n

    def ready(self) -> bool:
        """True when the window is full AND ``stride`` new samples
        have arrived since the last evaluation."""
        return self._ever_filled and self._since_last_eval >= self.stride

    def mark_evaluated(self) -> None:
        self._since_last_eval = 0

    def snapshot(self) -> list[Any]:
        return list(self.values)


# ----------------------------------------------------------------------
# Monitors
# ----------------------------------------------------------------------


class _BaseDriftMonitor:
    kind: ClassVar[MonitorKind]

    def __init__(
        self,
        *,
        baseline: Baseline,
        model_id: str,
        endpoint: str | None = None,
        window_size: int = 200,
        stride: int | None = None,
        detector: DriftDetector | None = None,
        emit: Callable[[LifecycleEvent], None] | None = None,
        source: str | None = None,
    ) -> None:
        if window_size < 2:
            raise ValueError("window_size must be at least 2")
        resolved_stride = stride if stride is not None else max(1, window_size // 10)
        if resolved_stride < 1:
            raise ValueError("stride must be at least 1")
        self.baseline = baseline
        self.model_id = model_id
        self.endpoint = endpoint
        self.window_size = window_size
        self.stride = resolved_stride
        self.detector = detector if detector is not None else default_detector()
        self._emit = emit or emit_lifecycle
        self._source = source or f"drift:{self.kind}:{model_id}"
        self._lock = threading.Lock()
        self._windows: dict[str, _SlidingWindow] = {}
        self.last_results: dict[str, DriftResult] = {}

    def _window(self, name: str) -> _SlidingWindow:
        win = self._windows.get(name)
        if win is None:
            win = _SlidingWindow(size=self.window_size, stride=self.stride)
            self._windows[name] = win
        return win

    def _check(self, feature_name: str, reference: FeatureBaseline) -> DriftResult | None:
        with self._lock:
            win = self._window(feature_name)
            if not win.ready():
                return None
            current = win.snapshot()
            win.mark_evaluated()
        result = self.detector.detect(reference=reference, current=current)
        self.last_results[feature_name] = result
        if result.drifted:
            self._emit(
                DriftDetected(
                    source=self._source,
                    monitor=self.kind,
                    model_id=self.model_id,
                    endpoint=self.endpoint,
                    feature=feature_name if feature_name != "__prediction__" else None,
                    metric=result.metric,
                    value=result.value,
                    threshold=result.threshold,
                    direction=result.direction,
                    sample_size=result.sample_size,
                )
            )
        return result


class DataDriftMonitor(_BaseDriftMonitor):
    """Compare inference *inputs* against the baseline.

    Call :meth:`observe` with each request's ``inputs`` payload (the
    runtime middleware does this automatically when attached). The
    per-feature window slides as new requests arrive; once full,
    the configured detector re-runs every ``stride`` observations
    and emits :class:`DriftDetected` if the test crosses threshold.
    """

    kind: ClassVar[MonitorKind] = "data"

    def observe(self, inputs: Any) -> list[DriftResult]:
        # Pass baseline feature names so positional ``list[list]``
        # batches (sklearn / xgboost tabular convention) map their
        # columns to the same names the baseline was captured with.
        records = _records_from_inputs(
            inputs, feature_names=tuple(f.name for f in self.baseline.features)
        )
        if not records:
            return []
        per_feature: dict[str, list[Any]] = {}
        for rec in records:
            for k, v in rec.items():
                per_feature.setdefault(k, []).append(v)
        results: list[DriftResult] = []
        for name, values in per_feature.items():
            reference = self.baseline.feature(name)
            if reference is None:
                continue
            with self._lock:
                self._window(name).extend(values)
            r = self._check(name, reference)
            if r is not None:
                results.append(r)
        return results


class PredictionDriftMonitor(_BaseDriftMonitor):
    """Compare model *outputs* against the baseline prediction distribution."""

    kind: ClassVar[MonitorKind] = "prediction"

    def observe(self, prediction: Any) -> DriftResult | None:
        if self.baseline.predictions is None:
            return None
        values = _flatten_predictions(prediction)
        if not values:
            return None
        with self._lock:
            self._window("__prediction__").extend(values)
        return self._check("__prediction__", self.baseline.predictions)


# ----------------------------------------------------------------------
# FastAPI integration
# ----------------------------------------------------------------------


def attach_drift_monitors(
    app: FastAPI,
    *,
    data_monitor: DataDriftMonitor | None = None,
    prediction_monitor: PredictionDriftMonitor | None = None,
) -> None:
    """Attach drift monitors to a FastAPI app built by :func:`build_app`.

    The runtime's ``/predict`` route checks ``app.state`` for these
    monitors on every request and feeds them the inputs / output.
    When neither monitor is attached the check is a single attribute
    read so the hot path stays effectively free for users who do not
    opt in.
    """
    if data_monitor is not None:
        app.state.ophelian_data_drift_monitor = data_monitor
    if prediction_monitor is not None:
        app.state.ophelian_prediction_drift_monitor = prediction_monitor


def build_monitors_from_baseline(
    baseline: Baseline,
    *,
    model_id: str,
    endpoint: str | None = None,
    window_size: int = 200,
    stride: int | None = None,
    test: TestName = "ks",
    threshold: float = 0.05,
) -> tuple[DataDriftMonitor | None, PredictionDriftMonitor | None]:
    """Build the canonical monitor pair from a loaded baseline.

    Returns ``(data_monitor, prediction_monitor)`` — either may be
    ``None`` when the baseline carries no matching reference. Used
    by the standalone provider's ``Deploy`` step to wire monitors
    automatically when the user sets ``drift_baseline`` on the node.
    """
    detector = default_detector(test=test, threshold=threshold)
    data_monitor: DataDriftMonitor | None = None
    if baseline.features:
        data_monitor = DataDriftMonitor(
            baseline=baseline,
            model_id=model_id,
            endpoint=endpoint,
            window_size=window_size,
            stride=stride,
            detector=detector,
        )
    prediction_monitor: PredictionDriftMonitor | None = None
    if baseline.predictions is not None:
        prediction_monitor = PredictionDriftMonitor(
            baseline=baseline,
            model_id=model_id,
            endpoint=endpoint,
            window_size=window_size,
            stride=stride,
            detector=detector,
        )
    return data_monitor, prediction_monitor


__all__ = [
    "Baseline",
    "DataDriftMonitor",
    "DriftDetected",
    "DriftDetector",
    "DriftResult",
    "EvidentlyDetector",
    "FeatureBaseline",
    "PredictionDriftMonitor",
    "ScipyDetector",
    "attach_drift_monitors",
    "build_monitors_from_baseline",
    "default_detector",
]
