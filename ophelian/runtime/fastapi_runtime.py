"""FastAPI inference runtime.

`build_app` returns a FastAPI app bound to a `ModelAdapter`, exposing
`/health` and `/predict` (and optionally `/metrics` for Prometheus
scrapes). Heavier runtimes (Triton, BentoML, Ray Serve) will plug in
here in later releases.

Every served request is wrapped in an OpenTelemetry span and feeds the
production-grade serve metrics promoted in Task #27:

* ``ophelian.serve.requests`` — counter, dim by route + method + status class
* ``ophelian.serve.latency`` — total HTTP latency histogram
* ``ophelian.serve.inference.duration`` — model-only histogram, separate
  from HTTP / serialization overhead so SREs can isolate model regressions
* ``ophelian.serve.inflight`` — current concurrent requests (UpDownCounter)
* ``ophelian.serve.idle.seconds`` — cumulative seconds the endpoint had
  zero in-flight requests (autoscaler reclaim signal)
* ``ophelian.serve.tokens.in`` / ``ophelian.serve.tokens.out`` — only
  populated when the model adapter exposes a token-usage shape

When the ``[otel]`` extra is not installed, every helper becomes a silent
no-op and the app behaves exactly as before.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, Response

from ophelian.models import registry
from ophelian.models.base import ModelAdapter
from ophelian.observability.events import (
    InferenceFailed,
    ModelLoaded,
    ModelUnloaded,
)
from ophelian.observability.events import (
    emit as emit_lifecycle,
)
from ophelian.observability.otel import (
    ATTR_FRAMEWORK,
    extract_token_usage,
    record_inference_tokens,
    record_serve_idle,
    record_serve_outcome,
    serve_inflight_dec,
    serve_inflight_inc,
    serve_request_span,
)


def _truthy_env(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def _install_otel_middleware(app: FastAPI, *, framework: str) -> None:
    """Wrap every request in an Ophelian serve span + the full metric set.

    Records, in order: in-flight increment → idle-time delta since last
    finish → serve span → invoke handler → record outcome (latency +
    inference duration when set + status class) → tokens (when set) →
    in-flight decrement → update last-finished timestamp.
    """
    state: dict[str, Any] = {
        "lock": threading.Lock(),
        "inflight": 0,
        "last_finished_at": time.monotonic(),
    }
    app.state.ophelian_serve_state = state

    @app.middleware("http")
    async def _otel_middleware(request: Request, call_next: Any) -> Response:
        method = request.method
        # Prefer the route template (e.g. ``/predict``) over the raw
        # path so the metric cardinality stays bounded; fall back to
        # the raw URL path if the route is unknown (404, etc.).
        route_obj = request.scope.get("route")
        route = getattr(route_obj, "path", None) or request.url.path

        # Idle accounting: if we are about to go from 0 → 1 in-flight,
        # the time since the last request finished is "true idle". We
        # only feed the counter on the rising edge — bursts of
        # overlapping requests do not contribute idle time.
        with state["lock"]:
            was_idle = state["inflight"] == 0
            state["inflight"] += 1
            idle_delta = (
                time.monotonic() - state["last_finished_at"] if was_idle else 0.0
            )
        if idle_delta > 0.0:
            record_serve_idle(route=route, idle_seconds=idle_delta)
        serve_inflight_inc(method=method, route=route)

        # Pre-create the slot the handler can write into before
        # returning. The middleware reads it after ``call_next``.
        request.state.inference_duration_s = None
        request.state.tokens_in = None
        request.state.tokens_out = None

        started = time.monotonic()
        with serve_request_span(method=method, route=route, framework=framework) as span:
            try:
                response: Response = await call_next(request)
            except Exception as exc:
                self_duration = time.monotonic() - started
                record_serve_outcome(
                    method=method,
                    route=route,
                    status_code=500,
                    duration_seconds=self_duration,
                    span=span,
                    inference_duration_seconds=getattr(
                        request.state, "inference_duration_s", None
                    ),
                )
                # Lifecycle: inference_failed — emitted INSIDE the
                # serve span so OTel mirroring lands on the same
                # request trace consumers already see.
                emit_lifecycle(
                    InferenceFailed(
                        source=f"fastapi:{framework}",
                        framework=framework,
                        route=route,
                        method=method,
                        error=repr(exc),
                    )
                )
                serve_inflight_dec(method=method, route=route)
                with state["lock"]:
                    state["inflight"] -= 1
                    state["last_finished_at"] = time.monotonic()
                raise

            self_duration = time.monotonic() - started
            inf_dur = getattr(request.state, "inference_duration_s", None)
            record_serve_outcome(
                method=method,
                route=route,
                status_code=response.status_code,
                duration_seconds=self_duration,
                span=span,
                inference_duration_seconds=inf_dur,
            )
            tokens_in = getattr(request.state, "tokens_in", None)
            tokens_out = getattr(request.state, "tokens_out", None)
            if tokens_in is not None or tokens_out is not None:
                record_inference_tokens(
                    method=method,
                    route=route,
                    framework=framework,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                )
            serve_inflight_dec(method=method, route=route)
            with state["lock"]:
                state["inflight"] -= 1
                state["last_finished_at"] = time.monotonic()
            return response


def _maybe_install_prometheus(app: FastAPI) -> bool:
    """Mount ``/metrics`` returning the Prometheus text exposition.

    Returns ``True`` when the route was actually mounted, ``False``
    when ``prometheus_client`` is not installed (so callers can decide
    whether to surface a startup error or just continue silently).
    """
    try:
        from prometheus_client import (
            CONTENT_TYPE_LATEST,
            REGISTRY,
            generate_latest,
        )
    except ImportError:
        return False

    @app.get("/metrics", include_in_schema=False)
    def _metrics() -> Response:
        # ``generate_latest(REGISTRY)`` aggregates everything written by
        # the OTel ``PrometheusMetricReader`` (registered by
        # ``auto_configure_from_env`` when ``OPHELIAN_OTEL_PROMETHEUS=1``).
        return Response(content=generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)

    return True


def build_app(
    *,
    framework: str,
    model_path: str | Path,
    enable_prometheus: bool = False,
    drift_baseline_path: str | Path | None = None,
    drift_endpoint: str | None = None,
    drift_window_size: int = 200,
    drift_stride: int | None = None,
    drift_test: str = "ks",
    drift_threshold: float = 0.05,
) -> FastAPI:
    """Construct a FastAPI app that serves the model under `model_path`.

    Parameters
    ----------
    framework, model_path
        Adapter framework name and model artifact location.
    enable_prometheus
        When ``True``, mount a ``/metrics`` endpoint that returns the
        Prometheus text exposition for whatever the OTel
        ``PrometheusMetricReader`` has aggregated. Requires the
        ``[otel]`` extra; silently no-op when ``prometheus_client`` is
        not installed. Off by default to keep the surface small for
        users who do not run Prometheus.
    """
    adapter_cls = registry.get(framework)
    adapter: ModelAdapter = adapter_cls()
    model = adapter.load(Path(model_path))
    # Lifecycle: model_loaded — fires once per app construction so
    # external systems (auto-rollback, audit trails) see the
    # transition without log scraping.
    emit_lifecycle(
        ModelLoaded(
            source=f"fastapi:{framework}",
            framework=framework,
            model_path=str(model_path),
        )
    )

    # Lifecycle: model_unloaded — fires from the FastAPI lifespan
    # post-yield branch so it triggers on every graceful shutdown
    # (TestClient.__exit__, uvicorn graceful stop). FastAPI removed
    # the legacy ``add_event_handler`` API in newer releases, so the
    # lifespan context manager is the only forward-compatible hook.
    from collections.abc import AsyncIterator
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            emit_lifecycle(
                ModelUnloaded(
                    source=f"fastapi:{framework}",
                    framework=framework,
                    model_path=str(model_path),
                )
            )

    app = FastAPI(
        title=f"ophelian-inference[{framework}]",
        version="0.1.0",
        lifespan=_lifespan,
    )
    if enable_prometheus:
        # Mounting ``/metrics`` is only half the story — the route
        # serves whatever lives in ``prometheus_client.REGISTRY``, and
        # OTel meter samples only land there when a
        # ``PrometheusMetricReader`` is attached to the active meter
        # provider. Force that wiring by setting the env var BEFORE
        # the first metric singleton is created (which lazily triggers
        # ``auto_configure_from_env``). When a meter provider has
        # already been installed externally (e.g. tests, or a host
        # process that pre-configured OTel), the reader must be
        # registered there too — see ``tests/conftest.py``.
        os.environ.setdefault("OPHELIAN_OTEL_PROMETHEUS", "1")
        from ophelian.observability.otel import auto_configure_from_env

        auto_configure_from_env()
    _install_otel_middleware(app, framework=framework)
    if enable_prometheus:
        _maybe_install_prometheus(app)

    # Drift hooks (Task #30): when a baseline path is provided, load
    # it and attach the canonical monitor pair to ``app.state`` so the
    # ``/predict`` route can feed observations on every request. We
    # do this here (rather than as a follow-up call from each
    # provider) so any caller of ``build_app`` gets the same opt-in
    # surface for free.
    if drift_baseline_path is not None:
        from typing import cast

        from ophelian.observability.drift import (
            Baseline,
            TestName,
            attach_drift_monitors,
            build_monitors_from_baseline,
        )

        baseline = Baseline.load(drift_baseline_path)
        data_monitor, prediction_monitor = build_monitors_from_baseline(
            baseline,
            model_id=str(model_path),
            endpoint=drift_endpoint,
            window_size=drift_window_size,
            stride=drift_stride,
            test=cast(TestName, drift_test),
            threshold=drift_threshold,
        )
        attach_drift_monitors(
            app,
            data_monitor=data_monitor,
            prediction_monitor=prediction_monitor,
        )

    # Expose the framework name as a span attribute on every span the
    # caller creates inside a request handler — useful for downstream
    # consumers that fan out by ML framework.
    app.state.ophelian_attrs = {ATTR_FRAMEWORK: framework}

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "framework": framework, "model": str(model_path)}

    @app.post("/predict")
    def predict(request: Request, payload: dict[str, Any]) -> dict[str, Any]:
        # Time the model call separately from the whole HTTP round-trip
        # so ``ophelian.serve.inference.duration`` measures pure model
        # work — the difference vs ``ophelian.serve.latency`` is the
        # HTTP / serialization overhead an SRE wants to alert on
        # independently.
        inputs = payload.get("inputs")
        t0 = time.monotonic()
        prediction = adapter.predict(model, inputs)
        request.state.inference_duration_s = time.monotonic() - t0
        tokens_in, tokens_out = extract_token_usage(prediction, payload)
        if tokens_in is not None:
            request.state.tokens_in = tokens_in
        if tokens_out is not None:
            request.state.tokens_out = tokens_out
        # Drift hooks (Task #30) — attribute reads are O(1) when no
        # monitor is attached, so users who do not opt in pay nothing.
        # Both observation calls are wrapped in suppress() because a
        # buggy monitor must never break the serve hot path.
        import contextlib as _contextlib

        data_monitor = getattr(app.state, "ophelian_data_drift_monitor", None)
        if data_monitor is not None:
            with _contextlib.suppress(Exception):  # pragma: no cover - defensive
                data_monitor.observe(inputs)
        prediction_monitor = getattr(app.state, "ophelian_prediction_drift_monitor", None)
        if prediction_monitor is not None:
            with _contextlib.suppress(Exception):  # pragma: no cover - defensive
                prediction_monitor.observe(prediction)
        return {"prediction": prediction}

    return app


def app_from_env() -> FastAPI:
    """Zero-arg factory used by ``uvicorn --factory`` inside containers.

    Reads ``OPHELIAN_FRAMEWORK`` and ``OPHELIAN_MODEL_PATH`` from the
    environment so the Standalone provider can launch::

        python -m uvicorn --factory ophelian.runtime.fastapi_runtime:app_from_env

    inside the deploy container without having to inject keyword arguments.
    Set ``OPHELIAN_PROMETHEUS=1`` to also mount ``/metrics``.
    """
    framework = os.environ.get("OPHELIAN_FRAMEWORK")
    model_path = os.environ.get("OPHELIAN_MODEL_PATH")
    if not framework or not model_path:
        raise RuntimeError(
            "app_from_env requires OPHELIAN_FRAMEWORK and OPHELIAN_MODEL_PATH "
            "environment variables to be set."
        )
    enable_prom = _truthy_env(os.environ.get("OPHELIAN_PROMETHEUS"))
    # Drift hooks (Task #30) — env-driven so the container deploy
    # path has parity with the in-process ``build_app`` kwargs.
    baseline_path = os.environ.get("OPHELIAN_DRIFT_BASELINE_PATH") or None
    drift_endpoint = os.environ.get("OPHELIAN_DRIFT_ENDPOINT") or None
    window_size = int(os.environ.get("OPHELIAN_DRIFT_WINDOW_SIZE", "200"))
    stride_env = os.environ.get("OPHELIAN_DRIFT_STRIDE")
    stride = int(stride_env) if stride_env else None
    test = os.environ.get("OPHELIAN_DRIFT_TEST", "ks")
    threshold = float(os.environ.get("OPHELIAN_DRIFT_THRESHOLD", "0.05"))
    return build_app(
        framework=framework,
        model_path=model_path,
        enable_prometheus=enable_prom,
        drift_baseline_path=baseline_path,
        drift_endpoint=drift_endpoint,
        drift_window_size=window_size,
        drift_stride=stride,
        drift_test=test,
        drift_threshold=threshold,
    )


class FastAPIRuntime:
    """Convenience wrapper used by the standalone provider's deploy step."""

    def __init__(
        self,
        *,
        framework: str,
        model_path: str | Path,
        enable_prometheus: bool = False,
    ) -> None:
        self.framework = framework
        self.model_path = Path(model_path)
        self.enable_prometheus = enable_prometheus

    def app(self) -> FastAPI:
        return build_app(
            framework=self.framework,
            model_path=self.model_path,
            enable_prometheus=self.enable_prometheus,
        )

    def serve(self, *, host: str = "0.0.0.0", port: int = 8000) -> None:  # pragma: no cover
        import uvicorn

        uvicorn.run(self.app(), host=host, port=port)
