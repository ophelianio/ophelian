"""OpenTelemetry first-class instrumentation backbone.

This module promotes OpenTelemetry from the optional ``emit_event``
escape-hatch in :mod:`ophelian.observability` to a real, opinionated
instrumentation layer. When the ``[otel]`` extra is installed and the
standard OTel environment variables are set (e.g.
``OTEL_EXPORTER_OTLP_ENDPOINT``), every Ophelian pipeline run, every
step, and every served inference request emits a span and feeds a
small, stable set of metrics — with **zero behaviour change** for
users who do not install the extra.

Public attribute schema
-----------------------
The following attribute names are the **public contract** that any
downstream consumer (a dashboard, an alerting rule, a SaaS that
ingests Ophelian telemetry) is allowed to depend on. They are stable
across the v1.x line:

* ``ophelian.run_id``         — the per-run correlation id
* ``ophelian.pipeline.name``  — pipeline name
* ``ophelian.step.name``      — step name (child spans only)
* ``ophelian.step.kind``      — step kind (``data``/``train``/...)
* ``ophelian.env.class``      — ``standalone`` / ``aws`` / ``gcp`` / ``azure`` / ``auto`` / ``eks`` / ``gke`` / ``aks``
* ``ophelian.provider``       — provider name (often equals env class)
* ``ophelian.region``         — cloud region or ``"local"``
* ``ophelian.status``         — terminal status: ``success`` / ``failed`` / ``skipped``
* ``ophelian.framework``      — ML framework (serve spans only)

Standard HTTP semantic conventions are used for serve-request spans
(``http.method``, ``http.route``, ``http.status_code``).

Metric names
------------
* ``ophelian.pipeline.runs``       — counter, dim: ``pipeline``, ``status``
* ``ophelian.step.duration``       — histogram (seconds), dim: ``step``, ``kind``, ``status``
* ``ophelian.serve.requests``      — counter, dim: ``route``, ``method``, ``status``
* ``ophelian.serve.latency``       — histogram (seconds), dim: ``route``, ``method``, ``status``
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import time
from collections.abc import Iterator, Mapping
from typing import Any

logger = logging.getLogger("ophelian.observability.otel")

# ----------------------------------------------------------------------
# Public attribute schema (stable contract)
# ----------------------------------------------------------------------

ATTR_RUN_ID = "ophelian.run_id"
ATTR_PIPELINE_NAME = "ophelian.pipeline.name"
ATTR_STEP_NAME = "ophelian.step.name"
ATTR_STEP_KIND = "ophelian.step.kind"
ATTR_ENV_CLASS = "ophelian.env.class"
ATTR_PROVIDER = "ophelian.provider"
ATTR_REGION = "ophelian.region"
ATTR_STATUS = "ophelian.status"
ATTR_FRAMEWORK = "ophelian.framework"
# HTTP status class is the SRE-friendly bucket — "2xx" / "3xx" /
# "4xx" / "5xx". It keeps alerting rules cheap to author (no need to
# enumerate every status_code) while keeping cardinality bounded.
ATTR_HTTP_STATUS_CLASS = "http.status_class"

METRIC_PIPELINE_RUNS = "ophelian.pipeline.runs"
METRIC_STEP_DURATION = "ophelian.step.duration"
METRIC_SERVE_REQUESTS = "ophelian.serve.requests"
METRIC_SERVE_LATENCY = "ophelian.serve.latency"
METRIC_SERVE_INFERENCE_DURATION = "ophelian.serve.inference.duration"
METRIC_SERVE_INFLIGHT = "ophelian.serve.inflight"
METRIC_SERVE_QUEUE_DEPTH = "ophelian.serve.queue.depth"
METRIC_SERVE_IDLE_SECONDS = "ophelian.serve.idle.seconds"
METRIC_SERVE_TOKENS_IN = "ophelian.serve.tokens.in"
METRIC_SERVE_TOKENS_OUT = "ophelian.serve.tokens.out"

INSTRUMENTATION_NAME = "ophelian"

# ----------------------------------------------------------------------
# Lazy / safe OTel availability
# ----------------------------------------------------------------------


def is_otel_available() -> bool:
    """Return ``True`` when the ``opentelemetry`` package is importable."""
    try:
        import opentelemetry  # noqa: F401
    except ImportError:
        return False
    return True


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


# ----------------------------------------------------------------------
# Auto-configuration
# ----------------------------------------------------------------------

_auto_configured: bool = False


def auto_configure_from_env() -> None:
    """Best-effort wire-up of the global TracerProvider / MeterProvider.

    Honours the standard OTel environment variables:

    * ``OTEL_EXPORTER_OTLP_ENDPOINT`` — when set, install an OTLP HTTP
      exporter for both traces and metrics (provided the corresponding
      exporter package is installed). Without an endpoint we still
      install no-op providers so the rest of the codebase can call
      ``get_tracer`` / ``get_meter`` without conditional imports.
    * ``OTEL_SERVICE_NAME`` — service.name resource attribute
      (defaults to ``"ophelian"``).
    * ``OPHELIAN_OTEL_DISABLE=1`` — bypass auto-configuration entirely
      (escape hatch for users who manage providers themselves).
    * ``OPHELIAN_OTEL_CONSOLE=1`` — install console exporters even
      without an OTLP endpoint (useful for local debugging).

    Idempotent: subsequent calls are no-ops. Never overrides a
    pre-configured non-default tracer provider — if the user (or
    ``opentelemetry-instrument``) already wired one up, we respect it.
    """
    global _auto_configured
    if _auto_configured:
        return
    _auto_configured = True

    if _truthy(os.environ.get("OPHELIAN_OTEL_DISABLE")):
        return
    if not is_otel_available():
        return

    try:
        from opentelemetry import metrics, trace
        from opentelemetry.sdk.resources import Resource
    except ImportError:
        return

    service_name = os.environ.get("OTEL_SERVICE_NAME", "ophelian")
    resource = Resource.create({"service.name": service_name})
    otlp_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    want_console = _truthy(os.environ.get("OPHELIAN_OTEL_CONSOLE"))

    # ---- Tracer provider --------------------------------------------------
    current_tp = trace.get_tracer_provider()
    # The default ProxyTracerProvider has no ``add_span_processor``;
    # any real provider already in place we must leave alone.
    if not hasattr(current_tp, "add_span_processor"):
        try:
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import (
                BatchSpanProcessor,
                ConsoleSpanExporter,
            )

            tp = TracerProvider(resource=resource)
            installed_any = False
            if otlp_endpoint:
                try:
                    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                        OTLPSpanExporter,
                    )

                    tp.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
                    installed_any = True
                except ImportError:
                    logger.debug(
                        "OTEL_EXPORTER_OTLP_ENDPOINT is set but "
                        "opentelemetry-exporter-otlp-proto-http is not installed; "
                        "spans will not be exported."
                    )
            if want_console:
                tp.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
                installed_any = True
            if installed_any:
                trace.set_tracer_provider(tp)
        except Exception:  # pragma: no cover - defensive
            logger.debug("Tracer auto-configuration failed", exc_info=True)

    # ---- Meter provider ---------------------------------------------------
    current_mp = metrics.get_meter_provider()
    _proxy_or_noop_mp_names = {
        "_ProxyMeterProvider",
        "ProxyMeterProvider",
        "NoOpMeterProvider",
        "_NoOpMeterProvider",
        "DefaultMeterProvider",
    }
    if type(current_mp).__name__ in _proxy_or_noop_mp_names:
        try:
            from opentelemetry.sdk.metrics import MeterProvider
            from opentelemetry.sdk.metrics.export import (
                ConsoleMetricExporter,
                PeriodicExportingMetricReader,
            )

            readers: list[Any] = []
            if otlp_endpoint:
                try:
                    from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
                        OTLPMetricExporter,
                    )

                    readers.append(PeriodicExportingMetricReader(OTLPMetricExporter()))
                except ImportError:
                    logger.debug(
                        "OTEL_EXPORTER_OTLP_ENDPOINT is set but "
                        "opentelemetry-exporter-otlp-proto-http is not installed; "
                        "metrics will not be exported."
                    )
            if want_console:
                readers.append(PeriodicExportingMetricReader(ConsoleMetricExporter()))
            # Optional Prometheus scrape endpoint reader. The reader
            # registers itself with ``prometheus_client.REGISTRY``;
            # the FastAPI runtime serves that registry at ``/metrics``
            # when the user opts in via ``enable_prometheus`` /
            # ``OPHELIAN_PROMETHEUS=1``.
            if _truthy(os.environ.get("OPHELIAN_OTEL_PROMETHEUS")):
                try:
                    from opentelemetry.exporter.prometheus import (
                        PrometheusMetricReader,
                    )

                    readers.append(PrometheusMetricReader())
                except ImportError:
                    logger.debug(
                        "OPHELIAN_OTEL_PROMETHEUS is set but "
                        "opentelemetry-exporter-prometheus is not installed; "
                        "skipping the Prometheus scrape endpoint reader."
                    )
            if readers:
                metrics.set_meter_provider(MeterProvider(resource=resource, metric_readers=readers))
        except Exception:  # pragma: no cover - defensive
            logger.debug("Meter auto-configuration failed", exc_info=True)


def _reset_auto_configuration_for_tests() -> None:
    """Test hook — clears the idempotency latch."""
    global _auto_configured, _pipeline_runs, _step_duration, _serve_requests, _serve_latency
    global _serve_inference_duration, _serve_inflight, _serve_idle_seconds
    global _serve_tokens_in, _serve_tokens_out, _serve_queue_depth, _serve_queue_callbacks
    _auto_configured = False
    _pipeline_runs = None
    _step_duration = None
    _serve_requests = None
    _serve_latency = None
    _serve_inference_duration = None
    _serve_inflight = None
    _serve_idle_seconds = None
    _serve_tokens_in = None
    _serve_tokens_out = None
    _serve_queue_depth = None
    _serve_queue_callbacks = []


# ----------------------------------------------------------------------
# Tracer / meter accessors
# ----------------------------------------------------------------------


def get_tracer() -> Any:
    """Return the Ophelian tracer, or ``None`` when OTel is unavailable.

    Calling code should use :func:`pipeline_span` / :func:`step_span` /
    :func:`serve_request_span` rather than dealing with the tracer
    directly — those helpers are no-ops when this returns ``None``.
    """
    if not is_otel_available():
        return None
    try:
        from opentelemetry import trace

        return trace.get_tracer(INSTRUMENTATION_NAME)
    except Exception:  # pragma: no cover - defensive
        return None


def get_meter() -> Any:
    """Return the Ophelian meter, or ``None`` when OTel is unavailable."""
    if not is_otel_available():
        return None
    try:
        from opentelemetry import metrics

        return metrics.get_meter(INSTRUMENTATION_NAME)
    except Exception:  # pragma: no cover - defensive
        return None


# ----------------------------------------------------------------------
# Metric singletons (lazy)
# ----------------------------------------------------------------------

_pipeline_runs: Any = None
_step_duration: Any = None
_serve_requests: Any = None
_serve_latency: Any = None
_serve_inference_duration: Any = None
_serve_inflight: Any = None
_serve_idle_seconds: Any = None
_serve_tokens_in: Any = None
_serve_tokens_out: Any = None
# Queue-depth callbacks registered per (route, callable) pair. The
# observable gauge is created lazily on first registration so users
# who never wire a queue pay zero cost.
_serve_queue_depth: Any = None
_serve_queue_callbacks: list[Any] = []


def _pipeline_runs_counter() -> Any:
    global _pipeline_runs
    if _pipeline_runs is not None:
        return _pipeline_runs
    meter = get_meter()
    if meter is None:
        return None
    _pipeline_runs = meter.create_counter(
        name=METRIC_PIPELINE_RUNS,
        description="Total Ophelian pipeline runs, by terminal status.",
        unit="1",
    )
    return _pipeline_runs


def _step_duration_histogram() -> Any:
    global _step_duration
    if _step_duration is not None:
        return _step_duration
    meter = get_meter()
    if meter is None:
        return None
    _step_duration = meter.create_histogram(
        name=METRIC_STEP_DURATION,
        description="Step duration in seconds.",
        unit="s",
    )
    return _step_duration


def _serve_requests_counter() -> Any:
    global _serve_requests
    if _serve_requests is not None:
        return _serve_requests
    meter = get_meter()
    if meter is None:
        return None
    _serve_requests = meter.create_counter(
        name=METRIC_SERVE_REQUESTS,
        description="Total inference requests served.",
        unit="1",
    )
    return _serve_requests


def _serve_latency_histogram() -> Any:
    global _serve_latency
    if _serve_latency is not None:
        return _serve_latency
    meter = get_meter()
    if meter is None:
        return None
    _serve_latency = meter.create_histogram(
        name=METRIC_SERVE_LATENCY,
        description=(
            "Total request latency in seconds (HTTP-in to HTTP-out, "
            "includes inference + serialization + framework overhead)."
        ),
        unit="s",
    )
    return _serve_latency


def _serve_inference_duration_histogram() -> Any:
    global _serve_inference_duration
    if _serve_inference_duration is not None:
        return _serve_inference_duration
    meter = get_meter()
    if meter is None:
        return None
    _serve_inference_duration = meter.create_histogram(
        name=METRIC_SERVE_INFERENCE_DURATION,
        description=(
            "Wall-clock time spent inside the model adapter's predict "
            "call, in seconds. Excludes HTTP serialization and "
            "framework overhead — subtract from "
            "``ophelian.serve.latency`` to derive that overhead."
        ),
        unit="s",
    )
    return _serve_inference_duration


def _serve_inflight_updown_counter() -> Any:
    global _serve_inflight
    if _serve_inflight is not None:
        return _serve_inflight
    meter = get_meter()
    if meter is None:
        return None
    _serve_inflight = meter.create_up_down_counter(
        name=METRIC_SERVE_INFLIGHT,
        description="Concurrent in-flight requests currently being served.",
        unit="1",
    )
    return _serve_inflight


def _serve_idle_counter() -> Any:
    global _serve_idle_seconds
    if _serve_idle_seconds is not None:
        return _serve_idle_seconds
    meter = get_meter()
    if meter is None:
        return None
    _serve_idle_seconds = meter.create_counter(
        name=METRIC_SERVE_IDLE_SECONDS,
        description=(
            "Cumulative seconds the endpoint had zero in-flight "
            "requests. Useful for autoscaler reclaim-on-idle policies."
        ),
        unit="s",
    )
    return _serve_idle_seconds


def _serve_tokens_in_counter() -> Any:
    global _serve_tokens_in
    if _serve_tokens_in is not None:
        return _serve_tokens_in
    meter = get_meter()
    if meter is None:
        return None
    _serve_tokens_in = meter.create_counter(
        name=METRIC_SERVE_TOKENS_IN,
        description=(
            "Total prompt / input tokens consumed by inference "
            "requests. Only emitted when the model adapter exposes a "
            "token usage shape (e.g. OpenAI-compatible "
            "``response.usage.prompt_tokens``)."
        ),
        unit="1",
    )
    return _serve_tokens_in


def _serve_tokens_out_counter() -> Any:
    global _serve_tokens_out
    if _serve_tokens_out is not None:
        return _serve_tokens_out
    meter = get_meter()
    if meter is None:
        return None
    _serve_tokens_out = meter.create_counter(
        name=METRIC_SERVE_TOKENS_OUT,
        description=(
            "Total completion / output tokens produced by inference "
            "requests. Only emitted when the model adapter exposes a "
            "token usage shape."
        ),
        unit="1",
    )
    return _serve_tokens_out


def register_queue_depth_observer(
    callback: Any, *, route: str | None = None
) -> None:
    """Register an observable callback for ``ophelian.serve.queue.depth``.

    The callback is invoked by the OTel meter on each collection cycle
    and must return the current queue depth (int / float). When no
    queue is wired, the gauge is simply not populated.

    Multiple endpoints can register independently; we route their
    samples by the optional ``route`` label.
    """
    global _serve_queue_depth
    meter = get_meter()
    if meter is None:
        return
    try:
        from opentelemetry.metrics import CallbackOptions, Observation
    except ImportError:  # pragma: no cover - OTel API >=1.20 ships these
        return

    def _wrapped(_options: CallbackOptions) -> Iterator[Any]:
        try:
            value = float(callback())
        except Exception:  # pragma: no cover - defensive
            return
        attrs = _attrs({"http.route": route}) if route else {}
        yield Observation(value, attrs)

    _serve_queue_callbacks.append(_wrapped)
    if _serve_queue_depth is None:
        _serve_queue_depth = meter.create_observable_gauge(
            name=METRIC_SERVE_QUEUE_DEPTH,
            callbacks=_serve_queue_callbacks,
            description=(
                "Current depth of any request queue in front of the "
                "endpoint. Only populated when the runtime registers "
                "a queue-depth callback."
            ),
            unit="1",
        )


# ----------------------------------------------------------------------
# Span helpers (context managers)
# ----------------------------------------------------------------------


def _otel_safe(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return json.dumps(value, default=str)
    if isinstance(value, (list, tuple)):
        return [_otel_safe(v) for v in value]
    return repr(value)


def _attrs(items: Mapping[str, Any]) -> dict[str, Any]:
    return {k: _otel_safe(v) for k, v in items.items() if v is not None}


@contextlib.contextmanager
def pipeline_span(
    *,
    pipeline_name: str,
    run_id: str | None = None,
    env_class: str | None = None,
    provider: str | None = None,
    region: str | None = None,
    emit_metric: bool = True,
) -> Iterator[Any]:
    """Open the parent span for one pipeline run.

    Span lifetime is the with-body. On exit:

    - If an exception escapes the body, the span's ``ophelian.status``
      attribute is set to ``"failed"``.
    - Otherwise the caller is responsible for setting ``ophelian.status``
      via ``span.set_attribute(ATTR_STATUS, ...)`` to reflect the
      authoritative outcome (``PipelineResult.succeeded``).
    - If ``emit_metric`` is ``True`` (default), the
      ``ophelian.pipeline.runs`` counter is incremented exactly once
      with the inferred status. Callers who emit the counter themselves
      via :func:`record_pipeline_outcome` MUST pass ``emit_metric=False``
      to avoid double-counting.
    """
    auto_configure_from_env()
    tracer = get_tracer()
    if tracer is None:
        yield None
        return
    attributes = _attrs(
        {
            ATTR_PIPELINE_NAME: pipeline_name,
            ATTR_RUN_ID: run_id,
            ATTR_ENV_CLASS: env_class,
            ATTR_PROVIDER: provider,
            ATTR_REGION: region,
        }
    )
    raised = False
    try:
        with tracer.start_as_current_span(
            f"ophelian.pipeline.{pipeline_name}",
            attributes=attributes,
        ) as span:
            try:
                yield span
            except BaseException as exc:
                raised = True
                if span is not None:
                    if hasattr(span, "record_exception"):
                        span.record_exception(exc)
                    if hasattr(span, "set_attribute"):
                        span.set_attribute(ATTR_STATUS, "failed")
                raise
    finally:
        if emit_metric:
            record_pipeline_outcome(
                pipeline_name=pipeline_name,
                status="failed" if raised else "success",
                provider=provider,
            )


@contextlib.contextmanager
def step_span(
    *,
    step_name: str,
    step_kind: str,
    run_id: str | None = None,
    provider: str | None = None,
    emit_metric: bool = True,
) -> Iterator[Any]:
    """Open a child span for one step inside a pipeline run.

    On exit, errors are recorded on the span and re-raised. If
    ``emit_metric`` is ``True`` (default), records
    ``ophelian.step.duration`` (seconds) tagged with the step name,
    kind, and inferred terminal status (``success`` if no exception,
    ``failed`` otherwise).

    Callers that emit the duration metric themselves via
    :func:`record_step_outcome` (e.g. providers that derive status from
    ``StepResult.status`` rather than exception escape) MUST pass
    ``emit_metric=False`` to avoid double-counting.

    Cardinality contract: exactly **one** ``ophelian.step.duration``
    sample is recorded per step execution, on the side that owns the
    handler. The in-process :class:`StandaloneProvider` owns it on the
    host (via ``record_step_outcome``); the cloud worker owns it inside
    the worker process (via this helper with the default
    ``emit_metric=True``). Host-side container drivers must not also
    emit a duration sample for container-executed steps — that would
    double-count. Regression tests
    ``test_step_metrics_emitted_exactly_once_per_step`` and
    ``test_step_span_records_failed_duration_when_handler_raises``
    pin both halves of the contract.
    """
    auto_configure_from_env()
    tracer = get_tracer()
    started = time.monotonic()
    raised = False
    if tracer is None:
        try:
            yield None
        except BaseException:
            raised = True
            raise
        finally:
            if emit_metric:
                _record_step_duration(
                    step_name,
                    step_kind,
                    "failed" if raised else "success",
                    time.monotonic() - started,
                )
        return
    attributes = _attrs(
        {
            ATTR_STEP_NAME: step_name,
            ATTR_STEP_KIND: step_kind,
            ATTR_RUN_ID: run_id,
            ATTR_PROVIDER: provider,
        }
    )
    try:
        with tracer.start_as_current_span(
            f"ophelian.step.{step_name}",
            attributes=attributes,
        ) as span:
            try:
                yield span
            except BaseException as exc:
                raised = True
                if span is not None:
                    if hasattr(span, "record_exception"):
                        span.record_exception(exc)
                    if hasattr(span, "set_attribute"):
                        span.set_attribute(ATTR_STATUS, "failed")
                raise
    finally:
        if emit_metric:
            _record_step_duration(
                step_name,
                step_kind,
                "failed" if raised else "success",
                time.monotonic() - started,
            )


def record_step_outcome(
    *,
    step_name: str,
    step_kind: str,
    status: str,
    duration_seconds: float,
) -> None:
    """Record a step duration without owning the span lifecycle.

    Used by call sites where the span was opened around a wider try
    block (e.g. provider executors that translate exceptions into a
    failed ``StepResult`` rather than re-raising). Honours the same
    attribute schema as :func:`step_span`.
    """
    _record_step_duration(step_name, step_kind, status, duration_seconds)


def record_pipeline_outcome(
    *,
    pipeline_name: str,
    status: str,
    provider: str | None = None,
) -> None:
    """Increment the ``ophelian.pipeline.runs`` counter exactly once
    with the authoritative status the caller derived (typically from
    ``PipelineResult.succeeded``)."""
    counter = _pipeline_runs_counter()
    if counter is None:
        return
    counter.add(
        1,
        attributes=_attrs(
            {
                ATTR_PIPELINE_NAME: pipeline_name,
                ATTR_STATUS: status,
                ATTR_PROVIDER: provider,
            }
        ),
    )


def _record_step_duration(
    step_name: str,
    step_kind: str,
    status: str,
    duration_seconds: float,
) -> None:
    histogram = _step_duration_histogram()
    if histogram is None:
        return
    histogram.record(
        max(duration_seconds, 0.0),
        attributes=_attrs(
            {
                ATTR_STEP_NAME: step_name,
                ATTR_STEP_KIND: step_kind,
                ATTR_STATUS: status,
            }
        ),
    )


@contextlib.contextmanager
def serve_request_span(
    *,
    method: str,
    route: str,
    framework: str | None = None,
) -> Iterator[Any]:
    """Open a span for one inference request.

    The status code is not known up front — the caller sets it on
    exit via :func:`record_serve_outcome`. We emit the span name as
    ``ophelian.serve.{method} {route}`` so backends bucket per route.
    """
    auto_configure_from_env()
    tracer = get_tracer()
    if tracer is None:
        yield None
        return
    attributes = _attrs(
        {
            "http.method": method,
            "http.route": route,
            ATTR_FRAMEWORK: framework,
        }
    )
    with tracer.start_as_current_span(
        f"ophelian.serve.{method} {route}",
        attributes=attributes,
    ) as span:
        yield span


def _status_class(status_code: int) -> str:
    """Return ``"2xx"`` / ``"3xx"`` / ``"4xx"`` / ``"5xx"`` for a status code.

    Anything outside the 100..599 range is bucketed as ``"unknown"`` —
    we'd rather see a clear "unknown" data point than silently lie.
    """
    code = int(status_code)
    if 100 <= code <= 599:
        return f"{code // 100}xx"
    return "unknown"


def record_serve_outcome(
    *,
    method: str,
    route: str,
    status_code: int,
    duration_seconds: float,
    span: Any | None = None,
    inference_duration_seconds: float | None = None,
) -> None:
    """Tag the serve span and feed the per-request metrics.

    Records ``ophelian.serve.requests`` (counter) and
    ``ophelian.serve.latency`` (histogram, total HTTP time). When
    ``inference_duration_seconds`` is supplied, also records
    ``ophelian.serve.inference.duration`` so SREs can separate model
    work from HTTP / serialization overhead. The
    :data:`ATTR_HTTP_STATUS_CLASS` label is added to all three so
    alerting rules can target ``5xx`` without enumerating codes.
    """
    status_class = _status_class(status_code)
    if span is not None and hasattr(span, "set_attribute"):
        span.set_attribute("http.status_code", int(status_code))
        span.set_attribute(ATTR_HTTP_STATUS_CLASS, status_class)
        span.set_attribute(
            ATTR_STATUS,
            "success" if 200 <= int(status_code) < 400 else "failed",
        )
    dim = _attrs(
        {
            "http.method": method,
            "http.route": route,
            "http.status_code": int(status_code),
            ATTR_HTTP_STATUS_CLASS: status_class,
        }
    )
    counter = _serve_requests_counter()
    histogram = _serve_latency_histogram()
    if counter is not None:
        counter.add(1, attributes=dim)
    if histogram is not None:
        histogram.record(max(duration_seconds, 0.0), attributes=dim)
    if inference_duration_seconds is not None:
        inf_hist = _serve_inference_duration_histogram()
        if inf_hist is not None:
            inf_hist.record(max(inference_duration_seconds, 0.0), attributes=dim)


def serve_inflight_inc(*, method: str, route: str) -> None:
    """Increment the in-flight request gauge for one request entering."""
    counter = _serve_inflight_updown_counter()
    if counter is None:
        return
    counter.add(1, attributes=_attrs({"http.method": method, "http.route": route}))


def serve_inflight_dec(*, method: str, route: str) -> None:
    """Decrement the in-flight request gauge after a request finishes."""
    counter = _serve_inflight_updown_counter()
    if counter is None:
        return
    counter.add(-1, attributes=_attrs({"http.method": method, "http.route": route}))


def record_serve_idle(*, route: str | None, idle_seconds: float) -> None:
    """Add to the cumulative idle-time counter for an endpoint."""
    if idle_seconds <= 0.0:
        return
    counter = _serve_idle_counter()
    if counter is None:
        return
    attrs = _attrs({"http.route": route}) if route else {}
    counter.add(float(idle_seconds), attributes=attrs)


def _coerce_token_count(value: Any) -> int | None:
    """Best-effort numeric coercion for token-usage payloads.

    Some LLM SDKs return tokens as ``int``, some as ``float``, and a
    few wire-protocol implementations send them as numeric strings.
    We accept all three and return ``None`` for anything else — the
    metric stays silent rather than emit a fabricated zero.
    """
    if value is None:
        return None
    if isinstance(value, bool):  # bool is an int subclass — exclude it
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return None
    return None


def extract_token_usage(
    prediction: Any, payload: Any | None = None
) -> tuple[int | None, int | None]:
    """Best-effort detection of ``(tokens_in, tokens_out)`` from common
    LLM adapter return shapes.

    Recognised shapes (first match wins):

    * **OpenAI-compatible**: ``{"usage": {"prompt_tokens": N,
      "completion_tokens": M}}``.
    * **Anthropic-compatible**: ``{"usage": {"input_tokens": N,
      "output_tokens": M}}``.
    * **Explicit override**: top-level ``{"tokens_in": N,
      "tokens_out": M}`` — for adapter authors who want to feed the
      counters directly.

    Adapter authors writing a custom :class:`ModelAdapter` can call
    this helper from their own runtime integration to benefit from
    the same detection logic the FastAPI runtime uses.

    Returns ``(None, None)`` for any unrecognised shape — guessing
    would be worse than no signal.
    """
    if not isinstance(prediction, dict):
        return (None, None)
    if "tokens_in" in prediction or "tokens_out" in prediction:
        return (
            _coerce_token_count(prediction.get("tokens_in")),
            _coerce_token_count(prediction.get("tokens_out")),
        )
    usage = prediction.get("usage")
    if isinstance(usage, dict):
        tin_val = usage.get("prompt_tokens", usage.get("input_tokens"))
        tout_val = usage.get("completion_tokens", usage.get("output_tokens"))
        return (_coerce_token_count(tin_val), _coerce_token_count(tout_val))
    del payload  # reserved for future heuristics (e.g. tokenizing the prompt)
    return (None, None)


def record_inference_tokens(
    *,
    method: str,
    route: str,
    framework: str | None,
    tokens_in: int | None,
    tokens_out: int | None,
) -> None:
    """Record per-request token usage when the adapter exposed it."""
    dim = _attrs({"http.method": method, "http.route": route, ATTR_FRAMEWORK: framework})
    if tokens_in is not None:
        ctr = _serve_tokens_in_counter()
        if ctr is not None:
            ctr.add(int(tokens_in), attributes=dim)
    if tokens_out is not None:
        ctr = _serve_tokens_out_counter()
        if ctr is not None:
            ctr.add(int(tokens_out), attributes=dim)


__all__ = [
    "ATTR_ENV_CLASS",
    "ATTR_FRAMEWORK",
    "ATTR_HTTP_STATUS_CLASS",
    "ATTR_PIPELINE_NAME",
    "ATTR_PROVIDER",
    "ATTR_REGION",
    "ATTR_RUN_ID",
    "ATTR_STATUS",
    "ATTR_STEP_KIND",
    "ATTR_STEP_NAME",
    "INSTRUMENTATION_NAME",
    "METRIC_PIPELINE_RUNS",
    "METRIC_SERVE_IDLE_SECONDS",
    "METRIC_SERVE_INFERENCE_DURATION",
    "METRIC_SERVE_INFLIGHT",
    "METRIC_SERVE_LATENCY",
    "METRIC_SERVE_QUEUE_DEPTH",
    "METRIC_SERVE_REQUESTS",
    "METRIC_SERVE_TOKENS_IN",
    "METRIC_SERVE_TOKENS_OUT",
    "METRIC_STEP_DURATION",
    "auto_configure_from_env",
    "extract_token_usage",
    "get_meter",
    "get_tracer",
    "is_otel_available",
    "pipeline_span",
    "record_inference_tokens",
    "record_pipeline_outcome",
    "record_serve_idle",
    "record_serve_outcome",
    "record_step_outcome",
    "register_queue_depth_observer",
    "serve_inflight_dec",
    "serve_inflight_inc",
    "serve_request_span",
    "step_span",
]
