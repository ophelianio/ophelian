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

METRIC_PIPELINE_RUNS = "ophelian.pipeline.runs"
METRIC_STEP_DURATION = "ophelian.step.duration"
METRIC_SERVE_REQUESTS = "ophelian.serve.requests"
METRIC_SERVE_LATENCY = "ophelian.serve.latency"

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
            if readers:
                metrics.set_meter_provider(MeterProvider(resource=resource, metric_readers=readers))
        except Exception:  # pragma: no cover - defensive
            logger.debug("Meter auto-configuration failed", exc_info=True)


def _reset_auto_configuration_for_tests() -> None:
    """Test hook — clears the idempotency latch."""
    global _auto_configured, _pipeline_runs, _step_duration, _serve_requests, _serve_latency
    _auto_configured = False
    _pipeline_runs = None
    _step_duration = None
    _serve_requests = None
    _serve_latency = None


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
        description="Inference request latency in seconds.",
        unit="s",
    )
    return _serve_latency


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


def record_serve_outcome(
    *,
    method: str,
    route: str,
    status_code: int,
    duration_seconds: float,
    span: Any | None = None,
) -> None:
    """Tag the serve span with ``http.status_code`` and feed metrics."""
    if span is not None and hasattr(span, "set_attribute"):
        span.set_attribute("http.status_code", int(status_code))
        span.set_attribute(
            ATTR_STATUS,
            "success" if 200 <= status_code < 400 else "failed",
        )
    counter = _serve_requests_counter()
    histogram = _serve_latency_histogram()
    dim = _attrs(
        {
            "http.method": method,
            "http.route": route,
            "http.status_code": int(status_code),
        }
    )
    if counter is not None:
        counter.add(1, attributes=dim)
    if histogram is not None:
        histogram.record(max(duration_seconds, 0.0), attributes=dim)


__all__ = [
    "ATTR_ENV_CLASS",
    "ATTR_FRAMEWORK",
    "ATTR_PIPELINE_NAME",
    "ATTR_PROVIDER",
    "ATTR_REGION",
    "ATTR_RUN_ID",
    "ATTR_STATUS",
    "ATTR_STEP_KIND",
    "ATTR_STEP_NAME",
    "INSTRUMENTATION_NAME",
    "METRIC_PIPELINE_RUNS",
    "METRIC_SERVE_LATENCY",
    "METRIC_SERVE_REQUESTS",
    "METRIC_STEP_DURATION",
    "auto_configure_from_env",
    "get_meter",
    "get_tracer",
    "is_otel_available",
    "pipeline_span",
    "record_pipeline_outcome",
    "record_serve_outcome",
    "record_step_outcome",
    "serve_request_span",
    "step_span",
]
