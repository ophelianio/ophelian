"""Observability surface — structured logging + run-id correlation.

The library emits a single, configurable logger (``ophelian.*``) and
threads a per-run ``run_id`` through a :class:`~contextvars.ContextVar`
so every log line — whether emitted from the orchestrator, a driver,
the step_runner inside a container, or a model adapter — carries the
identifier that ties them all together.

Usage from library code::

    from ophelian.observability import bind_run, get_run_id, configure_logging

    configure_logging(json=True)
    with bind_run("run-abc123"):
        log_something()  # JSON line will contain "run_id": "run-abc123"

The :func:`emit_event` helper is a no-op unless OpenTelemetry is
installed and a tracer provider has been configured — that lets us
ship the OTel hook in v1.0 without forcing the dependency.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import sys
import time
from collections.abc import Iterator, Mapping
from contextvars import ContextVar
from typing import Any

# ----------------------------------------------------------------------
# Run-id context
# ----------------------------------------------------------------------

_run_id_ctx: ContextVar[str | None] = ContextVar("ophelian_run_id", default=None)


def get_run_id() -> str | None:
    """Return the current run id, or ``None`` when not bound."""
    return _run_id_ctx.get()


@contextlib.contextmanager
def bind_run(run_id: str) -> Iterator[None]:
    """Bind *run_id* for the duration of the ``with`` block."""
    token = _run_id_ctx.set(run_id)
    try:
        yield
    finally:
        _run_id_ctx.reset(token)


def set_run_id(run_id: str | None) -> None:
    """Bind *run_id* for the rest of this thread/task (no automatic reset).

    Prefer :func:`bind_run` from library code; this is for the
    ``step_runner`` entry point which has a single linear lifecycle.
    """
    _run_id_ctx.set(run_id)


# ----------------------------------------------------------------------
# JSON formatter
# ----------------------------------------------------------------------


_RESERVED = {
    "name",
    "msg",
    "args",
    "levelname",
    "levelno",
    "pathname",
    "filename",
    "module",
    "exc_info",
    "exc_text",
    "stack_info",
    "lineno",
    "funcName",
    "created",
    "msecs",
    "relativeCreated",
    "thread",
    "threadName",
    "processName",
    "process",
    "taskName",
    "message",
}


class JSONFormatter(logging.Formatter):
    """Minimal JSON log formatter with run_id correlation.

    Output shape::

        {"ts": "2026-05-01T12:34:56Z", "level": "INFO",
         "logger": "ophelian.providers.aws", "msg": "...",
         "run_id": "run-abc", "extra": {...}}
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        run_id = get_run_id()
        if run_id is not None:
            payload["run_id"] = run_id
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        # Anything the caller passed via `extra=` — we filter the
        # built-ins so the line stays small and predictable.
        extras: dict[str, Any] = {}
        for key, value in record.__dict__.items():
            if key in _RESERVED:
                continue
            if key.startswith("_"):
                continue
            try:
                json.dumps(value)
                extras[key] = value
            except TypeError:
                extras[key] = repr(value)
        if extras:
            payload["extra"] = extras
        return json.dumps(payload, default=str)


# ----------------------------------------------------------------------
# Logger configuration
# ----------------------------------------------------------------------


def _resolve_level(level: int | str | None) -> int:
    if level is None:
        env = os.environ.get("OPHELIAN_LOG_LEVEL", "INFO")
        resolved = logging.getLevelName(env.upper()) if isinstance(env, str) else int(env)
        return int(resolved) if isinstance(resolved, int) else logging.INFO
    if isinstance(level, str):
        resolved = logging.getLevelName(level.upper())
        return int(resolved) if isinstance(resolved, int) else logging.INFO
    return int(level)


def configure_logging(
    *,
    json: bool | None = None,
    level: int | str | None = None,
    stream: Any | None = None,
) -> logging.Logger:
    """Install a single handler on the ``ophelian`` logger.

    Re-callable: the handler is replaced rather than appended so tests
    can flip between JSON and human-readable mode without leaking
    duplicate output.

    Resolution rules
    ----------------
    * ``json=True``  -> JSON formatter (regardless of env)
    * ``json=False`` -> human formatter (regardless of env)
    * ``json=None``  -> JSON when ``OPHELIAN_LOG_FORMAT=json`` is set, else human
    * ``level``      -> log level (defaults to ``OPHELIAN_LOG_LEVEL`` or INFO)

    Default stream
    --------------
    JSON logs go to ``stdout`` (so log shippers / ``jq`` pipelines can
    read them on the standard data channel) and human logs go to
    ``stderr`` (so they don't pollute a script's stdout). Pass an
    explicit ``stream=`` to override.
    """
    logger = logging.getLogger("ophelian")
    for existing in list(logger.handlers):
        logger.removeHandler(existing)

    if json is None:
        env_fmt = os.environ.get("OPHELIAN_LOG_FORMAT", "").lower()
        json = env_fmt == "json"

    if stream is None:
        stream = sys.stdout if json else sys.stderr
    handler = logging.StreamHandler(stream)
    if json:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(_resolve_level(level))
    logger.propagate = False
    return logger


# Initialise on import so library code can `import logging; logging.getLogger("ophelian")`
# without first calling configure_logging — exact behaviour matches the v0.x default.
logger = configure_logging()


# ----------------------------------------------------------------------
# Optional OpenTelemetry event hook
# ----------------------------------------------------------------------


def emit_event(name: str, **attributes: Any) -> None:
    """Emit a span event named *name* on the active OTel span, if any.

    Silent no-op when ``opentelemetry`` is not installed or no tracer
    provider has been configured. This keeps the lib usable without
    OTel while giving operators a one-line hook for distributed tracing.
    """
    try:
        from opentelemetry import trace
    except ImportError:
        return
    try:
        span = trace.get_current_span()
        if span is None:
            return
        if hasattr(span, "is_recording") and not span.is_recording():
            return
        span.add_event(name, attributes={k: _otel_safe(v) for k, v in attributes.items()})
    except Exception:  # pragma: no cover - defensive
        return


def _otel_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return json.dumps(value, default=str)
    if isinstance(value, (list, tuple)):
        return [_otel_safe(v) for v in value]
    return repr(value)


from ophelian.observability.otel import (  # noqa: E402  (circular-safe; otel.py does not import from us)
    auto_configure_from_env as auto_configure_otel,
)
from ophelian.observability.otel import (  # noqa: E402
    extract_token_usage,
    is_otel_available,
    pipeline_span,
    record_inference_tokens,
    record_pipeline_outcome,
    record_serve_idle,
    record_serve_outcome,
    record_step_outcome,
    register_queue_depth_observer,
    serve_inflight_dec,
    serve_inflight_inc,
    serve_request_span,
    step_span,
)
from ophelian.observability.events import (  # noqa: E402
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
    emit as emit_lifecycle_event,
    on_event,
)

__all__ = [
    "InferenceFailed",
    "JSONFormatter",
    "LifecycleEvent",
    "ModelLoaded",
    "ModelSwapped",
    "ModelUnloaded",
    "PipelineCompleted",
    "PipelineStarted",
    "SpotInterruptionReceived",
    "StepCompleted",
    "StepFailed",
    "StepStarted",
    "auto_configure_otel",
    "bind_run",
    "configure_logging",
    "emit_event",
    "emit_lifecycle_event",
    "extract_token_usage",
    "get_run_id",
    "is_otel_available",
    "logger",
    "on_event",
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
    "set_run_id",
    "step_span",
]
