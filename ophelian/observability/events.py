"""Typed in-process lifecycle event bus for Ophelian (Task #29).

External systems (auto-rollback policies, alerting integrations,
audit trails) need a programmatic hook into Ophelian's lifecycle
without parsing structured logs. This module exposes a typed event
bus emitting domain-level events from the FastAPI runtime, the step
runner, the spot probe, and the pipeline orchestrator. Each event
also mirrors as an OpenTelemetry span event so backends already
wired through ``ophelian.observability.otel`` see lifecycle without
separate plumbing.

Subscriber failures never break the emitting code path: handlers are
dispatched in an isolated try/except and the failure is logged at
WARN with the handler identity.

Public surface (re-exported from :mod:`ophelian.observability`)::

    from ophelian.observability import (
        on_event, LifecycleEvent, ModelLoaded, ModelUnloaded,
        ModelSwapped, InferenceFailed, PipelineStarted, StepStarted,
        StepFailed, StepCompleted, PipelineCompleted,
        SpotInterruptionReceived,
    )

    def react(event: LifecycleEvent) -> None:
        ...

    unsubscribe = on_event(react)

The opaque ``context`` dict carried on every event is the same one
the cost ledger task threads through ``Pipeline.context`` so
multi-tenant consumers can route by e.g. ``tenant_id`` without any
extra plumbing.
"""

from __future__ import annotations

import contextlib
import json
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field, fields
from typing import Any, ClassVar

logger = logging.getLogger("ophelian.observability.events")


# ----------------------------------------------------------------------
# Event hierarchy
# ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class LifecycleEvent:
    """Base class for every lifecycle event.

    Subclasses set ``EVENT_NAME`` (the canonical identifier used by
    both the bus and the OTel span-event mirror) and add their own
    typed payload fields. ``run_id`` is set when the event fires
    inside a bound run; ``context`` is the opaque dict passed from
    ``Pipeline.context``.
    """

    EVENT_NAME: ClassVar[str] = "ophelian.lifecycle"
    source: str
    timestamp: float = field(default_factory=time.time)
    run_id: str | None = None
    context: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelLoaded(LifecycleEvent):
    EVENT_NAME: ClassVar[str] = "ophelian.model.loaded"
    framework: str
    model_path: str


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelUnloaded(LifecycleEvent):
    EVENT_NAME: ClassVar[str] = "ophelian.model.unloaded"
    framework: str
    model_path: str


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelSwapped(LifecycleEvent):
    """Emitted when a deployed endpoint hot-swaps to a new model artifact.

    The runtime has no built-in hot-swap path today; this event is
    part of the public contract so that custom deploy controllers
    (canary rollouts, blue/green) can publish it themselves and
    downstream subscribers can react uniformly.
    """

    EVENT_NAME: ClassVar[str] = "ophelian.model.swapped"
    framework: str
    previous_model_path: str
    new_model_path: str


@dataclass(frozen=True, slots=True, kw_only=True)
class InferenceFailed(LifecycleEvent):
    EVENT_NAME: ClassVar[str] = "ophelian.inference.failed"
    framework: str
    route: str
    method: str
    error: str


@dataclass(frozen=True, slots=True, kw_only=True)
class PipelineStarted(LifecycleEvent):
    EVENT_NAME: ClassVar[str] = "ophelian.pipeline.started"
    pipeline: str
    provider: str
    env_class: str


@dataclass(frozen=True, slots=True, kw_only=True)
class PipelineCompleted(LifecycleEvent):
    EVENT_NAME: ClassVar[str] = "ophelian.pipeline.completed"
    pipeline: str
    provider: str
    status: str  # "success" | "failed"


@dataclass(frozen=True, slots=True, kw_only=True)
class StepStarted(LifecycleEvent):
    EVENT_NAME: ClassVar[str] = "ophelian.step.started"
    step_name: str
    step_kind: str
    provider: str


@dataclass(frozen=True, slots=True, kw_only=True)
class StepCompleted(LifecycleEvent):
    EVENT_NAME: ClassVar[str] = "ophelian.step.completed"
    step_name: str
    step_kind: str
    provider: str
    duration_seconds: float
    status: str  # "success" | "skipped"


@dataclass(frozen=True, slots=True, kw_only=True)
class StepFailed(LifecycleEvent):
    EVENT_NAME: ClassVar[str] = "ophelian.step.failed"
    step_name: str
    step_kind: str
    provider: str
    duration_seconds: float | None
    error: str


@dataclass(frozen=True, slots=True, kw_only=True)
class SpotInterruptionReceived(LifecycleEvent):
    EVENT_NAME: ClassVar[str] = "ophelian.spot.interruption"
    deadline: float | None = None


# ----------------------------------------------------------------------
# Bus
# ----------------------------------------------------------------------


EventHandler = Callable[[LifecycleEvent], None]
Unsubscribe = Callable[[], None]

_lock = threading.RLock()
_subscribers: list[EventHandler] = []


def on_event(handler: EventHandler) -> Unsubscribe:
    """Register *handler* to be called for every emitted event.

    Returns a callable that, when invoked, removes *handler* from
    the dispatch list. Handlers receive every event type and are
    expected to filter via ``isinstance(event, ...)`` or
    ``event.EVENT_NAME``.
    """
    with _lock:
        _subscribers.append(handler)

    def _unsubscribe() -> None:
        with _lock, contextlib.suppress(ValueError):
            _subscribers.remove(handler)

    return _unsubscribe


def emit(event: LifecycleEvent) -> None:
    """Dispatch *event* to every registered handler and mirror it
    onto the active OTel span.

    Per-handler errors are caught and logged at WARN; they never
    propagate back to the emitting code path. Mirroring is a silent
    no-op when OTel is not installed or when no span is active.
    """
    _mirror_to_otel_span(event)
    with _lock:
        snapshot = list(_subscribers)
    for handler in snapshot:
        try:
            handler(event)
        except Exception:
            logger.warning(
                "Lifecycle handler %r raised on %s",
                handler,
                event.EVENT_NAME,
                exc_info=True,
            )


def _mirror_to_otel_span(event: LifecycleEvent) -> None:
    try:
        from opentelemetry.trace import get_current_span
    except ImportError:
        return
    try:
        span = get_current_span()
        if span is None:
            return
        if hasattr(span, "is_recording") and not span.is_recording():
            return
        span.add_event(event.EVENT_NAME, attributes=_event_to_otel_attrs(event))
    except Exception:  # pragma: no cover - defensive
        return


def _event_to_otel_attrs(event: LifecycleEvent) -> dict[str, Any]:
    """Flatten a frozen event into an OTel-attribute-safe mapping.

    OTel only accepts primitive scalar / sequence values, so the
    opaque ``context`` dict is JSON-serialised and any other
    non-primitive payload is repr'd. ``None`` is dropped because
    OTel SDKs reject it.
    """
    attrs: dict[str, Any] = {"event_type": event.EVENT_NAME}
    for f in fields(event):
        value = getattr(event, f.name)
        if value is None:
            continue
        if f.name == "context":
            attrs["context"] = json.dumps(value, default=str)
            continue
        if isinstance(value, (str, int, float, bool)):
            attrs[f.name] = value
        else:
            attrs[f.name] = repr(value)
    return attrs


def _clear_subscribers_for_tests() -> None:
    """Drop every registered subscriber. Test-only escape hatch."""
    with _lock:
        _subscribers.clear()


__all__ = [
    "EventHandler",
    "InferenceFailed",
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
    "Unsubscribe",
    "emit",
    "on_event",
]
