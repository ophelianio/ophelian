"""Spot-interruption detection and resumable-run plumbing.

EC2 spot instances can be reclaimed with two minutes of notice. The
provider treats that as a recoverable failure: the running step persists
its progress to the artifact store as a *checkpoint* and the run is tagged
``resumable``. The next time the user calls ``pipe.run(env=env)`` against
the same env, the provider sees the checkpoint, skips the steps that
already completed, and picks up where the interrupted step left off — on a
fresh instance.

The actual *probe* for spot interruption depends on where this runs:

* Inside the worker container, AWS exposes the imminent termination
  notice at ``http://169.254.169.254/latest/meta-data/spot/instance-action``.
  :func:`is_spot_interruption_imminent` polls it.

* From the host (the user's laptop running ``pipe.run(...)``), CloudWatch
  publishes ``EC2 Spot Instance Interruption Warning`` events. The host
  driver listens for them via ``describe_instance_status`` polling, which
  is good enough for a v0.5 MVP without an EventBridge rule.

Tests can inject a :class:`SpotInterruptionMonitor` with a deterministic
``trigger`` to exercise the resume path without any real AWS or HTTP.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

if TYPE_CHECKING:  # pragma: no cover
    from ophelian.stores.base import ArtifactStore

logger = logging.getLogger("ophelian.runtime.spot")

SPOT_METADATA_URL = "http://169.254.169.254/latest/meta-data/spot/instance-action"
DEFAULT_PROBE_TIMEOUT = 1.5


class SpotInterruption(RuntimeError):
    """Raised when a spot instance is being reclaimed mid-step."""

    def __init__(self, message: str, *, deadline: float | None = None) -> None:
        super().__init__(message)
        self.deadline = deadline


@dataclass
class Checkpoint:
    """In-memory representation of a resumable run state.

    The driver writes one of these to ``s3://<bucket>/<prefix>/<run_id>/checkpoint.json``
    whenever a step exits with a :class:`SpotInterruption`. On the next
    invocation, the provider loads the file and skips the steps named in
    ``completed_steps``, restarting from the one in ``in_flight_step``.
    """

    run_id: str
    pipeline: str
    completed_steps: list[str] = field(default_factory=list)
    in_flight_step: str | None = None
    artifacts: dict[str, dict[str, str]] = field(default_factory=dict)
    info: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(
            {
                "run_id": self.run_id,
                "pipeline": self.pipeline,
                "completed_steps": self.completed_steps,
                "in_flight_step": self.in_flight_step,
                "artifacts": self.artifacts,
                "info": self.info,
            },
            indent=2,
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, payload: str | bytes) -> Checkpoint:
        data = json.loads(payload)
        return cls(
            run_id=data["run_id"],
            pipeline=data["pipeline"],
            completed_steps=list(data.get("completed_steps", [])),
            in_flight_step=data.get("in_flight_step"),
            artifacts={k: dict(v) for k, v in data.get("artifacts", {}).items()},
            info=dict(data.get("info", {})),
        )


def checkpoint_key(run_id: str) -> str:
    """Conventional key used to store the resume checkpoint for *run_id*."""
    return f"checkpoints/{run_id}/checkpoint.json"


def save_checkpoint(store: ArtifactStore, checkpoint: Checkpoint) -> str:
    """Persist *checkpoint* into *store*; returns the URI."""
    payload = checkpoint.to_json().encode("utf-8")
    key = checkpoint_key(checkpoint.run_id)
    if hasattr(store, "put_bytes"):
        return store.put_bytes(key, payload)
    # Fallback for stores that only accept file paths.
    import tempfile

    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
        tmp.write(payload)
        tmp.flush()
        path = Path(tmp.name)
    try:
        return store.put(key, path)
    finally:
        path.unlink(missing_ok=True)


def load_checkpoint(store: ArtifactStore, run_id: str) -> Checkpoint | None:
    """Return the checkpoint for *run_id* if one is present, else ``None``."""
    key = checkpoint_key(run_id)
    if not store.exists(key):
        return None
    if hasattr(store, "get_bytes"):
        try:
            payload = store.get_bytes(key)
        except FileNotFoundError:
            return None
        return Checkpoint.from_json(payload)
    path = store.get(key)
    return Checkpoint.from_json(Path(path).read_bytes())


@dataclass
class SpotInterruptionMonitor:
    """Detect imminent spot interruption.

    Tests construct this with ``trigger=lambda: True`` to force the
    provider down the resume code path without any real AWS calls.
    """

    probe: Callable[[], bool] | None = None
    poll_seconds: float = 5.0
    _triggered: bool = False

    def check(self) -> bool:
        if self._triggered:
            return True
        probe = self.probe or default_metadata_probe
        try:
            if probe():
                self._triggered = True
                # Lifecycle: spot_interruption_received — emitted on
                # the rising 0→1 edge so subscribers fire exactly
                # once per interruption, even if ``check()`` is
                # polled in a loop.
                try:
                    from ophelian.observability.events import (
                        SpotInterruptionReceived,
                    )
                    from ophelian.observability.events import (
                        emit as emit_lifecycle,
                    )

                    emit_lifecycle(SpotInterruptionReceived(source="spot.monitor"))
                except Exception:  # pragma: no cover - defensive
                    pass
                return True
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("spot probe error: %s", exc)
        return False

    def force_trigger(self) -> None:
        """Mark the monitor as triggered (used by tests)."""
        self._triggered = True

    def watch(self, deadline: float | None = None) -> bool:
        """Block until either the probe reports interruption or *deadline* passes."""
        start = time.monotonic()
        while True:
            if self.check():
                return True
            if deadline is not None and time.monotonic() - start >= deadline:
                return False
            time.sleep(self.poll_seconds)


def default_metadata_probe() -> bool:
    """Probe the EC2 spot instance-action metadata endpoint.

    Returns True when a termination time is published.
    """
    try:
        import urllib.request

        # ``SPOT_METADATA_URL`` is a module-level constant pointing at the
        # EC2 instance metadata service (http://169.254.169.254/...). The
        # scheme is fixed and not derived from user input.
        with urllib.request.urlopen(  # nosec B310
            SPOT_METADATA_URL, timeout=DEFAULT_PROBE_TIMEOUT
        ) as resp:
            return bool(resp.status == 200)
    except Exception:
        return False


def is_resumable_uri(uri: str) -> bool:
    """Return True if *uri* points at a checkpoint file we can read back."""
    parsed = urlparse(uri)
    return parsed.scheme in {"s3", "file", ""} and uri.endswith("checkpoint.json")


__all__ = [
    "Checkpoint",
    "SpotInterruption",
    "SpotInterruptionMonitor",
    "checkpoint_key",
    "default_metadata_probe",
    "is_resumable_uri",
    "load_checkpoint",
    "save_checkpoint",
]
