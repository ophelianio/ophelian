"""Append-only cost ledger (Task #28).

Every terminal pipeline run writes one structured row to a local
JSONL file (default ``~/.ophelian/ledger.jsonl``, override via
``OPHELIAN_LEDGER_PATH``). The file is the foundation for the
``ophelian costs`` showback CLI and for any downstream consumer
that wants to attribute spend by team / project / tenant.

Design notes
------------

* Append-only JSONL, one row per run. Easy to ``tail`` / ``jq`` /
  pipe into Loki without parsing a custom format.
* Atomic appends under a ``fcntl.flock`` sidecar file so concurrent
  writers (parallel pipelines on the same host) cannot interleave a
  partial line.
* ``schema_version`` is part of every row so downstream consumers
  can rely on the contract evolving forward-compatibly.
* Failures are best-effort: ledger I/O never breaks a user pipeline.
* The ``context`` dict is the same opaque labels bag that
  :class:`ophelian.core.nodes.Pipeline.context` propagates through
  every lifecycle event, so ledger rows and OTel spans line up on
  the same keys without extra plumbing on the caller's side.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("ophelian.observability.ledger")

SCHEMA_VERSION = 1
"""Bumped on any breaking change to the row shape."""

ENV_VAR_PATH = "OPHELIAN_LEDGER_PATH"
"""Environment variable that overrides the default ledger location."""

ENV_VAR_DISABLED = "OPHELIAN_LEDGER_DISABLED"
"""Set to ``1`` to skip ledger writes entirely (useful in CI / tests)."""

_DEFAULT_PATH = Path.home() / ".ophelian" / "ledger.jsonl"


def ledger_path() -> Path:
    """Return the active ledger path, honouring ``OPHELIAN_LEDGER_PATH``."""
    override = os.environ.get(ENV_VAR_PATH)
    return Path(override).expanduser() if override else _DEFAULT_PATH


@dataclass(frozen=True)
class LedgerRow:
    """One terminal-state row of the cost ledger.

    Field stability is part of the public contract — see
    ``docs/ledger.md``. New fields go at the end and must default
    to a sensible value so older consumers stay green.
    """

    schema_version: int
    run_id: str
    pipeline: str
    timestamp: float  # epoch seconds, UTC
    env_class: str
    provider: str
    region: str
    gpu_type: str | None
    instance: str | None
    hours: float
    hourly_usd: float | None
    estimated_usd: float | None
    actual_usd: float | None
    status: str  # "success" | "failed"
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@contextlib.contextmanager
def _file_lock(target: Path) -> Iterator[None]:
    """Acquire an exclusive lock on a sidecar ``.lock`` file.

    Uses ``fcntl.flock`` on POSIX. On platforms without ``fcntl``
    (Windows) we fall back to a best-effort spinning-create on an
    O_EXCL lock file with a short deadline. The sidecar file lets us
    keep ``target`` purely append-only and avoids fighting other
    tools for an exclusive handle on the ledger itself.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    lock_path = target.with_suffix(target.suffix + ".lock")
    try:
        import fcntl
    except ImportError:  # pragma: no cover - Windows
        deadline = time.monotonic() + 5.0
        while True:
            try:
                fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
                break
            except FileExistsError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.01)
        try:
            yield
        finally:
            os.close(fd)
            with contextlib.suppress(OSError):
                lock_path.unlink()
        return
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def append_row(row: LedgerRow, *, path: Path | None = None) -> Path:
    """Append *row* to the ledger atomically. Returns the file path written."""
    target = path or ledger_path()
    payload = json.dumps(row.to_dict(), default=str, sort_keys=True)
    with _file_lock(target):
        with open(target, "a", encoding="utf-8") as fh:
            fh.write(payload + "\n")
            fh.flush()
            os.fsync(fh.fileno())
    return target


def read_rows(path: Path | None = None) -> list[dict[str, Any]]:
    """Read every row from the ledger. Malformed lines are skipped + logged."""
    target = path or ledger_path()
    if not target.exists():
        return []
    rows: list[dict[str, Any]] = []
    with open(target, encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError:
                logger.warning("Skipping malformed ledger row: %s", stripped[:120])
    return rows


def emit_run_row(
    *,
    pipeline_name: str,
    run_id: str | None,
    env_class: str,
    provider: str,
    region: str | None,
    gpu_type: str | None,
    instance: str | None,
    hours: float,
    hourly_usd: float | None,
    status: str,
    context: dict[str, Any] | None,
    path: Path | None = None,
) -> Path | None:
    """Build and append one ledger row. Best-effort: errors are logged.

    ``actual_usd`` is computed as ``hourly_usd * hours`` so failed
    runs report cost incurred up to the failure point (the caller
    is expected to pass the duration accumulated across completed
    + in-flight steps). ``estimated_usd`` mirrors ``actual_usd``
    today; the field exists so the planned "show estimate before
    launch" task can populate a separate up-front number without a
    schema bump.
    """
    if os.environ.get(ENV_VAR_DISABLED) in {"1", "true", "True"}:
        return None
    if hourly_usd is not None and hours > 0:
        actual: float | None = round(float(hourly_usd) * float(hours), 6)
    else:
        actual = None
    row = LedgerRow(
        schema_version=SCHEMA_VERSION,
        run_id=run_id or pipeline_name,
        pipeline=pipeline_name,
        timestamp=time.time(),
        env_class=env_class,
        provider=provider,
        region=region or "local",
        gpu_type=gpu_type,
        instance=instance,
        hours=round(float(hours), 6),
        hourly_usd=float(hourly_usd) if hourly_usd is not None else None,
        estimated_usd=actual,
        actual_usd=actual,
        status=status,
        context=dict(context) if context else {},
    )
    try:
        return append_row(row, path=path)
    except Exception:  # pragma: no cover - best-effort
        logger.warning("Failed to write ledger row", exc_info=True)
        return None


__all__ = [
    "ENV_VAR_DISABLED",
    "ENV_VAR_PATH",
    "LedgerRow",
    "SCHEMA_VERSION",
    "append_row",
    "emit_run_row",
    "ledger_path",
    "read_rows",
]
