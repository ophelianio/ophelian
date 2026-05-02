"""On-disk JSON cache for live pricing fetches.

A single file (default ``~/.cache/ophelian/pricing.json``, overridable
via the ``OPHELIAN_PRICING_CACHE_DIR`` env var) holds the last
successful fetch for every provider. Used by
:func:`ophelian.pricing.live.fetch_live` to keep subsequent
``Auto(...)`` calls fully offline within the configured TTL.

All public helpers are tolerant of corrupt files / missing parents —
nothing here is allowed to raise; pricing is a *helper* path, never the
critical path.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger("ophelian.pricing.cache")


def _cache_path() -> Path:
    base = os.environ.get("OPHELIAN_PRICING_CACHE_DIR")
    if base:
        return Path(base)
    return Path.home() / ".cache" / "ophelian" / "pricing.json"


def _load_cache(path: Path | None = None) -> dict[str, Any]:
    path = path or _cache_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except Exception:  # pragma: no cover - corrupt cache
        logger.debug("Pricing cache at %s was unreadable; ignoring", path)
        return {}
    return data if isinstance(data, dict) else {}


def _save_cache(payload: dict[str, Any], path: Path | None = None) -> None:
    path = path or _cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, default=str, indent=2))


__all__ = ["_cache_path", "_load_cache", "_save_cache"]
