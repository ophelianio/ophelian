"""Tiny stdlib HTTP helpers shared by the live pricing fetchers.

Lives outside :mod:`ophelian.pricing.live` so it can be imported by
all three cloud-specific submodules without creating a circular
dependency through the live package's own ``__init__``.

The module deliberately uses :mod:`urllib` rather than :pypi:`requests`:
live pricing is a *best-effort enrichment* path, never the critical
path, and we don't want to add a transitive dependency for a feature
the user can opt out of by leaving ``allow_live=False``.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger("ophelian.pricing.http")


def _http_get_json(url: str, *, timeout: float = 8.0) -> Any:
    """GET *url* and return parsed JSON. Returns ``None`` on any error."""
    try:
        from urllib.error import URLError
        from urllib.request import Request, urlopen
    except ImportError:  # pragma: no cover - urllib is stdlib
        return None
    # Reject non-HTTP(S) schemes defensively: while callers only pass URLs
    # from internal pricing-source constants, restricting the scheme here
    # closes the door on accidental ``file://`` / custom-scheme regressions.
    if not (url.startswith("https://") or url.startswith("http://")):
        logger.debug("Live pricing GET %s rejected: only http(s) is allowed", url)
        return None
    try:
        request = Request(url, headers={"Accept": "application/json"})
        with urlopen(request, timeout=timeout) as response:  # nosec B310
            raw = response.read()
    except (URLError, TimeoutError, ValueError, OSError) as exc:
        logger.debug("Live pricing GET %s failed: %s", url, exc)
        return None
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        logger.debug("Live pricing JSON decode failed for %s: %s", url, exc)
        return None


def _url_quote(value: str) -> str:
    from urllib.parse import quote

    return quote(value, safe="")


__all__ = ["_http_get_json", "_url_quote"]
