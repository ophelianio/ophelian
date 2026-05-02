"""The :func:`lookup_cheapest` primitive that drives the auto-router.

Combines static-table quotes (always) with live-fetch quotes (opt-in)
and returns the single cheapest match. This is the function
:class:`ophelian.envs.auto.Auto` calls when picking a GPU.
"""

from __future__ import annotations

import logging

from ophelian.pricing.live import fetch_live
from ophelian.pricing.quotes import PriceQuote
from ophelian.pricing.static import static_quotes

logger = logging.getLogger("ophelian.pricing.lookup")


def lookup_cheapest(
    gpu_family: str,
    *,
    providers: list[str] | None = None,
    regions: list[str] | None = None,
    spot: bool | None = None,
    allow_live: bool = False,
    cache_ttl_seconds: int = 24 * 3600,
) -> PriceQuote | None:
    """Return the cheapest :class:`PriceQuote` matching the filters.

    Parameters
    ----------
    gpu_family:
        Friendly GPU family name (``"A100"``, ``"H100"``, ``"T4"``, ...)
    providers:
        Subset of ``["aws", "gcp", "azure"]``. ``None`` = all.
    regions:
        Subset of provider-specific region names. ``None`` = all known.
    spot:
        ``True`` = only spot/preemptible quotes; ``False`` = only on-demand;
        ``None`` (default) = consider both.
    allow_live:
        When ``True`` and :pypi:`requests` is installed, also consult
        the cloud providers' pricing endpoints (with on-disk cache).
        Defaults to ``False`` so the lookup is fully offline-safe.
    cache_ttl_seconds:
        TTL for the on-disk live-pricing cache.
    """
    quotes = static_quotes(gpu_family, providers=providers, regions=regions, spot=spot)
    if allow_live:
        try:
            quotes.extend(
                fetch_live(
                    gpu_family,
                    providers=providers,
                    regions=regions,
                    cache_ttl_seconds=cache_ttl_seconds,
                )
            )
        except Exception as exc:  # pragma: no cover - network-dependent
            logger.debug("Live pricing fetch failed (%s); using static only.", exc)
    if not quotes:
        return None
    return min(quotes, key=lambda q: q.hourly_usd)


__all__ = ["lookup_cheapest"]
