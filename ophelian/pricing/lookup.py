"""The :func:`lookup_cheapest` primitive that drives the auto-router.

Combines static-table quotes (always) with live-fetch quotes (opt-in)
and returns the single cheapest match. This is the function
:class:`ophelian.envs.auto.Auto` calls when picking a GPU.
"""

from __future__ import annotations

import logging

from ophelian.pricing.live import fetch_live, fetch_live_with_meta
from ophelian.pricing.quotes import PriceQuote
from ophelian.pricing.static import static_quotes

logger = logging.getLogger("ophelian.pricing.lookup")

_ALL_PROVIDERS = ("aws", "gcp", "azure")


def lookup_cheapest_with_meta(
    gpu_family: str,
    *,
    providers: list[str] | None = None,
    regions: list[str] | None = None,
    spot: bool | None = None,
    allow_live: bool = False,
    cache_ttl_seconds: int = 24 * 3600,
) -> tuple[PriceQuote | None, dict[str, str]]:
    """Like :func:`lookup_cheapest` but also returns per-provider provenance.

    Returns a ``(quote, data_quality)`` tuple where ``data_quality``
    maps every provider in :data:`_ALL_PROVIDERS` to one of:

    * ``"disabled"`` — provider excluded by the caller's
      ``providers=[...]`` filter; never consulted.
    * ``"static"`` — ``allow_live=False``, OR live fetch was attempted
      but raised (network/auth) and we fell back to the static table.
    * ``"live"`` — live API returned at least one quote this call.
    * ``"cached@<N>h"`` — served from the on-disk pricing cache, age
      in whole hours.
    * ``"unavailable"`` — live path was attempted but returned no
      usable quotes for this provider.

    Vocabulary mirrors :func:`ophelian.pricing.live.fetch_live_with_meta`
    plus the two states only this layer can know about (``"disabled"``,
    ``"static"``).

    Parameters are documented in :func:`lookup_cheapest`.
    """
    consulted = (
        [p for p in _ALL_PROVIDERS if p in providers]
        if providers is not None
        else list(_ALL_PROVIDERS)
    )
    data_quality: dict[str, str] = {
        p: ("static" if p in consulted else "disabled") for p in _ALL_PROVIDERS
    }

    quotes = static_quotes(gpu_family, providers=providers, regions=regions, spot=spot)

    if allow_live:
        try:
            live_quotes, live_meta = fetch_live_with_meta(
                gpu_family,
                providers=providers,
                regions=regions,
                cache_ttl_seconds=cache_ttl_seconds,
            )
            quotes.extend(live_quotes)
            for provider, value in live_meta.items():
                data_quality[provider] = value
        except Exception as exc:  # pragma: no cover - network-dependent
            logger.debug("Live pricing fetch failed (%s); using static only.", exc)
            # data_quality already says "static" for consulted providers.

    if not quotes:
        return None, data_quality
    return min(quotes, key=lambda q: q.hourly_usd), data_quality


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

    Thin wrapper around :func:`lookup_cheapest_with_meta` that drops
    the provenance map. Kept for backward compatibility.

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
    quote, _ = lookup_cheapest_with_meta(
        gpu_family,
        providers=providers,
        regions=regions,
        spot=spot,
        allow_live=allow_live,
        cache_ttl_seconds=cache_ttl_seconds,
    )
    return quote


# Re-export for back-compat: the resilience test (and any user code in
# the wild) that does `monkeypatch.setattr(lookup_mod, "fetch_live", ...)`
# still finds the symbol — we just don't call it from this module
# anymore. The new monkeypatch target is ``fetch_live_with_meta``.
__all__ = ["lookup_cheapest", "lookup_cheapest_with_meta", "fetch_live"]
