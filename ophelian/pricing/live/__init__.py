"""Live pricing fetchers + on-disk cache coordinator.

:func:`fetch_live` is the single entry point. It dispatches to the
per-cloud fetchers (:mod:`.aws`, :mod:`.azure`, :mod:`.gcp`), persists
the result to a JSON cache and serves cache hits straight back without
hitting the network.

The function never raises on transport errors — it returns whatever it
could fetch and logs at debug level so the static fallback can still
drive routing.
"""

from __future__ import annotations

import logging
import time

from ophelian.pricing.cache import _load_cache, _save_cache
from ophelian.pricing.live.aws import _fetch_aws_spot_quotes
from ophelian.pricing.live.azure import _fetch_azure_retail_quotes
from ophelian.pricing.live.gcp import _fetch_gcp_billing_quotes
from ophelian.pricing.quotes import PriceQuote

logger = logging.getLogger("ophelian.pricing.live")


_ALL_PROVIDERS = ("aws", "azure", "gcp")


def fetch_live_with_meta(
    gpu_family: str,
    *,
    providers: list[str] | None = None,
    regions: list[str] | None = None,
    cache_ttl_seconds: int = 24 * 3600,
) -> tuple[list[PriceQuote], dict[str, str]]:
    """Like :func:`fetch_live` but also returns per-provider provenance.

    Returns a ``(quotes, meta)`` tuple where ``meta`` maps each
    consulted provider to one of:

    * ``"live"`` — fetched fresh from the cloud's pricing API and
      returned at least one quote.
    * ``"cached@<N>h"`` — the on-disk cache was within
      ``cache_ttl_seconds`` and contained at least one usable entry
      for this provider after the GPU-family / regions filters were
      applied. ``<N>`` is the cache age rounded down to whole hours.
    * ``"unavailable"`` — the provider was consulted but returned no
      usable quotes (network failed, no credentials, API rate-limited,
      no SKUs match the GPU family / region filter, etc.). The router
      will fall back to the static table for this provider.

    ``meta`` only contains keys for providers actually consulted —
    providers excluded by the caller's ``providers=[...]`` filter are
    omitted entirely (the consumer can mark those as ``"disabled"``).

    The strategy and source labels are documented in :func:`fetch_live`;
    this function adds no I/O beyond what :func:`fetch_live` already
    does.
    """
    consulted: tuple[str, ...] = (
        tuple(p for p in _ALL_PROVIDERS if p in providers)
        if providers is not None
        else _ALL_PROVIDERS
    )
    cache = _load_cache()
    fetched_at = float(cache.get("_fetched_at", 0.0))
    family_upper = gpu_family.upper()

    cache_fresh = fetched_at > 0.0 and time.time() - fetched_at <= cache_ttl_seconds
    if cache_fresh:
        out: list[PriceQuote] = []
        per_provider: dict[str, list[PriceQuote]] = {p: [] for p in consulted}
        for entry in cache.get("quotes", []):
            try:
                quote = PriceQuote(**entry)
            except Exception:
                continue
            if quote.gpu_family.upper() != family_upper:
                continue
            if quote.provider not in consulted:
                continue
            if regions and quote.region not in regions:
                continue
            out.append(quote)
            per_provider[quote.provider].append(quote)

        age_hours = max(0, int((time.time() - fetched_at) // 3600))
        meta = {p: f"cached@{age_hours}h" if per_provider[p] else "unavailable" for p in consulted}
        return out, meta

    fetched: dict[str, list[PriceQuote]] = {}
    if "aws" in consulted:
        fetched["aws"] = _fetch_aws_spot_quotes(gpu_family, regions)
    if "azure" in consulted:
        fetched["azure"] = _fetch_azure_retail_quotes(gpu_family, regions)
    if "gcp" in consulted:
        fetched["gcp"] = _fetch_gcp_billing_quotes(gpu_family, regions)

    out = []
    for provider_quotes in fetched.values():
        out.extend(provider_quotes)

    refreshed_providers = {p for p, qs in fetched.items() if qs}
    if refreshed_providers:
        preserved_entries = [
            q
            for q in cache.get("quotes", [])
            if isinstance(q, dict) and q.get("provider") not in refreshed_providers
        ]
        for provider_quotes in fetched.values():
            preserved_entries.extend(
                {
                    "provider": q.provider,
                    "region": q.region,
                    "instance": q.instance,
                    "gpu_family": q.gpu_family,
                    "gpu_count": q.gpu_count,
                    "hourly_usd": q.hourly_usd,
                    "spot": q.spot,
                    "source": q.source,
                    "notes": q.notes,
                }
                for q in provider_quotes
            )
        _save_cache({"_fetched_at": time.time(), "quotes": preserved_entries})

    meta = {p: ("live" if fetched.get(p) else "unavailable") for p in consulted}
    return out, meta


def fetch_live(
    gpu_family: str,
    *,
    providers: list[str] | None = None,
    regions: list[str] | None = None,
    cache_ttl_seconds: int = 24 * 3600,
) -> list[PriceQuote]:
    """Return live quotes for *gpu_family* across AWS, Azure, and GCP.

    Thin wrapper around :func:`fetch_live_with_meta` that drops the
    provenance map. Kept for backward compatibility with callers that
    only need the quotes (notebooks, ``lookup_cheapest`` legacy path).

    Strategy (honest, what's actually possible today, per provider):

    * **AWS** — calls ``ec2.describe_spot_price_history`` per relevant
      region for the instance types we know map to ``gpu_family``.
      Free, public, no extra credentials beyond standard boto3.
      Source label: ``aws-spot-history``.
    * **Azure** — hits the public **Retail Prices API**
      (``https://prices.azure.com/api/retail/prices``). No auth, no
      API key required. Returns true on-demand and Spot prices for
      every Azure VM SKU; we filter to the SKUs we recognise from
      the static table and skip Windows variants.
      Source label: ``azure-retail-api``.
    * **GCP** — uses the **Cloud Billing Catalog API**
      (``https://cloudbilling.googleapis.com/v1/services/6F81-5844-456A``)
      when the ``GOOGLE_API_KEY`` env var is set. The catalog only
      exposes per-SKU prices (compute / RAM / GPU billed separately)
      so we publish a *live GPU rate x gpu_count + static compute
      base* aggregate. Honest source label: ``gcp-billing-catalog``.
      Without ``GOOGLE_API_KEY``, GCP silently falls back to the
      static table and a debug log is emitted.

    Results from all three providers are persisted to a single JSON
    cache under ``OPHELIAN_PRICING_CACHE_DIR`` (default
    ``~/.cache/ophelian/pricing.json``) for ``cache_ttl_seconds``
    (default 24 h) so subsequent ``Auto(...)`` calls are fully
    offline. The function never raises on transport errors — it
    returns whatever it could fetch and logs at debug level so the
    static fallback can still drive routing.
    """
    quotes, _ = fetch_live_with_meta(
        gpu_family,
        providers=providers,
        regions=regions,
        cache_ttl_seconds=cache_ttl_seconds,
    )
    return quotes


__all__ = ["fetch_live", "fetch_live_with_meta"]
