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


def fetch_live(
    gpu_family: str,
    *,
    providers: list[str] | None = None,
    regions: list[str] | None = None,
    cache_ttl_seconds: int = 24 * 3600,
) -> list[PriceQuote]:
    """Return live quotes for *gpu_family* across AWS, Azure, and GCP.

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
      base* aggregate. The static-compute base is derived once at
      import time from the curated table by subtracting the
      published static GPU rate. Honest source label:
      ``gcp-billing-catalog (live GPU + static compute)``.
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
    cache = _load_cache()
    fetched_at = float(cache.get("_fetched_at", 0.0))
    family_upper = gpu_family.upper()
    out: list[PriceQuote] = []

    cache_fresh = time.time() - fetched_at <= cache_ttl_seconds
    if cache_fresh:
        for entry in cache.get("quotes", []):
            try:
                quote = PriceQuote(**entry)
            except Exception:
                continue
            if quote.gpu_family.upper() != family_upper:
                continue
            if providers and quote.provider not in providers:
                continue
            if regions and quote.region not in regions:
                continue
            out.append(quote)
        return out

    fetched: dict[str, list[PriceQuote]] = {}
    if providers is None or "aws" in providers:
        fetched["aws"] = _fetch_aws_spot_quotes(gpu_family, regions)
    if providers is None or "azure" in providers:
        fetched["azure"] = _fetch_azure_retail_quotes(gpu_family, regions)
    if providers is None or "gcp" in providers:
        fetched["gcp"] = _fetch_gcp_billing_quotes(gpu_family, regions)

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

    return out


__all__ = ["fetch_live"]
