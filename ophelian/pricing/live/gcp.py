"""Live GCP pricing via the Cloud Billing Catalog API.

Endpoint:
``https://cloudbilling.googleapis.com/v1/services/6F81-5844-456A``
when the ``GOOGLE_API_KEY`` env var is set. The catalog only exposes
per-SKU prices (compute / RAM / GPU billed separately) so we publish a
*live GPU rate x gpu_count + static compute base* aggregate. The
static-compute base is derived once from the curated table by
subtracting the implied static GPU rate.

Honest source label:
``gcp-billing-catalog (live GPU + static compute)``.

Without ``GOOGLE_API_KEY``, GCP silently falls back to the static
table and a debug log is emitted.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from ophelian.pricing._http import _http_get_json, _url_quote
from ophelian.pricing.quotes import PriceQuote
from ophelian.pricing.static import STATIC_PRICES

logger = logging.getLogger("ophelian.pricing.live.gcp")

_GCP_COMPUTE_SERVICE_ID = "6F81-5844-456A"
_GCP_BILLING_URL = f"https://cloudbilling.googleapis.com/v1/services/{_GCP_COMPUTE_SERVICE_ID}/skus"

# Static per-GPU on-demand and spot rates we used when assembling the
# STATIC_PRICES table. Subtracting these gives us the implied compute
# base for each instance — the portion the live GPU rate gets added
# back on top of. Numbers from STATIC_PRICES_LAST_REVIEW.
_GCP_IMPLIED_GPU_RATES: dict[str, tuple[float, float]] = {
    # gpu_family: (on_demand_per_gpu_per_hour, spot_per_gpu_per_hour)
    "T4": (0.35, 0.13),
    "L4": (0.71, 0.28),
    "V100": (2.48, 0.74),
    "A100": (3.67, 1.40),
    "H100": (11.06, 3.75),  # a3-highgpu-8g rates / 8
}

# Cloud Billing Catalog descriptions use friendly GPU names like
# "Nvidia Tesla A100 GPU" and "Nvidia H100 80GB GPU". Map our family
# labels to substrings we can match against.
_GCP_GPU_DESCRIPTORS: dict[str, tuple[str, ...]] = {
    "T4": ("Tesla T4",),
    "L4": ("Nvidia L4", "L4 GPU"),
    "V100": ("Tesla V100",),
    "A100": ("Tesla A100",),
    "H100": ("H100 80GB", "H100 GPU"),
}

# Continent buckets the Cloud Billing API uses in SKU descriptions.
_GCP_REGION_CONTINENTS: dict[str, str] = {
    "us-central1": "Americas",
    "us-east1": "Americas",
    "us-east4": "Americas",
    "us-west1": "Americas",
    "us-west2": "Americas",
    "us-west4": "Americas",
    "europe-west1": "EMEA",
    "europe-west2": "EMEA",
    "europe-west3": "EMEA",
    "europe-west4": "EMEA",
    "asia-east1": "APAC",
    "asia-northeast1": "APAC",
    "asia-southeast1": "APAC",
}


def _fetch_gcp_billing_quotes(gpu_family: str, regions: list[str] | None) -> list[PriceQuote]:
    """Live GPU rate x gpu_count + static compute base for every known instance.

    Requires ``GOOGLE_API_KEY`` env var (Cloud Billing Catalog API
    needs an API key — service account is fine too via OAuth, but we
    keep the dependency surface tiny by sticking to the key path).
    Returns ``[]`` and logs at debug level if the key is missing,
    the catalog is unreachable, or the GPU descriptor is unknown.
    """
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_CLOUD_API_KEY")
    if not api_key:
        logger.debug(
            "Live GCP pricing skipped: GOOGLE_API_KEY not set "
            "(set it to query Cloud Billing Catalog API)."
        )
        return []

    family_upper = gpu_family.upper()
    descriptors = _GCP_GPU_DESCRIPTORS.get(family_upper)
    gcp_table = STATIC_PRICES.get("gcp", {}).get(family_upper, {})
    if not descriptors or not gcp_table:
        return []

    target_regions = list(regions) if regions else list(gcp_table.keys())
    target_regions = [r for r in target_regions if r in gcp_table and r in _GCP_REGION_CONTINENTS]
    if not target_regions:
        return []

    live_rates = _gcp_fetch_gpu_rates(api_key=api_key, descriptors=descriptors)
    if not live_rates:
        return []

    implied = _GCP_IMPLIED_GPU_RATES.get(family_upper)
    if implied is None:
        return []
    implied_on_demand, implied_spot = implied

    out: list[PriceQuote] = []
    for region in target_regions:
        continent = _GCP_REGION_CONTINENTS[region]
        for instance, gpu_count, static_hourly, is_spot in gcp_table[region]:
            tier_key = (continent, "Preemptible" if is_spot else "OnDemand")
            live_per_gpu = live_rates.get(tier_key)
            if live_per_gpu is None:
                continue
            implied_per_gpu = implied_spot if is_spot else implied_on_demand
            compute_base = max(0.0, static_hourly - implied_per_gpu * gpu_count)
            total = round(live_per_gpu * gpu_count + compute_base, 4)
            if total <= 0:
                continue
            out.append(
                PriceQuote(
                    provider="gcp",
                    region=region,
                    instance=instance,
                    gpu_family=family_upper,
                    gpu_count=gpu_count,
                    hourly_usd=total,
                    spot=is_spot,
                    source="gcp-billing-catalog",
                    notes=(
                        f"live GPU rate ${live_per_gpu:.3f}/GPU/h x {gpu_count} "
                        f"+ static compute base ${compute_base:.3f}/h"
                    ),
                )
            )
    return out


def _gcp_fetch_gpu_rates(
    *, api_key: str, descriptors: tuple[str, ...]
) -> dict[tuple[str, str], float]:
    """Return ``{(continent, usage_type): per_gpu_per_hour_usd}`` for *descriptors*.

    Walks the Cloud Billing Catalog SKUs page-by-page (capped at a
    handful of pages) and pulls out the GPU SKUs whose ``description``
    contains any of *descriptors*. Spot SKUs are detected via
    ``category.usageType == 'Preemptible'`` (Google's catalog still
    uses the legacy "Preemptible" label even for the new Spot tier).
    """
    rates: dict[tuple[str, str], float] = {}
    page_token = ""
    pages = 0
    while pages < 10:
        url = f"{_GCP_BILLING_URL}?key={api_key}&pageSize=2000"
        if page_token:
            url += f"&pageToken={_url_quote(page_token)}"
        payload = _http_get_json(url, timeout=10.0)
        if not isinstance(payload, dict):
            break
        skus = payload.get("skus")
        if not isinstance(skus, list):
            break
        for sku in skus:
            if not isinstance(sku, dict):
                continue
            description = str(sku.get("description", ""))
            if not any(token in description for token in descriptors):
                continue
            category = sku.get("category") or {}
            if not isinstance(category, dict):
                continue
            if category.get("resourceFamily") != "Compute":
                continue
            if category.get("resourceGroup") != "GPU":
                continue
            usage = str(category.get("usageType", ""))
            if usage not in {"OnDemand", "Preemptible"}:
                continue
            service_regions = sku.get("serviceRegions")
            if not isinstance(service_regions, list):
                continue
            continent = _gcp_continent_from_description(description)
            if continent is None:
                continue
            unit_price = _gcp_unit_price(sku)
            if unit_price is None:
                continue
            key = (continent, usage)
            existing = rates.get(key)
            if existing is None or unit_price < existing:
                rates[key] = unit_price
        next_token = payload.get("nextPageToken")
        if not isinstance(next_token, str) or not next_token:
            break
        page_token = next_token
        pages += 1
    return rates


def _gcp_continent_from_description(description: str) -> str | None:
    """Cloud Billing Catalog descriptions end with ``running in <continent>``."""
    for continent in ("Americas", "EMEA", "APAC"):
        if continent in description:
            return continent
    return None


def _gcp_unit_price(sku: dict[str, Any]) -> float | None:
    pricing_info = sku.get("pricingInfo")
    if not isinstance(pricing_info, list) or not pricing_info:
        return None
    expression = (
        pricing_info[0].get("pricingExpression") if isinstance(pricing_info[0], dict) else None
    )
    if not isinstance(expression, dict):
        return None
    tiered = expression.get("tieredRates")
    if not isinstance(tiered, list) or not tiered:
        return None
    unit_price = tiered[0].get("unitPrice") if isinstance(tiered[0], dict) else None
    if not isinstance(unit_price, dict):
        return None
    units = unit_price.get("units")
    nanos = unit_price.get("nanos")
    try:
        return float(int(units or 0)) + float(int(nanos or 0)) / 1e9
    except (TypeError, ValueError):
        return None


__all__ = ["_fetch_gcp_billing_quotes"]
