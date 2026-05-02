"""Live Azure pricing via the public Retail Prices API.

Endpoint: ``https://prices.azure.com/api/retail/prices`` — public, no
auth, no API key required. Returns true on-demand and Spot prices for
every Azure VM SKU; we filter to the SKUs we recognise from the static
table and skip Windows variants.

Returns ``[]`` if the network is unavailable or the API returns
malformed data — the static table covers the same ground for those
cases.
"""

from __future__ import annotations

import logging
from typing import Any

from ophelian.pricing._http import _http_get_json, _url_quote
from ophelian.pricing.quotes import PriceQuote
from ophelian.pricing.static import STATIC_PRICES, _gpu_count_for_instance

logger = logging.getLogger("ophelian.pricing.live.azure")

_AZURE_RETAIL_URL = "https://prices.azure.com/api/retail/prices"


def _fetch_azure_retail_quotes(gpu_family: str, regions: list[str] | None) -> list[PriceQuote]:
    """Hit the Azure Retail Prices API for every known SKU/region pair."""
    family_upper = gpu_family.upper()
    azure_table = STATIC_PRICES.get("azure", {}).get(family_upper, {})
    if not azure_table:
        return []

    target_regions = list(regions) if regions else list(azure_table.keys())
    target_regions = [r for r in target_regions if r in azure_table]
    if not target_regions:
        return []

    skus_by_region: dict[str, set[str]] = {}
    for region in target_regions:
        skus_by_region[region] = {instance for instance, *_ in azure_table[region]}

    out: list[PriceQuote] = []
    for region, skus in skus_by_region.items():
        for sku in skus:
            items = _azure_retail_query(region=region, sku_name=sku)
            for item in items:
                product = str(item.get("productName", ""))
                if "Windows" in product:
                    continue
                meter = str(item.get("meterName", ""))
                price = item.get("retailPrice", item.get("unitPrice"))
                if price is None:
                    continue
                try:
                    price_f = float(price)
                except (TypeError, ValueError):
                    continue
                if price_f <= 0:
                    continue
                is_spot = "Spot" in meter or "Low Priority" in meter
                gpu_count = _gpu_count_for_instance(sku, "azure")
                out.append(
                    PriceQuote(
                        provider="azure",
                        region=region,
                        instance=sku,
                        gpu_family=family_upper,
                        gpu_count=gpu_count,
                        hourly_usd=round(price_f, 4),
                        spot=is_spot,
                        source="azure-retail-api",
                    )
                )
    return out


def _azure_retail_query(*, region: str, sku_name: str) -> list[dict[str, Any]]:
    """One ``$filter`` query against the Azure Retail Prices API.

    Always returns a list (possibly empty); never raises. Pages
    through ``NextPageLink`` up to a small bound so a stray
    misconfigured query can never spin forever.
    """
    odata_filter = (
        f"serviceName eq 'Virtual Machines' "
        f"and armRegionName eq '{region}' "
        f"and armSkuName eq '{sku_name}'"
    )
    url = f"{_AZURE_RETAIL_URL}?$filter={_url_quote(odata_filter)}"
    items: list[dict[str, Any]] = []
    pages = 0
    while url and pages < 5:
        payload = _http_get_json(url, timeout=8.0)
        if not isinstance(payload, dict):
            break
        page_items = payload.get("Items")
        if isinstance(page_items, list):
            items.extend(item for item in page_items if isinstance(item, dict))
        next_link = payload.get("NextPageLink")
        url = next_link if isinstance(next_link, str) and next_link else ""
        pages += 1
    return items


__all__ = ["_azure_retail_query", "_fetch_azure_retail_quotes"]
