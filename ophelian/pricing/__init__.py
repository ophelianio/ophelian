"""GPU pricing tables + Auto-router cost lookups.

The :func:`lookup_cheapest` function is the small primitive on which
:func:`ophelian.envs.auto.Auto` builds: given a GPU family and a list
of regions, return the cheapest provider/region/instance triple that
satisfies the request.

We ship a hand-curated **static fallback** table covering AWS, GCP and
Azure for the most-common GPU families (T4, L4, V100, A10G, A100, H100).
The numbers are honest order-of-magnitude on-demand prices captured at
release time and recorded in :data:`STATIC_PRICES_LAST_REVIEW`. They
exist so the router works even when the user's environment has no
network access to live pricing endpoints — there's nothing more
embarrassing than an "ML infra tool" that refuses to start because a
pricing API rate-limited it.

When the optional :pypi:`requests` dependency is available we *also*
support a ``fetch_live`` path that hits the official pricing APIs and
caches results on disk for ``cache_ttl_seconds`` (default: 24 h). The
fetcher is opt-in — :func:`lookup_cheapest` only consults it when the
caller passes ``allow_live=True``.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("ophelian.pricing")


STATIC_PRICES_LAST_REVIEW = "2026-04-15"
"""When the static fallback prices were last spot-checked.

Bump this string whenever you refresh :data:`STATIC_PRICES`. The
auto-router surfaces it in ``--explain`` output so users know how stale
the fallback is.
"""


@dataclass(frozen=True)
class PriceQuote:
    """A single (provider, region, instance) on-demand or spot quote.

    ``hourly_usd`` is the per-hour billing rate; ``spot`` indicates
    whether the price is the spot/preemptible discount or the
    on-demand rate. ``gpu_family`` is the user-facing label
    (``"A100"``, ``"H100"``, ...) — *not* the SKU-specific name (e.g.
    ``"nvidia-tesla-a100"`` on GCP) which lives in ``instance``.
    """

    provider: str
    region: str
    instance: str
    gpu_family: str
    gpu_count: int
    hourly_usd: float
    spot: bool = False
    source: str = "static"
    notes: str = ""

    def annual_cost(self, hours: float = 24 * 30) -> float:
        return self.hourly_usd * hours


# ---------------------------------------------------------------------------
# Static fallback table
# ---------------------------------------------------------------------------
# All numbers are captured from the public on-demand pricing pages of
# each cloud at the date in STATIC_PRICES_LAST_REVIEW. They are deliberately
# *conservative* — we round up to the nearest cent — because we'd rather
# the router pick the cheaper option when prices have actually dropped
# than over-promise a saving that no longer exists.
#
# Every entry is `(instance, gpu_count, hourly_usd, spot=False)`. We
# include both on-demand and the headline spot rate where the cloud
# publishes one; the router treats spot as a separate tier.

_T = list[tuple[str, int, float, bool]]


STATIC_PRICES: dict[str, dict[str, dict[str, _T]]] = {
    "aws": {
        "T4": {
            "us-east-1": [("g4dn.xlarge", 1, 0.526, False), ("g4dn.xlarge", 1, 0.20, True)],
            "us-west-2": [("g4dn.xlarge", 1, 0.526, False)],
            "eu-west-1": [("g4dn.xlarge", 1, 0.586, False)],
        },
        "A10G": {
            "us-east-1": [("g5.xlarge", 1, 1.006, False), ("g5.xlarge", 1, 0.40, True)],
            "us-west-2": [("g5.xlarge", 1, 1.006, False)],
        },
        "V100": {
            "us-east-1": [("p3.2xlarge", 1, 3.06, False), ("p3.2xlarge", 1, 1.10, True)],
        },
        "A100": {
            "us-east-1": [("p4d.24xlarge", 8, 32.77, False), ("p4d.24xlarge", 8, 12.0, True)],
            "us-west-2": [("p4d.24xlarge", 8, 32.77, False)],
        },
        "H100": {
            "us-east-1": [("p5.48xlarge", 8, 98.32, False), ("p5.48xlarge", 8, 35.0, True)],
        },
    },
    "gcp": {
        "T4": {
            "us-central1": [("n1-standard-4+t4", 1, 0.350, False), ("n1-standard-4+t4", 1, 0.13, True)],
            "europe-west4": [("n1-standard-4+t4", 1, 0.385, False)],
        },
        "L4": {
            "us-central1": [("g2-standard-4", 1, 0.71, False), ("g2-standard-4", 1, 0.28, True)],
        },
        "V100": {
            "us-central1": [("n1-standard-8+v100", 1, 2.48, False)],
        },
        "A100": {
            "us-central1": [("a2-highgpu-1g", 1, 3.67, False), ("a2-highgpu-1g", 1, 1.40, True)],
            "us-east1": [("a2-highgpu-1g", 1, 3.67, False)],
        },
        "H100": {
            "us-central1": [("a3-highgpu-8g", 8, 88.49, False), ("a3-highgpu-8g", 8, 30.0, True)],
        },
    },
    "azure": {
        "T4": {
            "eastus": [("Standard_NC4as_T4_v3", 1, 0.526, False), ("Standard_NC4as_T4_v3", 1, 0.20, True)],
            "westeurope": [("Standard_NC4as_T4_v3", 1, 0.586, False)],
        },
        "V100": {
            "eastus": [("Standard_NC6s_v3", 1, 3.06, False), ("Standard_NC6s_v3", 1, 0.92, True)],
        },
        "A100": {
            "eastus": [("Standard_NC24ads_A100_v4", 1, 3.67, False), ("Standard_NC24ads_A100_v4", 1, 1.50, True)],
            "westeurope": [("Standard_NC24ads_A100_v4", 1, 4.10, False)],
        },
        "H100": {
            "eastus": [("Standard_NC40ads_H100_v5", 1, 12.29, False)],
        },
    },
}


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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def static_quotes(
    gpu_family: str,
    *,
    providers: list[str] | None = None,
    regions: list[str] | None = None,
    spot: bool | None = None,
) -> list[PriceQuote]:
    """Return every quote from the static table that matches the filters."""
    family_upper = gpu_family.upper()
    out: list[PriceQuote] = []
    for provider, families in STATIC_PRICES.items():
        if providers and provider not in providers:
            continue
        family_table = families.get(family_upper)
        if not family_table:
            continue
        for region, entries in family_table.items():
            if regions and region not in regions:
                continue
            for instance, gpu_count, hourly, is_spot in entries:
                if spot is True and not is_spot:
                    continue
                if spot is False and is_spot:
                    continue
                out.append(
                    PriceQuote(
                        provider=provider,
                        region=region,
                        instance=instance,
                        gpu_family=family_upper,
                        gpu_count=gpu_count,
                        hourly_usd=hourly,
                        spot=is_spot,
                        source="static",
                        notes=f"static fallback as of {STATIC_PRICES_LAST_REVIEW}",
                    )
                )
    return out


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
    quotes = static_quotes(
        gpu_family, providers=providers, regions=regions, spot=spot
    )
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
            q for q in cache.get("quotes", [])
            if isinstance(q, dict) and q.get("provider") not in refreshed_providers
        ]
        for provider_quotes in fetched.values():
            preserved_entries.extend(
                {
                    "provider": q.provider, "region": q.region,
                    "instance": q.instance, "gpu_family": q.gpu_family,
                    "gpu_count": q.gpu_count, "hourly_usd": q.hourly_usd,
                    "spot": q.spot, "source": q.source, "notes": q.notes,
                }
                for q in provider_quotes
            )
        _save_cache({"_fetched_at": time.time(), "quotes": preserved_entries})

    return out


def _fetch_aws_spot_quotes(
    gpu_family: str, regions: list[str] | None
) -> list[PriceQuote]:
    """Query ``ec2.describe_spot_price_history`` for the given GPU family.

    Falls back to an empty list (with a debug log) if ``boto3`` is not
    installed, no AWS credentials are available, or any API call
    raises. The static table covers the same ground for those cases.
    """
    family_upper = gpu_family.upper()
    aws_table = STATIC_PRICES.get("aws", {}).get(family_upper, {})
    instance_types = sorted({
        instance
        for region_entries in aws_table.values()
        for instance, _count, _price, *_rest in region_entries
    })
    if not instance_types:
        return []

    target_regions = list(regions) if regions else list(aws_table.keys())
    if not target_regions:
        return []

    try:
        import boto3
        from botocore.exceptions import BotoCoreError, ClientError
    except ImportError:  # pragma: no cover - boto3 optional
        logger.debug("Live AWS pricing skipped: boto3 not installed.")
        return []

    out: list[PriceQuote] = []
    for region in target_regions:
        try:
            ec2 = boto3.client("ec2", region_name=region)
            resp = ec2.describe_spot_price_history(
                InstanceTypes=instance_types,
                ProductDescriptions=["Linux/UNIX"],
                MaxResults=len(instance_types) * 5,
            )
        except (BotoCoreError, ClientError, Exception) as exc:  # pragma: no cover
            logger.debug("AWS spot history failed for %s: %s", region, exc)
            continue
        latest: dict[str, tuple[float, str]] = {}
        for item in resp.get("SpotPriceHistory", []):
            instance = item.get("InstanceType")
            price = item.get("SpotPrice")
            if not instance or price is None:
                continue
            try:
                price_f = float(price)
            except (TypeError, ValueError):
                continue
            current = latest.get(instance)
            if current is None or price_f < current[0]:
                latest[instance] = (price_f, item.get("AvailabilityZone", region))
        for instance, (price_f, _az) in latest.items():
            out.append(
                PriceQuote(
                    provider="aws",
                    region=region,
                    instance=instance,
                    gpu_family=gpu_family.upper(),
                    gpu_count=_gpu_count_for_instance(instance, "aws"),
                    hourly_usd=round(price_f, 4),
                    spot=True,
                    source="aws-spot-history",
                )
            )
    return out


# ---------------------------------------------------------------------------
# Azure Retail Prices API (https://prices.azure.com)
# ---------------------------------------------------------------------------
# Public, no auth, no API key. Returns OnDemand and Spot/Low Priority
# unit prices for every Azure VM SKU. We filter to the SKUs we know
# about from the static table and skip the Windows variants.

_AZURE_RETAIL_URL = "https://prices.azure.com/api/retail/prices"


def _fetch_azure_retail_quotes(
    gpu_family: str, regions: list[str] | None
) -> list[PriceQuote]:
    """Hit the Azure Retail Prices API for every known SKU/region pair.

    Returns ``[]`` if ``urllib`` blows up (no network, DNS down, etc.)
    or if the API returns malformed data — the static table covers
    the same ground for those cases.
    """
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
        skus_by_region[region] = {
            instance for instance, *_ in azure_table[region]
        }

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


# ---------------------------------------------------------------------------
# GCP Cloud Billing Catalog API (live GPU + static compute base)
# ---------------------------------------------------------------------------
# The catalog only exposes per-SKU prices (compute, RAM, GPU billed
# separately); we publish a *live GPU rate x gpu_count + static
# compute base* aggregate. The static-compute base is derived once
# from the curated table by subtracting the implied static GPU rate.

_GCP_COMPUTE_SERVICE_ID = "6F81-5844-456A"
_GCP_BILLING_URL = (
    f"https://cloudbilling.googleapis.com/v1/services/{_GCP_COMPUTE_SERVICE_ID}/skus"
)

# Static per-GPU on-demand and spot rates we used when assembling the
# STATIC_PRICES table. Subtracting these gives us the implied compute
# base for each instance — the portion the live GPU rate gets added
# back on top of. Numbers from STATIC_PRICES_LAST_REVIEW.
_GCP_IMPLIED_GPU_RATES: dict[str, tuple[float, float]] = {
    # gpu_family: (on_demand_per_gpu_per_hour, spot_per_gpu_per_hour)
    "T4":   (0.35,  0.13),
    "L4":   (0.71,  0.28),
    "V100": (2.48,  0.74),
    "A100": (3.67,  1.40),
    "H100": (11.06, 3.75),  # a3-highgpu-8g rates / 8
}

# Cloud Billing Catalog descriptions use friendly GPU names like
# "Nvidia Tesla A100 GPU" and "Nvidia H100 80GB GPU". Map our family
# labels to substrings we can match against.
_GCP_GPU_DESCRIPTORS: dict[str, tuple[str, ...]] = {
    "T4":   ("Tesla T4",),
    "L4":   ("Nvidia L4", "L4 GPU"),
    "V100": ("Tesla V100",),
    "A100": ("Tesla A100",),
    "H100": ("H100 80GB", "H100 GPU"),
}

# Continent buckets the Cloud Billing API uses in SKU descriptions.
_GCP_REGION_CONTINENTS: dict[str, str] = {
    "us-central1": "Americas", "us-east1": "Americas",
    "us-east4": "Americas", "us-west1": "Americas",
    "us-west2": "Americas", "us-west4": "Americas",
    "europe-west1": "EMEA", "europe-west2": "EMEA",
    "europe-west3": "EMEA", "europe-west4": "EMEA",
    "asia-east1": "APAC", "asia-northeast1": "APAC",
    "asia-southeast1": "APAC",
}


def _fetch_gcp_billing_quotes(
    gpu_family: str, regions: list[str] | None
) -> list[PriceQuote]:
    """Live GPU rate x gpu_count + static compute base for every known instance.

    Requires ``GOOGLE_API_KEY`` env var (Cloud Billing Catalog API
    needs an API key — service account is fine too via OAuth, but we
    keep the dependency surface tiny by sticking to the key path).
    Returns ``[]`` and logs at debug level if the key is missing,
    the catalog is unreachable, or the GPU descriptor is unknown.
    """
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get(
        "GOOGLE_CLOUD_API_KEY"
    )
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
    target_regions = [
        r for r in target_regions
        if r in gcp_table and r in _GCP_REGION_CONTINENTS
    ]
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
            compute_base = max(
                0.0, static_hourly - implied_per_gpu * gpu_count
            )
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
    expression = pricing_info[0].get("pricingExpression") if isinstance(
        pricing_info[0], dict
    ) else None
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


# ---------------------------------------------------------------------------
# Tiny stdlib HTTP helper (kept here so the module has zero hard
# dependencies beyond boto3 for AWS).
# ---------------------------------------------------------------------------


def _http_get_json(url: str, *, timeout: float = 8.0) -> Any:
    """GET *url* and return parsed JSON. Returns ``None`` on any error."""
    try:
        from urllib.error import URLError
        from urllib.request import Request, urlopen
    except ImportError:  # pragma: no cover - urllib is stdlib
        return None
    try:
        request = Request(url, headers={"Accept": "application/json"})
        with urlopen(request, timeout=timeout) as response:
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


def _gpu_count_for_instance(instance: str, provider: str) -> int:
    """Best-effort gpu_count lookup against the static table."""
    for family_table in STATIC_PRICES.get(provider, {}).values():
        if not isinstance(family_table, dict):
            continue
        for region_entries in family_table.values():
            for inst, count, *_ in region_entries:
                if inst == instance:
                    return int(count)
    return 1


@dataclass
class RouterDecision:
    """The chosen :class:`PriceQuote` plus context for ``--explain`` output."""

    quote: PriceQuote
    considered: list[PriceQuote] = field(default_factory=list)

    @property
    def savings_vs_most_expensive(self) -> float:
        if not self.considered:
            return 0.0
        worst = max(self.considered, key=lambda q: q.hourly_usd).hourly_usd
        return max(0.0, worst - self.quote.hourly_usd)


def explain(decision: RouterDecision) -> str:
    """Render a short, copy-pasteable explanation of a router decision."""
    q = decision.quote
    lines = [
        f"Cheapest {q.gpu_family}: {q.provider}/{q.region} on {q.instance}",
        f"  {q.hourly_usd:.3f} USD/hour ({'spot' if q.spot else 'on-demand'}, source={q.source})",
    ]
    if decision.considered:
        runners_up = [
            qq
            for qq in sorted(decision.considered, key=lambda q: q.hourly_usd)[:5]
            if qq is not q
        ]
        if runners_up:
            lines.append("  Runners-up:")
            for qq in runners_up:
                lines.append(
                    f"    - {qq.provider}/{qq.region} {qq.instance:>30}  "
                    f"{qq.hourly_usd:.3f} USD/h"
                )
    return "\n".join(lines)


__all__ = [
    "STATIC_PRICES",
    "STATIC_PRICES_LAST_REVIEW",
    "PriceQuote",
    "RouterDecision",
    "explain",
    "fetch_live",
    "lookup_cheapest",
    "static_quotes",
]
