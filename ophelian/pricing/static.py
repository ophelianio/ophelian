"""Hand-curated static GPU pricing fallback table.

This module exists so the auto-router works even when the user's
environment has no network access to live pricing endpoints — there's
nothing more embarrassing than an "ML infra tool" that refuses to
start because a pricing API rate-limited it.

Numbers are captured from the public on-demand pricing pages of each
cloud at :data:`STATIC_PRICES_LAST_REVIEW`. They are deliberately
*conservative* — we round up to the nearest cent — because we'd rather
the router pick the cheaper option when prices have actually dropped
than over-promise a saving that no longer exists.

Every entry is ``(instance, gpu_count, hourly_usd, spot=False)``. We
include both on-demand and the headline spot rate where the cloud
publishes one; the router treats spot as a separate tier.
"""

from __future__ import annotations

from ophelian.pricing.quotes import PriceQuote

STATIC_PRICES_LAST_REVIEW = "2026-04-15"
"""When the static fallback prices were last spot-checked.

Bump this string whenever you refresh :data:`STATIC_PRICES`. The
auto-router surfaces it in ``--explain`` output so users know how stale
the fallback is.
"""


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
            "us-central1": [
                ("n1-standard-4+t4", 1, 0.350, False),
                ("n1-standard-4+t4", 1, 0.13, True),
            ],
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
            "eastus": [
                ("Standard_NC4as_T4_v3", 1, 0.526, False),
                ("Standard_NC4as_T4_v3", 1, 0.20, True),
            ],
            "westeurope": [("Standard_NC4as_T4_v3", 1, 0.586, False)],
        },
        "V100": {
            "eastus": [("Standard_NC6s_v3", 1, 3.06, False), ("Standard_NC6s_v3", 1, 0.92, True)],
        },
        "A100": {
            "eastus": [
                ("Standard_NC24ads_A100_v4", 1, 3.67, False),
                ("Standard_NC24ads_A100_v4", 1, 1.50, True),
            ],
            "westeurope": [("Standard_NC24ads_A100_v4", 1, 4.10, False)],
        },
        "H100": {
            "eastus": [("Standard_NC40ads_H100_v5", 1, 12.29, False)],
        },
    },
}


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


__all__ = [
    "STATIC_PRICES",
    "STATIC_PRICES_LAST_REVIEW",
    "static_quotes",
]
