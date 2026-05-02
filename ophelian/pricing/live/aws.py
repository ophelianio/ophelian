"""Live AWS spot pricing via ``ec2.describe_spot_price_history``.

Free, public, no extra credentials beyond standard boto3. Falls back
to an empty list (with a debug log) if boto3 is not installed, no AWS
credentials are available, or any API call raises — the static table
covers the same ground for those cases.
"""

from __future__ import annotations

import logging

from ophelian.pricing.quotes import PriceQuote
from ophelian.pricing.static import STATIC_PRICES, _gpu_count_for_instance

logger = logging.getLogger("ophelian.pricing.live.aws")


def _fetch_aws_spot_quotes(gpu_family: str, regions: list[str] | None) -> list[PriceQuote]:
    """Query ``ec2.describe_spot_price_history`` for the given GPU family."""
    family_upper = gpu_family.upper()
    aws_table = STATIC_PRICES.get("aws", {}).get(family_upper, {})
    instance_types = sorted(
        {
            instance
            for region_entries in aws_table.values()
            for instance, _count, _price, *_rest in region_entries
        }
    )
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


__all__ = ["_fetch_aws_spot_quotes"]
