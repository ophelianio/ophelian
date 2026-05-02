"""Tests for the static price table + cost lookups."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from ophelian.pricing import (
    STATIC_PRICES,
    STATIC_PRICES_LAST_REVIEW,
    PriceQuote,
    RouterDecision,
    explain,
    fetch_live,
    lookup_cheapest,
    static_quotes,
)


def test_static_prices_cover_three_clouds() -> None:
    assert set(STATIC_PRICES.keys()) == {"aws", "gcp", "azure"}


def test_static_prices_have_a100_for_every_cloud() -> None:
    for cloud in ("aws", "gcp", "azure"):
        assert "A100" in STATIC_PRICES[cloud], f"{cloud} missing A100 prices"


def test_static_prices_review_date_is_iso_like() -> None:
    # Cheap structural check: bump it whenever the table is refreshed.
    assert len(STATIC_PRICES_LAST_REVIEW) == 10
    assert STATIC_PRICES_LAST_REVIEW[4] == "-"
    assert STATIC_PRICES_LAST_REVIEW[7] == "-"


def test_static_quotes_filters_by_provider() -> None:
    quotes = static_quotes("A100", providers=["gcp"])
    assert quotes
    assert {q.provider for q in quotes} == {"gcp"}


def test_static_quotes_filters_by_region() -> None:
    quotes = static_quotes("T4", regions=["us-east-1"])
    assert quotes
    assert {q.region for q in quotes} == {"us-east-1"}


def test_static_quotes_filters_by_spot_flag() -> None:
    on_demand = static_quotes("A100", spot=False)
    spot = static_quotes("A100", spot=True)
    assert on_demand and spot
    assert all(not q.spot for q in on_demand)
    assert all(q.spot for q in spot)


def test_lookup_cheapest_picks_minimum_hourly_usd() -> None:
    cheapest = lookup_cheapest("A100", spot=True)
    assert cheapest is not None
    candidates = static_quotes("A100", spot=True)
    assert cheapest.hourly_usd == min(q.hourly_usd for q in candidates)


def test_lookup_cheapest_unknown_gpu_returns_none() -> None:
    assert lookup_cheapest("BANANA-GPU") is None


def test_lookup_cheapest_unknown_region_returns_none() -> None:
    assert lookup_cheapest("A100", regions=["mars-1"]) is None


def test_price_quote_annual_cost() -> None:
    q = PriceQuote(
        provider="aws",
        region="us-east-1",
        instance="g5.xlarge",
        gpu_family="A10G",
        gpu_count=1,
        hourly_usd=1.0,
    )
    assert q.annual_cost(hours=10) == 10.0


def test_explain_includes_runners_up() -> None:
    quote = lookup_cheapest("A100", spot=True)
    assert quote is not None
    decision = RouterDecision(quote=quote, considered=static_quotes("A100", spot=True))
    text = explain(decision)
    assert "Cheapest A100" in text
    assert "Runners-up" in text or len(decision.considered) <= 1


def test_router_decision_savings_vs_most_expensive_is_non_negative() -> None:
    quote = lookup_cheapest("A100", spot=True)
    assert quote is not None
    decision = RouterDecision(quote=quote, considered=static_quotes("A100"))
    assert decision.savings_vs_most_expensive >= 0.0


def test_fetch_live_consults_disk_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPHELIAN_PRICING_CACHE_DIR", str(tmp_path / "cache.json"))
    payload = {
        "_fetched_at": time.time(),
        "quotes": [
            {
                "provider": "aws",
                "region": "us-east-1",
                "instance": "g5.xlarge",
                "gpu_family": "A10G",
                "gpu_count": 1,
                "hourly_usd": 0.10,
                "spot": True,
                "source": "live",
                "notes": "test",
            }
        ],
    }
    (tmp_path / "cache.json").write_text(json.dumps(payload))
    quotes = fetch_live("A10G")
    assert any(q.source == "live" and q.hourly_usd == 0.10 for q in quotes)


def _disable_live_http(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every live fetcher return ``[]`` so the test exercises only
    the cache path. Azure's Retail Prices API is public/no-auth, so
    without this the live path would actually run during CI."""
    from ophelian.pricing import live as _live

    monkeypatch.setattr(_live, "_fetch_aws_spot_quotes", lambda *a, **k: [])
    monkeypatch.setattr(_live, "_fetch_azure_retail_quotes", lambda *a, **k: [])
    monkeypatch.setattr(_live, "_fetch_gcp_billing_quotes", lambda *a, **k: [])


def test_fetch_live_ignores_stale_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _disable_live_http(monkeypatch)
    monkeypatch.setenv("OPHELIAN_PRICING_CACHE_DIR", str(tmp_path / "cache.json"))
    payload = {
        "_fetched_at": 0.0,  # 1970 — definitely older than 24h
        "quotes": [
            {
                "provider": "aws",
                "region": "us-east-1",
                "instance": "x",
                "gpu_family": "A100",
                "gpu_count": 1,
                "hourly_usd": 0.01,
            }
        ],
    }
    (tmp_path / "cache.json").write_text(json.dumps(payload))
    assert fetch_live("A100") == []


def test_fetch_live_missing_cache_returns_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _disable_live_http(monkeypatch)
    monkeypatch.setenv("OPHELIAN_PRICING_CACHE_DIR", str(tmp_path / "missing.json"))
    assert fetch_live("A100") == []


def test_fetch_live_aws_spot_uses_describe_spot_price_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Live path should hit ec2.describe_spot_price_history when cache is stale."""
    monkeypatch.setenv("OPHELIAN_PRICING_CACHE_DIR", str(tmp_path / "cache.json"))

    calls: list[dict[str, object]] = []

    class _FakeEC2:
        def describe_spot_price_history(self, **kwargs: object) -> dict[str, object]:
            calls.append(kwargs)
            return {
                "SpotPriceHistory": [
                    {
                        "InstanceType": "p4d.24xlarge",
                        "SpotPrice": "12.345",
                        "AvailabilityZone": "us-east-1a",
                    },
                    {
                        "InstanceType": "p4d.24xlarge",
                        "SpotPrice": "11.999",
                        "AvailabilityZone": "us-east-1b",
                    },
                ]
            }

    def _fake_client(name: str, region_name: str | None = None) -> _FakeEC2:
        assert name == "ec2"
        assert region_name == "us-east-1"
        return _FakeEC2()

    import boto3

    monkeypatch.setattr(boto3, "client", _fake_client)

    quotes = fetch_live("A100", providers=["aws"], regions=["us-east-1"])

    assert calls, "describe_spot_price_history was not called"
    assert any(
        q.provider == "aws"
        and q.region == "us-east-1"
        and q.instance == "p4d.24xlarge"
        and q.spot is True
        and q.source == "aws-spot-history"
        and q.hourly_usd == pytest.approx(11.999, rel=1e-3)
        for q in quotes
    ), quotes


def _install_fake_urlopen(
    monkeypatch: pytest.MonkeyPatch,
    handler: object,
) -> list[str]:
    """Install a fake ``urllib.request.urlopen`` that calls *handler(url)*.

    ``handler`` should return a JSON-serialisable dict for known URLs
    or raise ``URLError`` to simulate failure. Returns the list that
    will collect every URL the code under test fetched.
    """
    import io
    from urllib import request as urllib_request

    seen: list[str] = []

    class _FakeResp:
        def __init__(self, payload: dict[str, object]) -> None:
            self._buf = io.BytesIO(json.dumps(payload).encode("utf-8"))

        def __enter__(self) -> _FakeResp:
            return self

        def __exit__(self, *exc: object) -> None:
            self._buf.close()

        def read(self) -> bytes:
            return self._buf.read()

    def _fake_urlopen(req: object, timeout: float = 8.0) -> _FakeResp:
        url = req.full_url if hasattr(req, "full_url") else str(req)
        seen.append(url)
        payload = handler(url)  # type: ignore[operator]
        return _FakeResp(payload)

    monkeypatch.setattr(urllib_request, "urlopen", _fake_urlopen)
    return seen


def test_fetch_live_azure_retail_api_picks_spot_meter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Azure live path should hit the Retail Prices API and tag
    Spot meters as ``spot=True`` while skipping Windows variants."""
    monkeypatch.setenv("OPHELIAN_PRICING_CACHE_DIR", str(tmp_path / "cache.json"))
    # AWS + GCP are out of scope for this assertion; mute them so the
    # cache write only contains Azure quotes.
    from ophelian.pricing import live as _live

    monkeypatch.setattr(_live, "_fetch_aws_spot_quotes", lambda *a, **k: [])
    monkeypatch.setattr(_live, "_fetch_gcp_billing_quotes", lambda *a, **k: [])

    def handler(url: str) -> dict[str, object]:
        assert "prices.azure.com/api/retail/prices" in url
        return {
            "Items": [
                {
                    "armSkuName": "Standard_NC24ads_A100_v4",
                    "armRegionName": "eastus",
                    "productName": "Virtual Machines NCadsv4 Series",
                    "meterName": "NC24ads A100 v4 Spot",
                    "retailPrice": 0.7345,
                    "currencyCode": "USD",
                },
                {
                    "armSkuName": "Standard_NC24ads_A100_v4",
                    "armRegionName": "eastus",
                    "productName": "Virtual Machines NCadsv4 Series",
                    "meterName": "NC24ads A100 v4",
                    "retailPrice": 3.673,
                    "currencyCode": "USD",
                },
                {
                    "armSkuName": "Standard_NC24ads_A100_v4",
                    "armRegionName": "eastus",
                    "productName": "Virtual Machines NCadsv4 Series Windows",
                    "meterName": "NC24ads A100 v4",
                    "retailPrice": 99.99,
                    "currencyCode": "USD",
                },
            ],
            "NextPageLink": "",
        }

    seen = _install_fake_urlopen(monkeypatch, handler)

    quotes = fetch_live("A100", providers=["azure"], regions=["eastus"])

    assert seen, "Azure Retail Prices API was not called"
    spot = [q for q in quotes if q.provider == "azure" and q.spot]
    on_demand = [q for q in quotes if q.provider == "azure" and not q.spot]
    assert spot, "expected at least one Azure Spot quote"
    assert on_demand, "expected at least one Azure on-demand quote"
    assert spot[0].source == "azure-retail-api"
    assert spot[0].hourly_usd == pytest.approx(0.7345)
    # Windows variant must be filtered out.
    assert not any(q.hourly_usd == pytest.approx(99.99) for q in quotes)


def test_fetch_live_gcp_skipped_without_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GCP live pricing should be a graceful no-op when
    ``GOOGLE_API_KEY`` (or ``GOOGLE_CLOUD_API_KEY``) is unset."""
    monkeypatch.setenv("OPHELIAN_PRICING_CACHE_DIR", str(tmp_path / "cache.json"))
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_API_KEY", raising=False)
    from ophelian.pricing import live as _live

    monkeypatch.setattr(_live, "_fetch_aws_spot_quotes", lambda *a, **k: [])
    monkeypatch.setattr(_live, "_fetch_azure_retail_quotes", lambda *a, **k: [])

    quotes = fetch_live("A100", providers=["gcp"], regions=["us-central1"])
    assert quotes == []


def test_fetch_live_gcp_billing_catalog_overlays_live_gpu_rate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With ``GOOGLE_API_KEY`` set the GCP path should query the Cloud
    Billing Catalog API and produce ``hourly_usd = live_gpu_rate x
    gpu_count + static_compute_base`` for every known instance."""
    monkeypatch.setenv("OPHELIAN_PRICING_CACHE_DIR", str(tmp_path / "cache.json"))
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-test-key-do-not-use")
    from ophelian.pricing import live as _live

    monkeypatch.setattr(_live, "_fetch_aws_spot_quotes", lambda *a, **k: [])
    monkeypatch.setattr(_live, "_fetch_azure_retail_quotes", lambda *a, **k: [])

    def handler(url: str) -> dict[str, object]:
        assert "cloudbilling.googleapis.com" in url
        assert "key=fake-test-key-do-not-use" in url
        return {
            "skus": [
                {
                    "description": "Spot Preemptible Nvidia Tesla A100 GPU running in Americas",
                    "category": {
                        "resourceFamily": "Compute",
                        "resourceGroup": "GPU",
                        "usageType": "Preemptible",
                    },
                    "serviceRegions": ["us-central1", "us-east1"],
                    "pricingInfo": [
                        {
                            "pricingExpression": {
                                "tieredRates": [
                                    {"unitPrice": {"units": "1", "nanos": 100_000_000}},
                                ]
                            }
                        }
                    ],
                },
                {
                    "description": "Nvidia Tesla A100 GPU running in Americas",
                    "category": {
                        "resourceFamily": "Compute",
                        "resourceGroup": "GPU",
                        "usageType": "OnDemand",
                    },
                    "serviceRegions": ["us-central1", "us-east1"],
                    "pricingInfo": [
                        {
                            "pricingExpression": {
                                "tieredRates": [
                                    {"unitPrice": {"units": "3", "nanos": 200_000_000}},
                                ]
                            }
                        }
                    ],
                },
            ],
            "nextPageToken": "",
        }

    _install_fake_urlopen(monkeypatch, handler)

    quotes = fetch_live("A100", providers=["gcp"], regions=["us-central1"])
    spot = [q for q in quotes if q.provider == "gcp" and q.spot]
    on_demand = [q for q in quotes if q.provider == "gcp" and not q.spot]
    assert spot, "expected at least one GCP spot quote"
    assert on_demand, "expected at least one GCP on-demand quote"

    # Static spot for a2-highgpu-1g is $1.40/h with 1 A100 @ implied
    # $1.40/GPU; live spot is $1.10/GPU → expected total $1.10.
    assert spot[0].source == "gcp-billing-catalog"
    assert spot[0].hourly_usd == pytest.approx(1.10, abs=0.01)
    assert "live GPU rate" in spot[0].notes


def test_lookup_cheapest_prefers_live_over_static_when_cheaper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPHELIAN_PRICING_CACHE_DIR", str(tmp_path / "cache.json"))
    fresh_cache = {
        "_fetched_at": time.time(),
        "quotes": [
            {
                "provider": "aws",
                "region": "us-east-1",
                "instance": "p4d.24xlarge",
                "gpu_family": "A100",
                "gpu_count": 8,
                "hourly_usd": 0.50,
                "spot": True,
                "source": "aws-spot-history",
            }
        ],
    }
    (tmp_path / "cache.json").write_text(json.dumps(fresh_cache))

    chosen = lookup_cheapest("A100", providers=["aws"], allow_live=True)
    assert chosen is not None
    assert chosen.source == "aws-spot-history"
    assert chosen.hourly_usd == pytest.approx(0.50)
