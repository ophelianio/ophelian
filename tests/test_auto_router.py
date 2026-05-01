"""Tests for the Auto cost router."""

from __future__ import annotations

from typing import Any

import pytest
from ophelian.envs.auto import (
    SUPPORTED_PROVIDERS,
    Auto,
    AutoRouterError,
    _DryRunProvider,
    _gcp_instance_for,
)
from ophelian.pricing import PriceQuote

_CRED_VARS = (
    "AWS_ACCESS_KEY_ID",
    "AWS_PROFILE",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "AZURE_CLIENT_ID",
    "AZURE_SUBSCRIPTION_ID",
    "AZURE_RESOURCE_GROUP",
    "AZURE_STORAGE_ACCOUNT",
    "GOOGLE_CLOUD_PROJECT",
)


@pytest.fixture(autouse=True)
def _isolate_credentials(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Pretend the test runner has *no* cloud credentials configured."""
    for var in _CRED_VARS:
        monkeypatch.delenv(var, raising=False)
    # Re-point home so the credentials-on-disk probes don't see real
    # `~/.aws`, `~/.config/gcloud`, `~/.azure` directories on the
    # contributor's machine.
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()


def test_supported_providers_are_three() -> None:
    assert SUPPORTED_PROVIDERS == ("aws", "gcp", "azure")


def test_gcp_instance_split_handles_plus_form() -> None:
    quote = PriceQuote(
        provider="gcp",
        region="us-central1",
        instance="n1-standard-4+t4",
        gpu_family="T4",
        gpu_count=1,
        hourly_usd=0.13,
        spot=True,
    )
    machine, gpu = _gcp_instance_for(quote)
    assert machine == "n1-standard-4"
    assert gpu == "nvidia-tesla-t4"


def test_gcp_instance_split_handles_a100() -> None:
    """A100 lives on the ``a2-*`` family — the GPU is baked into the
    SKU, so we infer ``gpu_type`` from the machine prefix instead of
    requiring a ``+gpu`` suffix. (Returning ``None`` here would make
    ``GCPConfig`` reject the construction because ``gpu_count > 0``
    requires a ``gpu_type``.)"""
    quote = PriceQuote(
        provider="gcp",
        region="us-central1",
        instance="a2-highgpu-1g",
        gpu_family="A100",
        gpu_count=1,
        hourly_usd=1.4,
        spot=True,
    )
    machine, gpu = _gcp_instance_for(quote)
    assert machine == "a2-highgpu-1g"
    assert gpu == "nvidia-tesla-a100"


def test_gcp_instance_split_handles_h100_and_l4() -> None:
    """``a3-*`` implies H100 and ``g2-*`` implies L4."""
    h100 = PriceQuote(
        provider="gcp",
        region="us-central1",
        instance="a3-highgpu-8g",
        gpu_family="H100",
        gpu_count=8,
        hourly_usd=30.0,
        spot=True,
    )
    machine, gpu = _gcp_instance_for(h100)
    assert (machine, gpu) == ("a3-highgpu-8g", "nvidia-h100-80gb")

    l4 = PriceQuote(
        provider="gcp",
        region="us-central1",
        instance="g2-standard-4",
        gpu_family="L4",
        gpu_count=1,
        hourly_usd=0.28,
        spot=True,
    )
    machine, gpu = _gcp_instance_for(l4)
    assert (machine, gpu) == ("g2-standard-4", "nvidia-l4")


def test_dry_run_returns_dryrun_provider_with_decision() -> None:
    provider = Auto(
        cheapest_gpu="A100",
        regions=["us-east-1", "us-central1", "eastus"],
        providers=["aws", "gcp", "azure"],
        spot=True,
        dry_run=True,
        require_credentials=False,
    )
    assert isinstance(provider, _DryRunProvider)
    decision = provider.decision
    assert decision.quote.gpu_family == "A100"
    assert decision.quote.provider in {"aws", "gcp", "azure"}
    assert decision.quote.hourly_usd > 0
    # The dry-run picker should always favour the spot tier when one
    # exists for the requested GPU family.
    assert decision.quote.spot is True


def test_dry_run_describe_mentions_provider_and_price() -> None:
    provider = Auto(cheapest_gpu="A100", spot=True, dry_run=True, require_credentials=False)
    text = provider.describe()
    assert "auto-dry-run" in text
    assert "USD/h" in text


def test_no_quotes_raises_router_error() -> None:
    with pytest.raises(AutoRouterError, match="No quotes"):
        Auto(
            cheapest_gpu="BANANA-GPU",
            dry_run=True,
            require_credentials=False,
        )


def test_unknown_region_raises_router_error() -> None:
    with pytest.raises(AutoRouterError, match="No quotes"):
        Auto(
            cheapest_gpu="A100",
            regions=["mars-1"],
            dry_run=True,
            require_credentials=False,
        )


def test_auto_constructs_real_gcp_provider_for_a100_non_dry_run(
    tmp_path: Any,
) -> None:
    """Regression: ``Auto(cheapest_gpu='A100')`` in the default
    non-dry-run path must hand back a real ``GCPProvider`` (or AWS, or
    Azure) without tripping ``GCPConfig`` validation. The blocker was
    ``_gcp_instance_for`` returning ``gpu_type=None`` for ``a2-*``
    machines while ``gpu_count > 0`` was still passed through, which
    raised ``ValidationError`` at construction time."""
    from ophelian.providers.gcp import GCPProvider
    from ophelian.providers.gcp_drivers import LocalGCPDriver
    from ophelian.stores.local import LocalArtifactStore

    provider = Auto(
        cheapest_gpu="A100",
        regions=["us-central1"],
        providers=["gcp"],
        spot=True,
        require_credentials=False,
        project="ophelian-test",
        artifact_bucket="ophelian-test-bucket",
        # Bypass `google-cloud-storage` install so the test runs in any
        # environment; the router contract under test is provider
        # construction, not the artifact store backend.
        store=LocalArtifactStore(tmp_path),
        driver=LocalGCPDriver(),
    )
    assert isinstance(provider, GCPProvider)
    assert provider.config.machine_type.startswith("a2-")
    assert provider.config.gpu_type == "nvidia-tesla-a100"
    assert provider.config.gpu_count == 1
    assert provider.config.spot is True
    # Router stamps the quoted hourly price on the provider for cost
    # accounting in the run summary.
    assert getattr(provider, "_router_quote_hourly_usd", None) is not None


def test_auto_constructs_real_gcp_provider_for_l4_non_dry_run(
    tmp_path: Any,
) -> None:
    """Same as above but for L4 (``g2-*`` family)."""
    from ophelian.providers.gcp import GCPProvider
    from ophelian.providers.gcp_drivers import LocalGCPDriver
    from ophelian.stores.local import LocalArtifactStore

    provider = Auto(
        cheapest_gpu="L4",
        regions=["us-central1"],
        providers=["gcp"],
        spot=True,
        require_credentials=False,
        project="ophelian-test",
        artifact_bucket="ophelian-test-bucket",
        store=LocalArtifactStore(tmp_path),
        driver=LocalGCPDriver(),
    )
    assert isinstance(provider, GCPProvider)
    assert provider.config.machine_type.startswith("g2-")
    assert provider.config.gpu_type == "nvidia-l4"
    assert provider.config.gpu_count >= 1


def test_picks_cheapest_provider_for_a100_spot() -> None:
    # Force offline so the test is deterministic regardless of which
    # cloud's spot price has dropped this week. The other tests cover
    # the live-pricing code paths with mocked HTTP.
    provider = Auto(
        cheapest_gpu="A100",
        providers=["aws", "gcp", "azure"],
        spot=True,
        dry_run=True,
        require_credentials=False,
        allow_live=False,
    )
    # In the v1.0 static table GCP a2-highgpu-1g spot is the
    # cheapest A100 quote at $1.40/h. The router should pick it.
    assert provider.decision.quote.provider == "gcp"
    assert provider.decision.quote.hourly_usd == pytest.approx(1.40)


def test_require_credentials_blocks_when_nothing_detected() -> None:
    # No credentials on disk thanks to the autouse fixture; without
    # `require_credentials=False` the router should refuse to launch
    # rather than charge an account it does not own.
    with pytest.raises(AutoRouterError, match="credentials"):
        Auto(
            cheapest_gpu="A100",
            providers=["aws", "gcp", "azure"],
            spot=True,
            dry_run=False,
        )


def test_dry_run_provider_skips_every_step() -> None:
    from ophelian.core.compiler import GraphCompiler
    from ophelian.core.nodes import Data, Pipeline, Train

    pipe = Pipeline(
        [
            Data(
                name="ds",
                source="memory://toy",
                format="inline",
                options={"X": [[0.0]], "y": [0]},
            ),
            Train(
                name="trainer",
                framework="sklearn",
                model="sklearn.linear_model.LogisticRegression",
                data="ds",
            ),
        ]
    )
    plan = GraphCompiler().compile(pipe)
    provider = Auto(cheapest_gpu="A100", spot=True, dry_run=True, require_credentials=False)
    result = provider.execute(pipe, plan)
    assert [s.name for s in result.steps] == ["ds", "trainer"]
    assert all(s.status == "skipped" for s in result.steps)
    for step in result.steps:
        assert "would_run_on" in step.info
