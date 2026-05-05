"""Information contract tests — pricing + Auto router subsystem (seed).

These tests pin the *information* the framework communicates to its
users: dataclass field sets, source-label vocabularies, log-line
formats, error-message structure, and explainer output. They are the
extension of the export-surface contract already pinned by
``tests/test_public_api_contract.py`` to the other three categories of
public information (return shapes, log contracts, error contracts).

Tests marked ``xfail(strict=True)`` pin information item #16 of the
v1.1.0 Track C roadmap will introduce. When T001-T004 land, those
tests start passing — the strict marker turns XPASS into a hard CI
failure, forcing whoever ships #16 to remove the marker in the same
PR. That is how the contract goes live atomically with the feature.
"""

from __future__ import annotations

import dataclasses
import logging
import re
from typing import Any

import ophelian.pricing
import pytest
from ophelian.envs.auto import Auto, AutoRouterError
from ophelian.pricing import (
    PriceQuote,
    RouterDecision,
    explain,
    static_quotes,
)

# --- Snapshots (update intentionally, in the same commit as the change) ---

EXPECTED_PRICEQUOTE_FIELDS = {
    "provider",
    "region",
    "instance",
    "gpu_family",
    "gpu_count",
    "hourly_usd",
    "spot",
    "source",
    "notes",
}

EXPECTED_ROUTERDECISION_FIELDS = {
    "quote",
    "considered",
    "data_quality",  # added in v1.1.0 #16 T001
}

EXPECTED_SOURCE_LABELS = {
    "static",
    "aws-spot-history",
    "azure-retail-api",
    "gcp-billing-catalog",
}

DATA_QUALITY_PATTERN = re.compile(r"^(live|static|unavailable|disabled|cached@\d+h)$")

AUTO_LOG_FORMAT = re.compile(
    r"^Auto router selected "
    r"(?P<provider>aws|gcp|azure)/(?P<region>[\w\-]+) "
    r"(?P<instance>\S+) "
    r"@ (?P<price>\d+\.\d{3}) USD/h "
    r"\((?P<billing>spot|on-demand)\)"
    r" \| data: (?P<data>(?:(?:aws|gcp|azure)=\S+\s?){1,3})"
    r"\| considered: (?P<n>\d+) quotes$"
)


# --- Return-shape contracts -------------------------------------------------


def test_pricequote_field_set_is_stable() -> None:
    """PriceQuote field set is part of the public contract."""
    actual = {f.name for f in dataclasses.fields(PriceQuote)}
    missing = EXPECTED_PRICEQUOTE_FIELDS - actual
    extra = actual - EXPECTED_PRICEQUOTE_FIELDS
    assert not missing, f"PriceQuote lost fields: {missing}"
    assert not extra, (
        f"PriceQuote gained fields without snapshot update: {extra}. "
        "If intentional, update EXPECTED_PRICEQUOTE_FIELDS in the same commit."
    )


def test_routerdecision_field_set_is_stable() -> None:
    actual = {f.name for f in dataclasses.fields(RouterDecision)}
    missing = EXPECTED_ROUTERDECISION_FIELDS - actual
    extra = actual - EXPECTED_ROUTERDECISION_FIELDS
    assert not missing, f"RouterDecision lost fields: {missing}"
    assert not extra, (
        f"RouterDecision gained fields without snapshot update: {extra}. "
        "If intentional, update EXPECTED_ROUTERDECISION_FIELDS in the same commit."
    )


# --- Vocabulary contracts ---------------------------------------------------


def test_pricequote_source_label_vocabulary() -> None:
    """Every PriceQuote.source value emitted anywhere in the framework
    must be in the published vocabulary. New labels require an explicit
    snapshot update so downstream log parsers and dashboards do not
    break silently."""
    quotes: list[PriceQuote] = []
    for family in ("T4", "L4", "V100", "A10G", "A100", "H100"):
        quotes.extend(static_quotes(family))
    actual_sources = {q.source for q in quotes}
    unknown = actual_sources - EXPECTED_SOURCE_LABELS
    assert not unknown, (
        f"PriceQuote emitted unknown source labels: {unknown}. "
        "If intentional, add them to EXPECTED_SOURCE_LABELS."
    )


def test_data_quality_value_vocabulary() -> None:
    """Every value in RouterDecision.data_quality must match the
    published vocabulary regex. Prevents silent introduction of new
    provenance categories that downstream tooling does not recognise."""
    decision = _make_dummy_decision()
    for provider, value in decision.data_quality.items():
        assert DATA_QUALITY_PATTERN.match(value), (
            f"data_quality[{provider!r}] = {value!r} does not match "
            f"vocabulary pattern {DATA_QUALITY_PATTERN.pattern}"
        )


# --- Log-contract tests -----------------------------------------------------


def test_auto_router_log_format_is_stable(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Auto router log line is part of the public contract — on-call
    runbooks quote it and CI dashboards parse it. Any change must be
    explicit (update AUTO_LOG_FORMAT regex in the same commit)."""
    # The "ophelian" parent logger has propagate=False (configured by
    # ``ophelian.observability.configure_logging``), so caplog (attached
    # to root) never sees its records. Re-enable propagation just for
    # this test; monkeypatch undoes it on teardown.
    monkeypatch.setattr(logging.getLogger("ophelian"), "propagate", True)
    caplog.set_level(logging.INFO, logger="ophelian.envs.auto")
    Auto(
        cheapest_gpu="A100",
        regions=["us-east-1", "us-central1", "eastus"],
        dry_run=True,
        require_credentials=False,
    )
    selected = [r for r in caplog.records if r.message.startswith("Auto router selected")]
    assert len(selected) == 1, f"expected exactly one selection log, got {len(selected)}"
    rendered = selected[0].getMessage()
    assert AUTO_LOG_FORMAT.match(rendered), (
        f"Auto router log line does not match contract.\n"
        f"  got:  {rendered!r}\n"
        f"  want: {AUTO_LOG_FORMAT.pattern}"
    )


# --- Error-contract tests ---------------------------------------------------


def test_autorouter_no_quotes_error_contains_actionable_fields() -> None:
    """The 'no quotes available' branch must name the GPU family, the
    providers that were considered, and the regions filter so the user
    can see at a glance what to widen."""
    with pytest.raises(AutoRouterError) as exc:
        Auto(
            cheapest_gpu="DOES_NOT_EXIST",
            regions=["us-east-1"],
            dry_run=True,
            require_credentials=False,
        )
    msg = str(exc.value)
    assert "DOES_NOT_EXIST" in msg, "missing GPU family in error message"
    assert "providers=" in msg, "missing providers field in error message"
    assert "regions=" in msg, "missing regions field in error message"
    assert "Pricing table last reviewed" in msg, "missing freshness pointer"


def test_autorouter_no_credentials_error_names_picked_provider() -> None:
    """The 'no local credentials' branch must name the provider the
    router would have used so the user knows which cloud SDK to set up
    or which provider to exclude."""
    with pytest.raises(AutoRouterError) as exc:
        Auto(
            cheapest_gpu="A100",
            regions=["us-east-1", "us-central1", "eastus"],
            dry_run=False,
            require_credentials=True,
        )
    msg = str(exc.value)
    assert re.search(r"picked '(aws|gcp|azure)'", msg), (
        f"error message does not name the picked provider: {msg!r}"
    )
    assert "providers=[...]" in msg, "missing actionable hint about providers="


def test_autorouter_require_live_error_names_provider_and_actual_source() -> None:
    """The 'require_live failed' branch must name the provider that
    failed the live requirement and the actual provenance value
    received, so the user knows whether to retry, widen, or accept."""
    with pytest.raises(AutoRouterError) as exc:
        Auto(
            cheapest_gpu="A100",
            regions=["us-east-1", "us-central1", "eastus"],
            dry_run=True,
            require_credentials=False,
            require_live=["azure"],
            allow_live=False,  # force azure to be 'static', not 'live'
        )
    msg = str(exc.value)
    assert "require_live" in msg, "missing kwarg name in error"
    assert "azure" in msg, "missing failing provider in error"
    assert re.search(r"static|unavailable|cached@", msg), (
        f"error message does not surface the actual provenance: {msg!r}"
    )


# --- Explainer-contract test ------------------------------------------------


def test_explain_output_format() -> None:
    """`explain(decision)` is the human-facing summary used by the CLI
    `--explain` flag and quoted in issues/blog posts. Format is part
    of the public contract."""
    quote = PriceQuote(
        provider="azure",
        region="eastus",
        instance="Standard_NC24ads_A100_v4",
        gpu_family="A100",
        gpu_count=1,
        hourly_usd=0.735,
        spot=True,
        source="azure-retail-api",
    )
    runner_up = PriceQuote(
        provider="gcp",
        region="us-central1",
        instance="a2-highgpu-1g",
        gpu_family="A100",
        gpu_count=1,
        hourly_usd=1.400,
        spot=True,
        source="static",
    )
    decision = RouterDecision(quote=quote, considered=[quote, runner_up])
    out = explain(decision)
    assert out.startswith("Cheapest A100: azure/eastus on Standard_NC24ads_A100_v4")
    assert "0.735 USD/hour" in out
    assert "spot" in out
    assert "source=azure-retail-api" in out
    assert "Runners-up:" in out
    assert "gcp/us-central1" in out


# --- fetch_live_with_meta contract (xfail until T001) -----------------------


def test_fetch_live_with_meta_keys_match_providers_consulted() -> None:
    """The meta dict keys must exactly equal the set of providers
    consulted, no more and no less. Prevents silent gaps where a
    provider is queried but its provenance is not reported (or vice
    versa)."""
    from ophelian.pricing.live import fetch_live_with_meta  # type: ignore[attr-defined]

    quotes, meta = fetch_live_with_meta(
        "A100",
        providers=["aws", "azure"],
        regions=["us-east-1", "eastus"],
    )
    assert isinstance(quotes, list)
    assert isinstance(meta, dict)
    assert set(meta.keys()) == {"aws", "azure"}, (
        f"meta keys must equal providers consulted; got {set(meta.keys())}"
    )
    for value in meta.values():
        assert DATA_QUALITY_PATTERN.match(value), (
            f"meta value {value!r} not in vocabulary"
        )


# --- helpers ----------------------------------------------------------------


def _make_dummy_decision() -> RouterDecision:
    """Build a minimal RouterDecision for vocabulary assertions.

    Once T001 lands, RouterDecision will accept a `data_quality` kwarg.
    Until then, this helper raises a TypeError that the xfail wrapper
    will catch and report as XFAIL.
    """
    q = PriceQuote(
        provider="azure",
        region="eastus",
        instance="Standard_NC24ads_A100_v4",
        gpu_family="A100",
        gpu_count=1,
        hourly_usd=0.735,
        spot=True,
        source="azure-retail-api",
    )
    return RouterDecision(  # type: ignore[call-arg]
        quote=q,
        considered=[q],
        data_quality={"aws": "static", "gcp": "unavailable", "azure": "live"},
    )


# Ensure the pricing module re-exports what we contract-test against,
# so a typo in our import does not silently degrade these tests.
def test_imports_resolve_to_pricing_module() -> None:
    assert PriceQuote is ophelian.pricing.PriceQuote
    assert RouterDecision is ophelian.pricing.RouterDecision
    assert explain is ophelian.pricing.explain
    assert static_quotes is ophelian.pricing.static_quotes


def _placeholder_for_unused_imports() -> Any:  # pragma: no cover
    return AutoRouterError  # silence "imported but unused" if Auto path skipped
