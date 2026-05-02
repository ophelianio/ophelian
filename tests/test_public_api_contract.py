"""Public API contract tests.

These tests pin the user-visible export surface of the framework so a
silent rename or accidental drop can never ship to PyPI undetected.
The expected snapshots intentionally live next to the assertions —
when you genuinely add or remove a public symbol you update the
snapshot in the same commit and that diff stands as documentation of
a public-API change.
"""

from __future__ import annotations

import dataclasses
import inspect
from collections.abc import Callable
from typing import Any

import ophelian
import ophelian.data
import ophelian.pricing
import pytest

# Snapshots — bump in lockstep with intentional API changes.
EXPECTED_TOPLEVEL_ALL = {
    "AWS",
    "GCP",
    "Auto",
    "Azure",
    "Data",
    "Deploy",
    "Eval",
    "Pipeline",
    "Standalone",
    "Train",
    "Tune",
    "__version__",
}

EXPECTED_PRICING_ALL = {
    "STATIC_PRICES",
    "STATIC_PRICES_LAST_REVIEW",
    "PriceQuote",
    "RouterDecision",
    "explain",
    "fetch_live",
    "lookup_cheapest",
    "static_quotes",
}

EXPECTED_DATA_ALL = {"materialize"}


def test_toplevel_all_matches_snapshot() -> None:
    actual = set(ophelian.__all__)
    missing = EXPECTED_TOPLEVEL_ALL - actual
    extra = actual - EXPECTED_TOPLEVEL_ALL
    assert not missing, f"Public symbols disappeared from ophelian.__all__: {missing}"
    assert not extra, (
        f"New public symbols added without updating the snapshot: {extra}. "
        "If intentional, update EXPECTED_TOPLEVEL_ALL in the same commit."
    )


def test_pricing_all_matches_snapshot() -> None:
    actual = set(ophelian.pricing.__all__)
    missing = EXPECTED_PRICING_ALL - actual
    extra = actual - EXPECTED_PRICING_ALL
    assert not missing, f"Public symbols disappeared from ophelian.pricing.__all__: {missing}"
    assert not extra, (
        f"New public symbols added without updating the snapshot: {extra}. "
        "If intentional, update EXPECTED_PRICING_ALL in the same commit."
    )


def test_data_all_matches_snapshot() -> None:
    actual = set(ophelian.data.__all__)
    assert actual == EXPECTED_DATA_ALL, (
        f"ophelian.data public surface drifted. Expected {EXPECTED_DATA_ALL}, got {actual}."
    )


@pytest.mark.parametrize(
    "module, name",
    [(ophelian, n) for n in EXPECTED_TOPLEVEL_ALL]
    + [(ophelian.pricing, n) for n in EXPECTED_PRICING_ALL]
    + [(ophelian.data, n) for n in EXPECTED_DATA_ALL],
)
def test_every_public_symbol_resolves(module: Any, name: str) -> None:
    """Each name in `__all__` must actually be importable from the module.
    Catches ``__all__`` listing names that were removed from the body."""
    assert hasattr(module, name), f"{module.__name__}.{name} listed in __all__ but missing"


def test_node_factories_have_expected_signature() -> None:
    """The DSL constructors are the most user-visible API. Pin their
    public parameters so a rename causes a loud failure."""
    pipeline_params = set(inspect.signature(ophelian.Pipeline).parameters)
    assert {"name", "steps"}.issubset(pipeline_params), (
        f"Pipeline must accept name= and steps=, got {pipeline_params}"
    )

    data_params = set(inspect.signature(ophelian.Data).parameters)
    assert {"name", "source", "format"}.issubset(data_params), (
        f"Data must accept name=, source=, format=, got {data_params}"
    )

    train_params = set(inspect.signature(ophelian.Train).parameters)
    assert {"name", "framework", "model", "data"}.issubset(train_params), (
        f"Train must accept name=, framework=, model=, data=, got {train_params}"
    )


def test_lookup_cheapest_signature_is_stable() -> None:
    """``lookup_cheapest`` is what every consumer of the auto-router
    calls — its keyword arguments are part of the public contract."""
    sig = inspect.signature(ophelian.pricing.lookup_cheapest)
    params = sig.parameters
    assert "gpu_family" in params, "first positional arg must stay `gpu_family`"
    for kw in ("providers", "regions", "spot", "allow_live", "cache_ttl_seconds"):
        assert kw in params, f"lookup_cheapest dropped keyword argument: {kw}"


def test_pricequote_fields_are_stable() -> None:
    """``PriceQuote`` is a public dataclass. Renaming a field would
    silently break downstream code that introspects it."""
    assert dataclasses.is_dataclass(ophelian.pricing.PriceQuote), (
        "PriceQuote stopped being a dataclass — that breaks downstream "
        "code that uses dataclasses.asdict() or .__dataclass_fields__"
    )
    fields = {f.name for f in dataclasses.fields(ophelian.pricing.PriceQuote)}
    expected = {
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
    missing = expected - fields
    assert not missing, f"PriceQuote dropped public fields: {missing}"


def test_materialize_is_callable() -> None:
    """``ophelian.data.materialize`` must be callable; the dispatcher
    refactor moved bodies but it must still work as a function."""
    assert isinstance(ophelian.data.materialize, Callable)  # type: ignore[arg-type]
