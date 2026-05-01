"""Tests for the Azure env, provider and driver wiring."""

from __future__ import annotations

from pathlib import Path

import pytest
from ophelian import Azure, Data, Eval, Pipeline, Train
from ophelian.envs.azure import KNOWN_REGIONS, AzureConfig
from ophelian.providers.azure import AzureProvider
from ophelian.providers.azure_drivers import LocalAzureDriver
from ophelian.stores.azure_blob import AzureBlobArtifactStore

from tests._fake_azure import FakeBlobServiceClient

_X = [[0.0, 0.1], [0.1, 0.2], [0.9, 1.0], [1.0, 0.9]]
_Y = [0, 0, 1, 1]
_SUB = "11111111-2222-3333-4444-555555555555"


def _make_pipeline(name: str = "azure-test") -> Pipeline:
    return Pipeline(
        [
            Data(name="ds", source="memory://toy", format="inline", options={"X": _X, "y": _Y}),
            Train(
                name="trainer",
                framework="sklearn",
                model="sklearn.linear_model.LogisticRegression",
                data="ds",
                hyperparameters={"max_iter": 200},
            ),
            Eval(name="ev", model="trainer", data="ds", metrics=("accuracy",)),
        ],
        name=name,
    )


@pytest.fixture()
def store(tmp_path: Path) -> AzureBlobArtifactStore:
    return AzureBlobArtifactStore(
        account="ophelianacct",
        container="ophelian-azure-test",
        prefix="runs",
        client=FakeBlobServiceClient(),
        cache_dir=tmp_path / "cache",
        ensure_container=False,
    )


# ---------------------------------------------------------------------------
# AzureConfig validation
# ---------------------------------------------------------------------------


def test_azure_config_validates_subscription_id() -> None:
    with pytest.raises(ValueError, match="subscription_id"):
        AzureConfig(
            subscription_id="not-a-guid",
            resource_group="rg", location="eastus",
        )


def test_azure_config_validates_location() -> None:
    cfg = AzureConfig(subscription_id=_SUB, resource_group="rg", location="eastus")
    assert cfg.location == "eastus"
    assert "eastus" in KNOWN_REGIONS


def test_azure_config_rejects_bad_vm_size() -> None:
    with pytest.raises(ValueError, match="vm_size"):
        AzureConfig(
            subscription_id=_SUB, resource_group="rg",
            location="eastus", vm_size="d4s-v5-not-prefixed",
        )


def test_azure_config_aks_requires_cluster() -> None:
    with pytest.raises(ValueError, match="aks_cluster"):
        AzureConfig(
            subscription_id=_SUB, resource_group="rg",
            location="eastus", azure_backend="aks",
        )


def test_azure_config_aks_rejects_spot() -> None:
    with pytest.raises(ValueError, match="Spot"):
        AzureConfig(
            subscription_id=_SUB, resource_group="rg",
            location="eastus", azure_backend="aks",
            aks_cluster="ml", spot=True,
        )


def test_azure_config_with_resume_returns_copy() -> None:
    cfg = AzureConfig(subscription_id=_SUB, resource_group="rg", location="eastus")
    new = cfg.with_resume("run-xyz")
    assert new.resume_run_id == "run-xyz"
    assert cfg.resume_run_id is None


def test_azure_config_detects_gpu_vm_family() -> None:
    cfg = AzureConfig(
        subscription_id=_SUB, resource_group="rg",
        location="eastus", vm_size="Standard_NC6s_v3",
    )
    assert cfg.is_gpu_instance is True


# ---------------------------------------------------------------------------
# Azure factory
# ---------------------------------------------------------------------------


def test_azure_factory_accepts_region_alias(store: AzureBlobArtifactStore) -> None:
    provider = Azure(
        subscription_id=_SUB,
        resource_group="rg",
        region="eastus",
        vm_size="Standard_D4s_v5",
        artifact_account=store.account,
        artifact_container=store.container,
        driver=LocalAzureDriver(),
        store=store,
    )
    assert provider.config.location == "eastus"


def test_azure_factory_requires_location_or_region() -> None:
    with pytest.raises(TypeError, match="location"):
        Azure(subscription_id=_SUB, resource_group="rg")  # type: ignore[call-arg]


def test_azure_factory_rejects_conflicting_location_and_region() -> None:
    with pytest.raises(ValueError, match="conflicting"):
        Azure(
            subscription_id=_SUB,
            resource_group="rg",
            location="eastus",
            region="westus2",
        )


def test_azure_factory_returns_provider(store: AzureBlobArtifactStore) -> None:
    provider = Azure(
        subscription_id=_SUB,
        resource_group="rg",
        location="eastus",
        vm_size="Standard_D4s_v5",
        artifact_account=store.account,
        artifact_container=store.container,
        driver=LocalAzureDriver(),
        store=store,
    )
    assert isinstance(provider, AzureProvider)
    assert provider.name == "azure"


# ---------------------------------------------------------------------------
# Pipeline execution through LocalAzureDriver
# ---------------------------------------------------------------------------


def test_provider_runs_pipeline_with_local_driver(store: AzureBlobArtifactStore) -> None:
    driver = LocalAzureDriver()
    provider = AzureProvider(
        config=AzureConfig(
            subscription_id=_SUB,
            resource_group="rg",
            location="eastus",
            artifact_account=store.account,
            artifact_container=store.container,
        ),
        driver=driver,
        store=store,
    )

    result = _make_pipeline().run(env=provider)

    assert result.succeeded, [s.error for s in result.steps if s.error]
    assert [s.name for s in result.steps] == ["ds", "trainer", "ev"]
    assert result.step("ev").metrics["accuracy"] == 1.0
    assert driver.executed == ["ds", "trainer", "ev"]
    assert result.step("trainer").artifacts["model"].startswith("az://")
    assert provider.run_id.startswith("run-")


def test_provider_persists_artifacts_to_blob(store: AzureBlobArtifactStore) -> None:
    provider = AzureProvider(
        config=AzureConfig(
            subscription_id=_SUB,
            resource_group="rg",
            location="eastus",
            artifact_account=store.account,
            artifact_container=store.container,
        ),
        driver=LocalAzureDriver(),
        store=store,
    )
    _make_pipeline().run(env=provider)

    keys = list(store.list(""))
    assert any("trainer/artifacts/model" in k for k in keys), keys
    assert any("ev/artifacts" in k for k in keys), keys


def test_provider_failure_short_circuits_remaining_steps(
    store: AzureBlobArtifactStore,
) -> None:
    driver = LocalAzureDriver(fail_on=("trainer",))
    provider = AzureProvider(
        config=AzureConfig(
            subscription_id=_SUB,
            resource_group="rg",
            location="eastus",
            artifact_account=store.account,
            artifact_container=store.container,
        ),
        driver=driver,
        store=store,
    )
    result = _make_pipeline().run(env=provider)

    assert not result.succeeded
    statuses = {step.name: step.status for step in result.steps}
    assert statuses["ds"] == "success"
    assert statuses["trainer"] == "failed"
    assert "ev" not in statuses
    assert driver.executed == ["ds"]
