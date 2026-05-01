"""Tests for the GCP env, provider and driver wiring."""

from __future__ import annotations

from pathlib import Path

import pytest
from ophelian import GCP, Data, Eval, Pipeline, Train
from ophelian.envs.gcp import KNOWN_REGIONS, GCPConfig
from ophelian.providers.gcp import GCPProvider
from ophelian.providers.gcp_drivers import LocalGCPDriver
from ophelian.stores.gcs import GCSArtifactStore

from tests._fake_gcs import FakeGCSClient

_X = [[0.0, 0.1], [0.1, 0.2], [0.9, 1.0], [1.0, 0.9]]
_Y = [0, 0, 1, 1]


def _make_pipeline(name: str = "gcp-test") -> Pipeline:
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
def store(tmp_path: Path) -> GCSArtifactStore:
    return GCSArtifactStore(
        bucket="ophelian-gcp-test",
        prefix="runs",
        client=FakeGCSClient(),
        cache_dir=tmp_path / "cache",
        ensure_bucket=False,
    )


# ---------------------------------------------------------------------------
# GCPConfig validation
# ---------------------------------------------------------------------------


def test_gcp_config_validates_region() -> None:
    with pytest.raises(ValueError, match="region"):
        GCPConfig(project="p", region="not-a-real-region")


def test_gcp_config_accepts_lookalike_region() -> None:
    cfg = GCPConfig(project="p", region="europe-north1")
    assert cfg.region == "europe-north1"
    assert "us-central1" in KNOWN_REGIONS


def test_gcp_config_validates_machine_type() -> None:
    with pytest.raises(ValueError, match="machine_type"):
        GCPConfig(project="p", region="us-central1", machine_type="not-a-machine!")


def test_gcp_config_rejects_non_nvidia_gpu() -> None:
    with pytest.raises(ValueError, match="gpu_type"):
        GCPConfig(
            project="p", region="us-central1",
            machine_type="n1-standard-4",
            gpu_type="amd-mi300", gpu_count=1,
        )


def test_gcp_config_requires_gpu_type_when_count_set() -> None:
    with pytest.raises(ValueError, match="gpu_count"):
        GCPConfig(
            project="p", region="us-central1",
            machine_type="n1-standard-4",
            gpu_type=None, gpu_count=1,
        )


def test_gcp_config_defaults_zone_to_a() -> None:
    cfg = GCPConfig(project="p", region="us-central1")
    assert cfg.effective_zone == "us-central1-a"


def test_gcp_config_with_resume_returns_copy() -> None:
    cfg = GCPConfig(project="p", region="us-central1")
    new = cfg.with_resume("run-abc")
    assert new.resume_run_id == "run-abc"
    assert cfg.resume_run_id is None


def test_gcp_config_gke_requires_cluster() -> None:
    with pytest.raises(ValueError, match="gke_cluster"):
        GCPConfig(project="p", region="us-central1", gcp_backend="gke")


def test_gcp_config_gke_rejects_spot() -> None:
    with pytest.raises(ValueError, match="Preemptible"):
        GCPConfig(
            project="p", region="us-central1",
            gcp_backend="gke", gke_cluster="ml",
            spot=True,
        )


# ---------------------------------------------------------------------------
# GCP factory
# ---------------------------------------------------------------------------


def test_gcp_factory_accepts_preemptible_alias(store: GCSArtifactStore) -> None:
    provider = GCP(
        project="my-proj",
        region="us-central1",
        machine_type="n1-standard-4",
        preemptible=True,
        artifact_bucket=store.bucket,
        driver=LocalGCPDriver(),
        store=store,
    )
    assert provider.config.spot is True


def test_gcp_factory_rejects_conflicting_spot_and_preemptible(
    store: GCSArtifactStore,
) -> None:
    with pytest.raises(ValueError, match="conflicting"):
        GCP(
            project="my-proj",
            region="us-central1",
            spot=True,
            preemptible=False,
            artifact_bucket=store.bucket,
            driver=LocalGCPDriver(),
            store=store,
        )


def test_gcp_factory_returns_provider(store: GCSArtifactStore) -> None:
    provider = GCP(
        project="my-proj",
        region="us-central1",
        machine_type="n1-standard-4",
        artifact_bucket=store.bucket,
        driver=LocalGCPDriver(),
        store=store,
    )
    assert isinstance(provider, GCPProvider)
    assert provider.name == "gcp"


# ---------------------------------------------------------------------------
# Pipeline execution through LocalGCPDriver
# ---------------------------------------------------------------------------


def test_provider_runs_pipeline_with_local_driver(store: GCSArtifactStore) -> None:
    driver = LocalGCPDriver()
    provider = GCPProvider(
        config=GCPConfig(
            project="my-proj",
            region="us-central1",
            artifact_bucket=store.bucket,
        ),
        driver=driver,
        store=store,
    )

    result = _make_pipeline().run(env=provider)

    assert result.succeeded, [s.error for s in result.steps if s.error]
    assert [s.name for s in result.steps] == ["ds", "trainer", "ev"]
    assert result.step("ev").metrics["accuracy"] == 1.0
    assert driver.executed == ["ds", "trainer", "ev"]
    assert result.step("trainer").artifacts["model"].startswith("gs://")
    assert provider.run_id.startswith("run-")


def test_provider_persists_artifacts_to_gcs(store: GCSArtifactStore) -> None:
    provider = GCPProvider(
        config=GCPConfig(
            project="my-proj",
            region="us-central1",
            artifact_bucket=store.bucket,
        ),
        driver=LocalGCPDriver(),
        store=store,
    )
    _make_pipeline().run(env=provider)

    keys = list(store.list(""))
    assert any("trainer/artifacts/model" in k for k in keys), keys
    assert any("ev/artifacts" in k for k in keys), keys


def test_provider_failure_short_circuits_remaining_steps(store: GCSArtifactStore) -> None:
    driver = LocalGCPDriver(fail_on=("trainer",))
    provider = GCPProvider(
        config=GCPConfig(
            project="my-proj",
            region="us-central1",
            artifact_bucket=store.bucket,
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
