"""End-to-end tests for the StandaloneProvider in in-process mode.

These tests exercise the in-process executor — the container-mode tests
live in ``tests/test_docker_runtime.py`` (FakeDockerEngine) and
``tests/test_docker_integration.py`` (real daemon, auto-skipped when
unavailable).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from ophelian import Data, Deploy, Eval, Pipeline, Standalone, Train, Tune
from ophelian.providers.standalone import FakeDockerEngine, StandaloneProvider

# Tiny linearly-separable toy dataset reused across tests.
_X: list[list[float]] = [[0.0, 0.1], [0.1, 0.2], [0.9, 1.0], [1.0, 0.9]]
_Y: list[int] = [0, 0, 1, 1]


def _data_node(name: str = "ds") -> Data:
    return Data(name=name, source="memory://toy", format="inline", options={"X": _X, "y": _Y})


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    return tmp_path / "workspace"


@pytest.fixture()
def engine() -> FakeDockerEngine:
    return FakeDockerEngine()


@pytest.fixture()
def provider(workspace: Path, engine: FakeDockerEngine) -> StandaloneProvider:
    return StandaloneProvider(
        local=True, workspace=workspace, docker_engine=engine, container=False
    )


def test_standalone_factory_returns_provider() -> None:
    provider = Standalone(local=True, container=False)
    assert isinstance(provider, StandaloneProvider)
    assert provider.mode == "inprocess"


def test_standalone_rejects_remote() -> None:
    with pytest.raises(ValueError, match="local=True"):
        Standalone(local=False, container=False)


def test_pipeline_trains_and_deploys_a_real_model(provider: StandaloneProvider) -> None:
    pipeline = Pipeline(
        [
            _data_node(),
            Train(
                name="trainer",
                framework="sklearn",
                model="sklearn.linear_model.LogisticRegression",
                data="ds",
            ),
            Eval(name="ev", model="trainer", data="ds", metrics=("accuracy",)),
            Deploy(name="serve", model="trainer", port=9000),
        ],
        name="end-to-end",
    )

    result = pipeline.run(env=provider)

    assert result.succeeded, [s.error for s in result.steps if s.error]
    assert [s.name for s in result.steps] == ["ds", "trainer", "ev", "serve"]

    # Train step produced a real on-disk model artifact.
    model_dir = Path(result.step("trainer").artifacts["model"])
    assert model_dir.is_dir()
    assert (model_dir / "ophelian.json").exists()
    descriptor = json.loads((model_dir / "ophelian.json").read_text())
    assert descriptor["framework"] == "sklearn"

    # Eval step computed accuracy from real predictions.
    ev_metrics = result.step("ev").metrics
    assert "accuracy" in ev_metrics
    assert ev_metrics["accuracy"] == 1.0  # toy dataset is linearly separable

    # Deploy step exposed a working FastAPI app — /predict returns real predictions.
    app = provider.apps["serve"]
    client = TestClient(app)
    health = client.get("/health").json()
    assert health["status"] == "ok"
    prediction = client.post("/predict", json={"inputs": _X}).json()
    assert prediction["prediction"] == _Y


def test_data_step_writes_manifest_and_dataset(provider: StandaloneProvider) -> None:
    pipeline = Pipeline([_data_node("only-data")], name="data-only")
    result = pipeline.run(env=provider)
    artifacts = result.step("only-data").artifacts
    manifest = json.loads(Path(artifacts["manifest"]).read_text())
    dataset = json.loads(Path(artifacts["dataset"]).read_text())
    assert manifest["format"] == "inline"
    assert dataset["X"] == _X and dataset["y"] == _Y


def test_pipeline_dry_run_returns_empty_result(
    provider: StandaloneProvider, engine: FakeDockerEngine
) -> None:
    pipeline = Pipeline([_data_node()], name="dry")
    result = pipeline.run(env=provider, dry_run=True)
    assert result.steps == []
    assert engine.actions == []


def test_train_failure_short_circuits_pipeline(workspace: Path) -> None:
    provider = StandaloneProvider(
        local=True, workspace=workspace, docker_engine=FakeDockerEngine(), container=False
    )
    pipeline = Pipeline(
        [
            _data_node(),
            # Reference a model class that does not exist — adapter will raise.
            Train(
                name="trainer", framework="sklearn", model="ThisEstimatorDoesNotExist", data="ds"
            ),
            Deploy(name="serve", model="trainer"),
        ],
        name="should-fail",
    )
    result = pipeline.run(env=provider)
    assert not result.succeeded
    statuses = [s.status for s in result.steps]
    assert statuses == ["success", "failed"]
    assert "ThisEstimatorDoesNotExist" in (result.step("trainer").error or "")


def test_tune_step_promotes_train_artifact(provider: StandaloneProvider) -> None:
    pipeline = Pipeline(
        [
            _data_node(),
            Train(
                name="trainer",
                framework="sklearn",
                model="sklearn.linear_model.LogisticRegression",
                data="ds",
            ),
            Tune(
                name="hpo",
                train="trainer",
                search_space={"C": [0.1, 1.0]},
                strategy="random",
                max_trials=2,
            ),
        ],
        name="tune-it",
    )
    result = pipeline.run(env=provider)
    assert result.succeeded
    best_dir = Path(result.step("hpo").artifacts["best_model"])
    assert best_dir.is_dir()
    assert (best_dir / "trial.json").exists()
