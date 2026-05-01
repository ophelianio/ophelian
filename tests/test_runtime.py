"""Tests for the FastAPI inference runtime."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from ophelian.models.sklearn_adapter import SklearnAdapter
from ophelian.runtime import build_app
from ophelian.runtime.fastapi_runtime import app_from_env


@pytest.fixture()
def trained_sklearn_model(tmp_path: Path) -> Path:
    pytest.importorskip("sklearn")
    adapter = SklearnAdapter()
    model = adapter.train(
        model="sklearn.linear_model.LogisticRegression",
        data={"X": [[0.0], [1.0], [2.0], [3.0]], "y": [0, 0, 1, 1]},
        hyperparameters={"max_iter": 200},
        epochs=None,
        batch_size=None,
    )
    target = tmp_path / "model_dir"
    adapter.save(model, target)
    return target


def test_health_endpoint(trained_sklearn_model: Path) -> None:
    app = build_app(framework="sklearn", model_path=trained_sklearn_model)
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["framework"] == "sklearn"


def test_predict_endpoint(trained_sklearn_model: Path) -> None:
    app = build_app(framework="sklearn", model_path=trained_sklearn_model)
    client = TestClient(app)
    response = client.post("/predict", json={"inputs": [[0.0], [3.0]]})
    assert response.status_code == 200
    body = response.json()
    assert "prediction" in body
    assert len(body["prediction"]) == 2


def test_app_from_env_factory_uses_environment(
    trained_sklearn_model: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The container deploy path uses ``--factory app_from_env``; this test
    pins that contract so a refactor can't silently break the Docker
    deploy command."""
    monkeypatch.setenv("OPHELIAN_FRAMEWORK", "sklearn")
    monkeypatch.setenv("OPHELIAN_MODEL_PATH", str(trained_sklearn_model))
    app = app_from_env()
    client = TestClient(app)
    assert client.get("/health").status_code == 200
    response = client.post("/predict", json={"inputs": [[0.0], [3.0]]})
    assert response.status_code == 200
    assert len(response.json()["prediction"]) == 2


def test_app_from_env_factory_requires_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPHELIAN_FRAMEWORK", raising=False)
    monkeypatch.delenv("OPHELIAN_MODEL_PATH", raising=False)
    with pytest.raises(RuntimeError, match="OPHELIAN_FRAMEWORK"):
        app_from_env()
