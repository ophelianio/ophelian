"""Tests for the adapter registry and the sklearn adapter (only one without heavy deps)."""

from __future__ import annotations

from pathlib import Path

import pytest
from ophelian.models import ModelAdapter, register_adapter, registry
from ophelian.models.sklearn_adapter import SklearnAdapter


def test_registry_lists_builtins() -> None:
    names = registry.names()
    for expected in ("sklearn", "xgboost", "pytorch", "huggingface"):
        assert expected in names


def test_registry_get_unknown_raises() -> None:
    with pytest.raises(KeyError):
        registry.get("not-a-framework")


def test_register_decorator_adds_to_registry() -> None:
    @register_adapter
    class _Toy(ModelAdapter):
        framework = "toy"

        def train(self, **_kwargs):  # type: ignore[no-untyped-def]
            return object()

        def save(self, model, path):  # type: ignore[no-untyped-def]
            return Path(path)

        def load(self, path):  # type: ignore[no-untyped-def]
            return object()

        def predict(self, model, payload):  # type: ignore[no-untyped-def]
            return payload

    assert registry.get("toy") is _Toy


def test_register_rejects_blank_framework() -> None:
    with pytest.raises(ValueError):

        class _Bad(ModelAdapter):
            framework = ""

            def train(self, **_kwargs):  # type: ignore[no-untyped-def]
                return None

            def save(self, model, path):  # type: ignore[no-untyped-def]
                return Path(path)

            def load(self, path):  # type: ignore[no-untyped-def]
                return None

            def predict(self, model, payload):  # type: ignore[no-untyped-def]
                return payload

        register_adapter(_Bad)


def test_sklearn_adapter_train_save_load_predict_roundtrip(tmp_path: Path) -> None:
    pytest.importorskip("sklearn")

    adapter = SklearnAdapter()
    data = {
        "X": [[0.0], [1.0], [2.0], [3.0]],
        "y": [0, 0, 1, 1],
    }
    model = adapter.train(
        model="sklearn.linear_model.LogisticRegression",
        data=data,
        hyperparameters={"max_iter": 200},
        epochs=None,
        batch_size=None,
    )
    target = tmp_path / "model_dir"
    saved_path = adapter.save(model, target)
    assert saved_path.exists()

    loaded = adapter.load(target)
    predictions = adapter.predict(loaded, [[0.0], [3.0]])
    assert predictions == [0, 1]


def test_xgboost_adapter_train_save_load_predict_roundtrip(tmp_path: Path) -> None:
    pytest.importorskip("xgboost")
    pytest.importorskip("sklearn")
    from ophelian.models.xgboost_adapter import XGBoostAdapter

    adapter = XGBoostAdapter()
    # Slightly larger toy dataset so a small ensemble can clearly separate classes.
    x = [
        [0.0, 0.0],
        [0.1, 0.0],
        [0.0, 0.1],
        [0.1, 0.1],
        [1.0, 1.0],
        [0.9, 1.0],
        [1.0, 0.9],
        [0.9, 0.9],
    ]
    y = [0, 0, 0, 0, 1, 1, 1, 1]
    data = {"X": x, "y": y}
    model = adapter.train(
        model="XGBClassifier",
        data=data,
        hyperparameters={"n_estimators": 30, "max_depth": 3, "verbosity": 0},
        epochs=None,
        batch_size=None,
    )
    target = tmp_path / "xgb"
    adapter.save(model, target)

    loaded = adapter.load(target)
    predictions = adapter.predict(loaded, x)
    assert predictions == y


def test_pytorch_adapter_train_save_load_predict_roundtrip(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from ophelian.models.pytorch_adapter import PyTorchAdapter

    adapter = PyTorchAdapter()
    data = {
        "X": [[0.0], [1.0], [2.0], [3.0]],
        "y": [0.0, 1.0, 2.0, 3.0],
    }
    model = adapter.train(
        model="torch.nn.Linear",
        data=data,
        hyperparameters={"in_features": 1, "out_features": 1},
        epochs=200,
        batch_size=None,
    )
    target = tmp_path / "torch_model"
    saved = adapter.save(model, target)
    assert saved.exists()

    loaded = adapter.load(target)
    predictions = adapter.predict(loaded, [[5.0]])
    assert isinstance(predictions, list)
    assert len(predictions) == 1
    # After fitting an identity-ish line the prediction should be roughly 5.
    assert abs(predictions[0][0] - 5.0) < 1.5
