"""Adapter for XGBoost regressors and classifiers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

from ophelian.models.base import ModelAdapter, register_adapter


@register_adapter
class XGBoostAdapter(ModelAdapter):
    framework: ClassVar[str] = "xgboost"

    def train(
        self,
        *,
        model: str,
        data: Any,
        hyperparameters: dict[str, Any],
        epochs: int | None,
        batch_size: int | None,
    ) -> Any:
        del batch_size
        import xgboost as xgb

        params = dict(hyperparameters)
        if epochs is not None and "n_estimators" not in params:
            params["n_estimators"] = epochs
        cls_name = model or "XGBClassifier"
        estimator_cls = getattr(xgb, cls_name)
        estimator = estimator_cls(**params)
        if isinstance(data, dict) and "X" in data and "y" in data:
            estimator.fit(data["X"], data["y"])
        elif isinstance(data, tuple) and len(data) == 2:
            estimator.fit(*data)
        else:
            raise TypeError("XGBoostAdapter expects data as dict(X=..., y=...) or tuple (X, y)")
        return estimator

    def save(self, model: Any, path: Path) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        target = path / "model.json"
        model.save_model(target)
        # Remember the estimator class so `load` returns the same interface
        # (`predict` semantics differ between XGBClassifier and Booster).
        meta = {"class": type(model).__name__}
        (path / "xgboost.json").write_text(json.dumps(meta))
        return target

    def load(self, path: Path) -> Any:
        import xgboost as xgb

        target = path / "model.json"
        meta_path = path / "xgboost.json"
        cls_name = "XGBClassifier"
        if meta_path.exists():
            cls_name = json.loads(meta_path.read_text()).get("class", cls_name)
        estimator_cls = getattr(xgb, cls_name, None)
        if estimator_cls is None:  # pragma: no cover - defensive
            booster = xgb.Booster()
            booster.load_model(str(target))
            return booster
        estimator = estimator_cls()
        estimator.load_model(str(target))
        return estimator

    def predict(self, model: Any, payload: Any) -> Any:
        import xgboost as xgb

        if isinstance(model, xgb.Booster):
            return model.predict(xgb.DMatrix(payload)).tolist()
        return model.predict(payload).tolist()
