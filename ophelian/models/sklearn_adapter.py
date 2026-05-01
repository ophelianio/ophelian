"""Adapter for classic scikit-learn estimators."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from ophelian.models.base import ModelAdapter, register_adapter


@register_adapter
class SklearnAdapter(ModelAdapter):
    """Train, persist and serve any sklearn estimator referenced by class name."""

    framework: ClassVar[str] = "sklearn"

    def train(
        self,
        *,
        model: str,
        data: Any,
        hyperparameters: dict[str, Any],
        epochs: int | None,
        batch_size: int | None,
        resume_from: Path | None = None,
        checkpoint_dir: Path | None = None,
    ) -> Any:
        del epochs, batch_size, checkpoint_dir  # sklearn ignores these
        if resume_from is not None:
            import logging

            logging.getLogger(__name__).warning(
                "SklearnAdapter does not support mid-training resume — "
                "ignoring resume_from=%s and re-fitting from scratch.",
                resume_from,
            )
        estimator_cls = self._resolve_estimator(model)
        estimator = estimator_cls(**hyperparameters)
        if isinstance(data, dict) and "X" in data and "y" in data:
            estimator.fit(data["X"], data["y"])
        elif isinstance(data, tuple) and len(data) == 2:
            estimator.fit(*data)
        else:
            raise TypeError("SklearnAdapter expects data as dict(X=..., y=...) or tuple (X, y)")
        return estimator

    def save(self, model: Any, path: Path) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        target = path / "model.joblib"
        try:
            import joblib

            joblib.dump(model, target)
        except ImportError:
            import cloudpickle

            target = path / "model.pkl"
            target.write_bytes(cloudpickle.dumps(model))
        return target

    def load(self, path: Path) -> Any:
        joblib_path = path / "model.joblib"
        if joblib_path.exists():
            import joblib

            return joblib.load(joblib_path)
        pkl_path = path / "model.pkl"
        if pkl_path.exists():
            import cloudpickle

            return cloudpickle.loads(pkl_path.read_bytes())
        raise FileNotFoundError(f"No sklearn model artifact under {path}")

    def predict(self, model: Any, payload: Any) -> Any:
        return model.predict(payload).tolist()

    @staticmethod
    def _resolve_estimator(model: str) -> Any:
        """Resolve `model` to an sklearn estimator class.

        Accepts either a fully-qualified path (`sklearn.linear_model.LogisticRegression`)
        or a bare class name in which case sklearn's `all_estimators` is searched.
        """
        if "." in model:
            module_path, _, cls_name = model.rpartition(".")
            module = __import__(module_path, fromlist=[cls_name])
            return getattr(module, cls_name)
        from sklearn.utils import all_estimators

        for name, cls in all_estimators():
            if name == model:
                return cls
        raise ValueError(f"Unknown sklearn estimator: {model!r}")
