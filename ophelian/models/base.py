"""Adapter base class and discovery registry.

Adapters know how to take a `Train` node, produce a serialised model artifact,
load it back into memory, and run predictions. Built-in adapters live in this
package; third parties can register their own via the
`ophelian.adapters` entry-point group declared in `pyproject.toml`.
"""

from __future__ import annotations

import importlib
import importlib.metadata
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar


class ModelAdapter(ABC):
    """Contract every framework adapter must implement."""

    framework: ClassVar[str] = ""

    @abstractmethod
    def train(
        self,
        *,
        model: str,
        data: Any,
        hyperparameters: dict[str, Any],
        epochs: int | None,
        batch_size: int | None,
    ) -> Any:
        """Train and return a model object the adapter can later serialise."""

    @abstractmethod
    def save(self, model: Any, path: Path) -> Path:
        """Persist `model` under `path` and return the canonical artifact location."""

    @abstractmethod
    def load(self, path: Path) -> Any:
        """Re-hydrate a model previously persisted with `save`."""

    @abstractmethod
    def predict(self, model: Any, payload: Any) -> Any:
        """Run a single inference."""


class _Registry:
    """Lazy registry for adapter classes — keyed by framework name."""

    def __init__(self) -> None:
        self._adapters: dict[str, type[ModelAdapter]] = {}
        self._loaded_entrypoints = False

    def register(self, adapter_cls: type[ModelAdapter]) -> type[ModelAdapter]:
        if not adapter_cls.framework:
            raise ValueError(
                f"{adapter_cls.__name__} must declare a non-empty `framework` class attribute"
            )
        self._adapters[adapter_cls.framework] = adapter_cls
        return adapter_cls

    def get(self, framework: str) -> type[ModelAdapter]:
        self._ensure_entrypoints_loaded()
        try:
            return self._adapters[framework]
        except KeyError as exc:
            raise KeyError(
                f"No adapter registered for framework {framework!r}. "
                f"Available: {sorted(self._adapters)}"
            ) from exc

    def names(self) -> list[str]:
        self._ensure_entrypoints_loaded()
        return sorted(self._adapters)

    def _ensure_entrypoints_loaded(self) -> None:
        if self._loaded_entrypoints:
            return
        self._loaded_entrypoints = True
        try:
            entry_points = importlib.metadata.entry_points(group="ophelian.adapters")
        except TypeError:  # pragma: no cover — older importlib API
            all_eps = importlib.metadata.entry_points()
            entry_points = all_eps.select(group="ophelian.adapters")
        for ep in entry_points:
            try:
                adapter_cls = ep.load()
            except Exception:
                continue
            if isinstance(adapter_cls, type) and issubclass(adapter_cls, ModelAdapter):
                self.register(adapter_cls)

    def reset(self) -> None:
        """Test helper — clear the registry and re-trigger entry-point discovery."""
        self._adapters.clear()
        self._loaded_entrypoints = False


registry = _Registry()


def register_adapter(adapter_cls: type[ModelAdapter]) -> type[ModelAdapter]:
    """Decorator that registers a `ModelAdapter` subclass."""
    return registry.register(adapter_cls)
