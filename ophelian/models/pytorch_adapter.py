"""Adapter for PyTorch models (vision and tabular).

The model spec accepted by ``train`` is a fully-qualified class path. Two
shapes are supported in v0.1:

* ``torchvision.models.<name>`` — an off-the-shelf vision model (e.g.
  ``torchvision.models.resnet18``). ``hyperparameters`` are forwarded to the
  builder. With no training data it returns the constructed network so that
  Deploy / Eval can still load and serve it.
* Any importable ``torch.nn.Module`` subclass (e.g. ``torch.nn.Linear``).
  Given a tabular dataset of ``{"X": [[...]], "y": [...]}`` and a Linear-shaped
  model, the adapter runs a small SGD loop and returns the trained module.

``save`` writes the *entire* module with ``torch.save`` so ``load`` can return
a callable ready for ``predict``. This keeps the contract symmetric.
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any, ClassVar

from ophelian.models.base import ModelAdapter, register_adapter


@register_adapter
class PyTorchAdapter(ModelAdapter):
    framework: ClassVar[str] = "pytorch"

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
        import torch

        net = self._build_model(model, hyperparameters)
        if data is None:
            return net
        if isinstance(data, dict) and "X" in data and "y" in data:
            x_tensor = torch.tensor(data["X"], dtype=torch.float32)
            y_values = data["y"]
            loss_fn: torch.nn.Module
            if isinstance(y_values[0], int):
                y_tensor = torch.tensor(y_values, dtype=torch.long)
                loss_fn = torch.nn.CrossEntropyLoss()
            else:
                # Reshape regression targets to match model output shape
                # (most regressors output (N, 1)).
                y_tensor = torch.tensor(y_values, dtype=torch.float32).reshape(-1, 1)
                loss_fn = torch.nn.MSELoss()
            optim = torch.optim.SGD(net.parameters(), lr=hyperparameters.get("lr", 1e-1))
            for _ in range(epochs or 1):
                optim.zero_grad()
                out = net(x_tensor)
                loss = loss_fn(out, y_tensor)
                loss.backward()
                optim.step()
            return net
        # Iterable of (inputs, targets) batches — the original vision contract.
        loss_fn = torch.nn.CrossEntropyLoss()
        optim = torch.optim.SGD(net.parameters(), lr=hyperparameters.get("lr", 1e-3))
        for _ in range(epochs or 1):
            for batch in data:
                inputs, targets = batch
                optim.zero_grad()
                out = net(inputs)
                loss = loss_fn(out, targets)
                loss.backward()
                optim.step()
        return net

    def save(self, model: Any, path: Path) -> Path:
        import torch

        path.mkdir(parents=True, exist_ok=True)
        target = path / "model.pt"
        # Save the whole module so `load` can return a callable directly.
        torch.save(model, target)
        return target

    def load(self, path: Path) -> Any:
        import torch

        target = path / "model.pt"
        model = torch.load(target, map_location="cpu", weights_only=False)
        if hasattr(model, "eval"):
            model.eval()
        return model

    def predict(self, model: Any, payload: Any) -> Any:
        import torch

        tensor = (
            payload
            if isinstance(payload, torch.Tensor)
            else torch.tensor(payload, dtype=torch.float32)
        )
        with torch.no_grad():
            output = model(tensor)
        if hasattr(output, "tolist"):
            return output.tolist()
        return output

    # Keys consumed by the optimizer / training loop, not by the model ctor.
    _TRAIN_ONLY_KEYS: ClassVar[frozenset[str]] = frozenset({"lr", "momentum", "weight_decay"})

    @classmethod
    def _build_model(cls, model: str, hyperparameters: dict[str, Any]) -> Any:
        if "." not in model:
            raise ValueError(
                f"PyTorch model spec must be a fully qualified path "
                f"(e.g. 'torch.nn.Linear'); got {model!r}"
            )
        module_path, _, cls_name = model.rpartition(".")
        module = import_module(module_path)
        builder = getattr(module, cls_name)
        ctor_kwargs = {k: v for k, v in hyperparameters.items() if k not in cls._TRAIN_ONLY_KEYS}
        return builder(**ctor_kwargs)
