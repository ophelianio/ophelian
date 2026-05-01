"""Data loading helpers used by the standalone provider and adapters.

A `Data` node describes *where* a dataset lives. The loader knows how to
materialise it into something the adapter understands. Today we support:

- ``inline``: literal Python data baked into ``options`` — perfect for tests
  and ten-line quickstart snippets.
- ``csv`` / ``json`` / ``jsonl`` / ``parquet`` over a local ``file://`` or
  plain path — loaded with the lightest dependency available.
- ``synthetic``: small built-in toy datasets (``iris``, ``moons``, ``classification``)
  for demos that don't ship their own data.

Cloud / HuggingFace sources land in later milestones; for now the loader
records the URI in the manifest so downstream steps know what was requested.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ophelian.core.nodes import Data


def materialize(node: Data) -> dict[str, Any]:
    """Return a dataset dict ``{"X": ..., "y": ...}`` for the given Data node.

    Adapters expect this shape. Loaders that can't produce X/y (because the
    source is a remote bucket or HF dataset we don't yet support) raise a
    descriptive error instead of silently returning empty data.
    """
    fmt = node.format
    if fmt == "inline":
        return _materialize_inline(node)
    if fmt == "synthetic":
        return _materialize_synthetic(node)
    if fmt in {"csv", "json", "jsonl", "parquet"}:
        return _materialize_local_file(node)
    raise NotImplementedError(
        f"Data format {fmt!r} is not yet implemented in the standalone loader."
    )


def _materialize_inline(node: Data) -> dict[str, Any]:
    options = node.options
    if "X" not in options or "y" not in options:
        raise ValueError("Data(format='inline') requires options={'X': [...], 'y': [...]}")
    return {"X": list(options["X"]), "y": list(options["y"])}


def _materialize_synthetic(node: Data) -> dict[str, Any]:
    name = node.options.get("name", node.source.removeprefix("synthetic://") or "iris")
    if name == "iris":
        try:
            from sklearn.datasets import load_iris

            iris = load_iris()
            return {"X": iris.data.tolist(), "y": iris.target.tolist()}
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("synthetic iris requires scikit-learn") from exc
    if name == "classification":
        try:
            from sklearn.datasets import make_classification

            x, y = make_classification(
                n_samples=node.options.get("n_samples", 200),
                n_features=node.options.get("n_features", 4),
                n_classes=node.options.get("n_classes", 2),
                random_state=node.options.get("random_state", 0),
            )
            return {"X": x.tolist(), "y": y.tolist()}
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("synthetic classification requires scikit-learn") from exc
    raise ValueError(f"Unknown synthetic dataset: {name!r}")


def _materialize_local_file(node: Data) -> dict[str, Any]:
    path = _resolve_local_path(node.source)
    target_column = node.options.get("target", "y")
    if node.format == "csv":
        rows = list(csv.DictReader(path.read_text().splitlines()))
        x = [[float(v) for k, v in r.items() if k != target_column] for r in rows]
        y = [_coerce(r[target_column]) for r in rows]
        return {"X": x, "y": y}
    if node.format == "jsonl":
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        x = [[float(v) for k, v in r.items() if k != target_column] for r in rows]
        y = [_coerce(r[target_column]) for r in rows]
        return {"X": x, "y": y}
    if node.format == "json":
        payload = json.loads(path.read_text())
        return {"X": list(payload["X"]), "y": list(payload["y"])}
    if node.format == "parquet":
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Loading parquet requires pandas (and pyarrow).") from exc
        df = pd.read_parquet(path)
        y = df[target_column].tolist()
        x = df.drop(columns=[target_column]).values.tolist()
        return {"X": x, "y": y}
    raise NotImplementedError(f"Unsupported format {node.format!r}")  # pragma: no cover


def _resolve_local_path(source: str) -> Path:
    parsed = urlparse(source)
    if parsed.scheme in {"", "file"}:
        path = Path(parsed.path or source.removeprefix("file://"))
    else:
        raise NotImplementedError(f"Standalone loader cannot fetch remote source {source!r} yet.")
    if not path.exists():
        raise FileNotFoundError(f"Local data source not found: {path}")
    return path


def _coerce(value: Any) -> Any:
    if isinstance(value, str):
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            return value
    return value


__all__ = ["materialize"]
