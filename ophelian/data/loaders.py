"""Per-format dataset loaders for :func:`ophelian.data.materialize`.

Each ``_materialize_*`` helper turns a :class:`~ophelian.core.nodes.Data`
node into the ``{"X": [...], "y": [...]}`` shape that adapters expect.
The dispatcher in :mod:`ophelian.data` picks one based on
``node.format``.

Splitting these out of the package ``__init__`` keeps the public
import path (``from ophelian.data import materialize``) cheap — the
dispatcher only pays for what it dispatches to — and makes each loader
trivially testable in isolation.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ophelian.core.nodes import Data


def _materialize_inline(node: Data) -> dict[str, Any]:
    """Inline data baked into ``options`` — used by tests and 10-line snippets."""
    options = node.options
    if "X" not in options or "y" not in options:
        raise ValueError("Data(format='inline') requires options={'X': [...], 'y': [...]}")
    return {"X": list(options["X"]), "y": list(options["y"])}


def _materialize_synthetic(node: Data) -> dict[str, Any]:
    """Built-in toy datasets (``iris``, ``classification``) for demos."""
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
    """Load ``csv`` / ``json`` / ``jsonl`` / ``parquet`` from a local path or ``s3://`` URI."""
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


def _materialize_huggingface(node: Data) -> dict[str, Any]:
    """Materialise a text corpus for the HuggingFace adapter.

    Returns a JSON-serialisable ``{"texts": [str, ...]}`` payload (the
    standalone provider round-trips datasets through JSON, so a live
    ``datasets.Dataset`` object cannot flow here — the HuggingFace
    adapter tokenises these raw strings itself).

    Two sources are supported:

    * **Inline** — ``options={"texts": ["...", "..."]}``. Fully offline,
      ideal for the quickstart example and tests.
    * **Hub** — a ``hf://<dataset-id>`` source, loaded via the optional
      ``datasets`` library. Pick the text column with
      ``options={"text_column": "..."}`` (default ``"text"``), the split
      with ``options={"split": "..."}`` (default ``"train"``), and cap
      rows with ``options={"limit": N}``.
    """
    options = node.options or {}

    # Inline text — no network, no extra dependency.
    if "texts" in options:
        texts = [str(t) for t in options["texts"]]
        if not texts:
            raise ValueError(
                "Data(format='huggingface') with inline texts requires a "
                "non-empty options={'texts': [...]}"
            )
        return {"texts": texts}

    # Hub dataset — needs the `datasets` package (ships with the
    # `huggingface` extra).
    source = node.source or ""
    if source.startswith("hf://"):
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise NotImplementedError(
                "Loading a HuggingFace Hub dataset requires the `datasets` "
                "package: install `ophelian[huggingface]` (or `pip install "
                "datasets`), or pass inline text via "
                "options={'texts': [...]}."
            ) from exc

        name = source.removeprefix("hf://")
        split = options.get("split", "train")
        text_column = options.get("text_column", "text")
        # Revision pinning is the caller's responsibility (same contract as
        # the model adapters): pass options={'revision': '<sha-or-tag>'} to
        # fetch a fixed dataset snapshot instead of the mutable branch head.
        revision = options.get("revision")
        dataset = load_dataset(name, split=split, revision=revision)
        if text_column not in dataset.column_names:
            raise KeyError(
                f"Data(format='huggingface') source {source!r} has no column "
                f"{text_column!r}; available columns: {dataset.column_names}. "
                "Set options={'text_column': '...'}."
            )
        column = dataset[text_column]
        limit = options.get("limit")
        if limit is not None:
            column = column[: int(limit)]
        return {"texts": [str(t) for t in column]}

    raise ValueError(
        "Data(format='huggingface') needs either options={'texts': [...]} "
        "or an 'hf://<dataset-id>' source."
    )


def _resolve_local_path(source: str) -> Path:
    """Turn a ``file://`` / plain / ``s3://`` source into a local :class:`Path`.

    ``s3://`` is downloaded to a temp file via
    :func:`ophelian.stores.s3.fetch_s3_object_to_tempfile`. Anything else
    raises :class:`NotImplementedError` so the user knows exactly which
    cloud source they're missing instead of getting a vague
    ``FileNotFoundError``.
    """
    parsed = urlparse(source)
    if parsed.scheme in {"", "file"}:
        path = Path(parsed.path or source.removeprefix("file://"))
    elif parsed.scheme == "s3":
        from ophelian.stores.s3 import fetch_s3_object_to_tempfile

        path = fetch_s3_object_to_tempfile(source)
    else:
        raise NotImplementedError(f"Standalone loader cannot fetch remote source {source!r} yet.")
    if not path.exists():
        raise FileNotFoundError(f"Local data source not found: {path}")
    return path


def _coerce(value: Any) -> Any:
    """Best-effort numeric coercion of CSV/JSONL string cells."""
    if isinstance(value, str):
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            return value
    return value
