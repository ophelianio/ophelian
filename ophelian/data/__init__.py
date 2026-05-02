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

This module is the public entry point. The actual format-specific loaders
live in :mod:`ophelian.data.loaders` so ``__init__`` stays a thin
dispatcher and each loader can be imported (and tested) in isolation.
"""

from __future__ import annotations

from typing import Any

from ophelian.core.nodes import Data
from ophelian.data.loaders import (
    _materialize_inline,
    _materialize_local_file,
    _materialize_synthetic,
)


def materialize(node: Data) -> dict[str, Any]:
    """Return a dataset dict ``{"X": ..., "y": ...}`` for the given Data node.

    Adapters expect this shape. Loaders that can't produce X/y (because the
    source is a remote bucket or HF dataset we don't yet support) raise a
    descriptive error instead of silently returning empty data.

    ``s3://`` sources are downloaded transparently to a local temp file and
    then materialised through the same per-format loader.
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


__all__ = ["materialize"]
