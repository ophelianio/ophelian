"""Offline tests for the standalone data loaders.

These exercise the ``materialize`` dispatcher without any network access
or heavy ML dependency — in particular the ``huggingface`` inline-text
path added so ``examples/huggingface_pipeline.py`` runs on a fresh
checkout.
"""

from __future__ import annotations

import json

import pytest
from ophelian.core.nodes import Data
from ophelian.data import materialize


def test_huggingface_inline_texts_roundtrip() -> None:
    node = Data(
        name="corpus",
        source="inline://",
        format="huggingface",
        options={"texts": ["hello world", "second line"]},
    )
    payload = materialize(node)
    assert payload == {"texts": ["hello world", "second line"]}
    # The standalone provider round-trips datasets through JSON, so the
    # payload must be JSON-serialisable (a live datasets.Dataset would not be).
    assert json.loads(json.dumps(payload)) == payload


def test_huggingface_inline_coerces_to_str() -> None:
    node = Data(
        name="corpus",
        source="inline://",
        format="huggingface",
        options={"texts": [1, 2.5]},
    )
    assert materialize(node) == {"texts": ["1", "2.5"]}


def test_huggingface_empty_texts_is_rejected() -> None:
    node = Data(name="corpus", source="inline://", format="huggingface", options={"texts": []})
    with pytest.raises(ValueError, match="non-empty"):
        materialize(node)


def test_huggingface_without_texts_or_hf_source_is_rejected() -> None:
    node = Data(name="corpus", source="file://x", format="huggingface", options={})
    with pytest.raises(ValueError, match="hf://"):
        materialize(node)


def test_valid_but_unhandled_format_raises_not_implemented() -> None:
    # ``image-folder`` is an accepted Data.format literal that the
    # standalone loader does not implement yet; it must fail with the
    # descriptive dispatcher error rather than a KeyError.
    node = Data(name="x", source="file://imgs", format="image-folder", options={})
    with pytest.raises(NotImplementedError, match="not yet implemented"):
        materialize(node)
