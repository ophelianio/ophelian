"""Tests for the local artifact store."""

from __future__ import annotations

from pathlib import Path

import pytest
from ophelian.stores import LocalArtifactStore


def test_put_and_get_file(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "store")
    src = tmp_path / "model.bin"
    src.write_bytes(b"weights")

    uri = store.put("models/v1/model.bin", src)
    assert uri.startswith("file://")
    assert store.exists("models/v1/model.bin")
    fetched = store.get("models/v1/model.bin")
    assert fetched.read_bytes() == b"weights"


def test_put_and_get_directory(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "store")
    src = tmp_path / "src"
    (src / "inner").mkdir(parents=True)
    (src / "inner" / "weights.bin").write_bytes(b"abc")

    store.put("models/dir", src)
    fetched = store.get("models/dir")
    assert (fetched / "inner" / "weights.bin").read_bytes() == b"abc"


def test_get_missing_raises(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "store")
    with pytest.raises(FileNotFoundError):
        store.get("missing")


def test_delete_removes_artifact(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "store")
    src = tmp_path / "x.txt"
    src.write_text("hi")
    store.put("a/x.txt", src)
    store.delete("a/x.txt")
    assert not store.exists("a/x.txt")
