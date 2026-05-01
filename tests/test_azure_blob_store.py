"""Unit tests for the Azure Blob-backed artifact store using a fake client."""

from __future__ import annotations

from pathlib import Path

import pytest
from ophelian.stores.azure_blob import (
    AzureBlobArtifactStore,
    parse_az_uri,
)

from tests._fake_azure import FakeBlobServiceClient


@pytest.fixture()
def store(tmp_path: Path) -> AzureBlobArtifactStore:
    return AzureBlobArtifactStore(
        account="ophelianacct",
        container="artifacts",
        prefix="run-x",
        client=FakeBlobServiceClient(),
        cache_dir=tmp_path / "cache",
        ensure_container=False,
    )


def test_parse_az_uri_round_trips() -> None:
    account, container, key = parse_az_uri("az://acct/cont/some/key.txt")
    assert account == "acct"
    assert container == "cont"
    assert key == "some/key.txt"


def test_parse_az_uri_rejects_non_az() -> None:
    with pytest.raises(ValueError, match="Not an az"):
        parse_az_uri("file:///tmp/foo")


def test_parse_az_uri_requires_container_and_key() -> None:
    with pytest.raises(ValueError, match="container"):
        parse_az_uri("az://acct/only-container")


def test_put_and_get_file(store: AzureBlobArtifactStore, tmp_path: Path) -> None:
    src = tmp_path / "model.bin"
    src.write_bytes(b"weights")

    uri = store.put("models/v1/model.bin", src)
    assert uri == "az://ophelianacct/artifacts/run-x/models/v1/model.bin"
    assert store.exists("models/v1/model.bin")

    fetched = store.get("models/v1/model.bin", tmp_path / "out.bin")
    assert fetched.read_bytes() == b"weights"


def test_put_directory_uploads_every_file(
    store: AzureBlobArtifactStore, tmp_path: Path
) -> None:
    src = tmp_path / "model"
    (src / "inner").mkdir(parents=True)
    (src / "inner" / "weights.bin").write_bytes(b"abc")
    (src / "config.json").write_text("{}")

    store.put("models/dir", src)

    listed = sorted(store.list("models/dir"))
    assert "models/dir/inner/weights.bin" in listed
    assert "models/dir/config.json" in listed

    fetched = store.get("models/dir", tmp_path / "round-trip")
    assert fetched.is_dir()
    assert (fetched / "inner" / "weights.bin").read_bytes() == b"abc"
    assert (fetched / "config.json").read_text() == "{}"


def test_put_bytes_round_trip(store: AzureBlobArtifactStore) -> None:
    store.put_bytes("checkpoints/run-1.json", b'{"step": 1}')
    assert store.exists("checkpoints/run-1.json")
    assert store.get_bytes("checkpoints/run-1.json") == b'{"step": 1}'


def test_get_missing_raises(store: AzureBlobArtifactStore, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        store.get("does/not/exist.bin", tmp_path / "x.bin")


def test_delete_removes_object(store: AzureBlobArtifactStore) -> None:
    store.put_bytes("scratch.json", b"{}")
    assert store.exists("scratch.json")
    store.delete("scratch.json")
    assert not store.exists("scratch.json")


def test_uri_uses_full_prefix(store: AzureBlobArtifactStore) -> None:
    assert store.uri("a/b") == "az://ophelianacct/artifacts/run-x/a/b"


def test_empty_account_rejected() -> None:
    with pytest.raises(ValueError, match="account"):
        AzureBlobArtifactStore(
            account="",
            container="c",
            client=FakeBlobServiceClient(),
            ensure_container=False,
        )


def test_empty_container_rejected() -> None:
    with pytest.raises(ValueError, match="container"):
        AzureBlobArtifactStore(
            account="acct",
            container="",
            client=FakeBlobServiceClient(),
            ensure_container=False,
        )
