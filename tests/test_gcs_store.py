"""Unit tests for the GCS-backed artifact store using a fake client.

We deliberately do not depend on ``google-cloud-storage`` in CI because
the SDK is heavy, drags in protobuf, and would force every contributor
to install it just to run the local-only suite. The store accepts a
``client=`` injection precisely for this scenario.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from ophelian.stores.gcs import (
    GCSArtifactStore,
    parse_gs_uri,
)

from tests._fake_gcs import FakeGCSClient


@pytest.fixture()
def store(tmp_path: Path) -> GCSArtifactStore:
    return GCSArtifactStore(
        bucket="ophelian-test",
        prefix="run-x",
        client=FakeGCSClient(),
        cache_dir=tmp_path / "cache",
        ensure_bucket=False,
    )


def test_parse_gs_uri_round_trips() -> None:
    bucket, key = parse_gs_uri("gs://my-bucket/some/key.txt")
    assert bucket == "my-bucket"
    assert key == "some/key.txt"


def test_parse_gs_uri_rejects_non_gs() -> None:
    with pytest.raises(ValueError, match="Not a gs"):
        parse_gs_uri("file:///tmp/foo")


def test_parse_gs_uri_requires_bucket() -> None:
    with pytest.raises(ValueError, match="bucket"):
        parse_gs_uri("gs:///just/a/key")


def test_put_and_get_file(store: GCSArtifactStore, tmp_path: Path) -> None:
    src = tmp_path / "model.bin"
    src.write_bytes(b"weights")

    uri = store.put("models/v1/model.bin", src)
    assert uri == "gs://ophelian-test/run-x/models/v1/model.bin"
    assert store.exists("models/v1/model.bin")

    fetched = store.get("models/v1/model.bin", tmp_path / "out.bin")
    assert fetched.read_bytes() == b"weights"


def test_put_directory_uploads_every_file(store: GCSArtifactStore, tmp_path: Path) -> None:
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


def test_put_bytes_round_trip(store: GCSArtifactStore) -> None:
    store.put_bytes("checkpoints/run-1.json", b'{"step": 1}')
    assert store.exists("checkpoints/run-1.json")
    assert store.get_bytes("checkpoints/run-1.json") == b'{"step": 1}'


def test_get_missing_raises(store: GCSArtifactStore, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        store.get("does/not/exist.bin", tmp_path / "x.bin")


def test_delete_removes_object(store: GCSArtifactStore) -> None:
    store.put_bytes("scratch.json", b"{}")
    assert store.exists("scratch.json")
    store.delete("scratch.json")
    assert not store.exists("scratch.json")


def test_uri_uses_full_prefix(store: GCSArtifactStore) -> None:
    assert store.uri("a/b") == "gs://ophelian-test/run-x/a/b"


def test_empty_bucket_name_rejected() -> None:
    with pytest.raises(ValueError, match="bucket"):
        GCSArtifactStore(bucket="", client=FakeGCSClient(), ensure_bucket=False)
