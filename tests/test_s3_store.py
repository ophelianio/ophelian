"""Unit tests for the S3-backed artifact store, mocked with moto."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import boto3
import pytest
from moto import mock_aws
from ophelian.stores.s3 import (
    S3ArtifactStore,
    fetch_s3_object_to_tempfile,
    parse_s3_uri,
    s3_store_from_uri,
)


@pytest.fixture()
def s3_env() -> Iterator[None]:
    with mock_aws():
        yield


@pytest.fixture()
def store(s3_env: None) -> S3ArtifactStore:
    return S3ArtifactStore(bucket="ophelian-test", region="us-east-1")


def test_parse_s3_uri_round_trips() -> None:
    bucket, key = parse_s3_uri("s3://my-bucket/some/key.txt")
    assert bucket == "my-bucket"
    assert key == "some/key.txt"


def test_parse_s3_uri_rejects_non_s3() -> None:
    with pytest.raises(ValueError, match="Not an s3"):
        parse_s3_uri("file:///tmp/foo")


def test_put_and_get_file(store: S3ArtifactStore, tmp_path: Path) -> None:
    src = tmp_path / "model.bin"
    src.write_bytes(b"weights")

    uri = store.put("models/v1/model.bin", src)
    assert uri.startswith("s3://ophelian-test/")
    assert store.exists("models/v1/model.bin")
    fetched = store.get("models/v1/model.bin", tmp_path / "out.bin")
    assert fetched.read_bytes() == b"weights"


def test_put_directory_uploads_all_files(
    store: S3ArtifactStore, tmp_path: Path
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


def test_put_bytes_and_get_bytes(store: S3ArtifactStore) -> None:
    store.put_bytes("checkpoints/run-1/checkpoint.json", b'{"step": 1}')
    assert store.exists("checkpoints/run-1/checkpoint.json")
    payload = store.get_bytes("checkpoints/run-1/checkpoint.json")
    assert payload == b'{"step": 1}'


def test_get_missing_raises(store: S3ArtifactStore, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        store.get_bytes("missing/key")
    with pytest.raises(FileNotFoundError):
        store.get("missing/key", tmp_path / "x")


def test_delete_removes_single_object(store: S3ArtifactStore, tmp_path: Path) -> None:
    src = tmp_path / "a.txt"
    src.write_text("hi")
    store.put("a.txt", src)
    store.delete("a.txt")
    assert not store.exists("a.txt")


def test_delete_removes_directory(store: S3ArtifactStore, tmp_path: Path) -> None:
    src = tmp_path / "dir"
    src.mkdir()
    (src / "x.txt").write_text("x")
    (src / "y.txt").write_text("y")
    store.put("trees/dir", src)

    store.delete("trees/dir")
    assert list(store.list("trees/dir")) == []


def test_prefix_is_prepended(s3_env: None, tmp_path: Path) -> None:
    store = S3ArtifactStore(bucket="ophelian-test", prefix="runs/run-1", region="us-east-1")
    src = tmp_path / "report.txt"
    src.write_text("ok")
    uri = store.put("report.txt", src)
    assert uri == "s3://ophelian-test/runs/run-1/report.txt"
    assert store.exists("report.txt")


def test_s3_store_from_uri(s3_env: None) -> None:
    store = s3_store_from_uri("s3://ophelian-test/runs/r1", region="us-east-1")
    assert store.bucket == "ophelian-test"
    assert store.prefix == "runs/r1"


def test_fetch_s3_object_helper(s3_env: None, tmp_path: Path) -> None:
    client = boto3.client("s3", region_name="us-east-1")
    client.create_bucket(Bucket="data-bucket")
    client.put_object(Bucket="data-bucket", Key="datasets/sample.json", Body=b'{"X": [], "y": []}')
    cache = tmp_path / "cache"
    path = fetch_s3_object_to_tempfile(
        "s3://data-bucket/datasets/sample.json", cache_dir=cache
    )
    assert path.read_text() == '{"X": [], "y": []}'


def test_fetch_s3_object_missing_raises(s3_env: None, tmp_path: Path) -> None:
    client = boto3.client("s3", region_name="us-east-1")
    client.create_bucket(Bucket="data-bucket")
    with pytest.raises(FileNotFoundError):
        fetch_s3_object_to_tempfile(
            "s3://data-bucket/missing.json", cache_dir=tmp_path / "c"
        )


def test_constructor_rejects_blank_bucket() -> None:
    with pytest.raises(ValueError, match="bucket"):
        S3ArtifactStore(bucket="")
