"""Tiny in-memory fake of the google-cloud-storage client surface.

Only implements the bits :class:`ophelian.stores.gcs.GCSArtifactStore`
actually touches: ``Client.bucket(name)`` returning a bucket whose
``blob(name)`` exposes ``upload_from_filename``, ``upload_from_string``,
``download_to_filename``, ``download_as_bytes``, ``exists``, ``delete``;
plus ``Client.list_blobs(bucket, prefix=...)`` returning iterable
objects with a ``name`` attribute.

We deliberately keep this tiny — the goal is to exercise the store's
own logic, not to reimplement GCS.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any


class _FakeBlob:
    def __init__(self, bucket: _FakeBucket, name: str) -> None:
        self._bucket = bucket
        self.name = name

    def upload_from_filename(self, path: str) -> None:
        with open(path, "rb") as fh:
            self._bucket._objects[self.name] = fh.read()

    def upload_from_string(self, data: Any) -> None:
        if isinstance(data, str):
            data = data.encode()
        self._bucket._objects[self.name] = bytes(data)

    def download_to_filename(self, path: str) -> None:
        if self.name not in self._bucket._objects:
            raise FileNotFoundError(self.name)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(self._bucket._objects[self.name])

    def download_as_bytes(self) -> bytes:
        if self.name not in self._bucket._objects:
            raise FileNotFoundError(self.name)
        return self._bucket._objects[self.name]

    def exists(self) -> bool:
        return self.name in self._bucket._objects

    def delete(self) -> None:
        self._bucket._objects.pop(self.name, None)


class _FakeBucket:
    def __init__(self, name: str) -> None:
        self.name = name
        self._objects: OrderedDict[str, bytes] = OrderedDict()
        self._exists = True

    def exists(self) -> bool:
        return self._exists

    def blob(self, name: str) -> _FakeBlob:
        return _FakeBlob(self, name)


class FakeGCSClient:
    def __init__(self) -> None:
        self.buckets: dict[str, _FakeBucket] = {}

    def bucket(self, name: str) -> _FakeBucket:
        return self.buckets.setdefault(name, _FakeBucket(name))

    def create_bucket(self, bucket: _FakeBucket | str, location: str | None = None) -> _FakeBucket:
        name = bucket.name if isinstance(bucket, _FakeBucket) else bucket
        return self.buckets.setdefault(name, _FakeBucket(name))

    def list_blobs(self, bucket_name: str, prefix: str = "") -> list[_FakeBlob]:
        bucket = self.buckets.get(bucket_name)
        if bucket is None:
            return []
        return [_FakeBlob(bucket, name) for name in bucket._objects if name.startswith(prefix)]
