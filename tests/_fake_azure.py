"""Tiny in-memory fake of the azure-storage-blob client surface.

Implements the small slice :class:`AzureBlobArtifactStore` actually
uses: ``BlobServiceClient.get_container_client(name)`` returning a
container with ``upload_blob``, ``download_blob``, ``delete_blob``,
``list_blobs(name_starts_with=...)``, and ``get_blob_client(name)``.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any


class _FakeDownload:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def readall(self) -> bytes:
        return self._payload


class _FakeBlobClient:
    def __init__(self, container: _FakeContainer, name: str) -> None:
        self._container = container
        self.name = name

    def exists(self) -> bool:
        return self.name in self._container._objects


class _FakeBlobEntry:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeContainer:
    def __init__(self, name: str) -> None:
        self.name = name
        self._objects: OrderedDict[str, bytes] = OrderedDict()

    def create_container(self) -> None:
        return None

    def upload_blob(self, name: str, data: Any, overwrite: bool = False) -> None:
        if isinstance(data, (bytes, bytearray)):
            payload = bytes(data)
        elif hasattr(data, "read"):
            payload = data.read()
        else:
            payload = bytes(data)
        if not overwrite and name in self._objects:
            raise FileExistsError(name)
        self._objects[name] = payload

    def download_blob(self, name: str) -> _FakeDownload:
        if name not in self._objects:
            raise FileNotFoundError(name)
        return _FakeDownload(self._objects[name])

    def delete_blob(self, name: str) -> None:
        self._objects.pop(name, None)

    def list_blobs(self, name_starts_with: str = "") -> list[_FakeBlobEntry]:
        return [_FakeBlobEntry(n) for n in self._objects if n.startswith(name_starts_with)]

    def get_blob_client(self, name: str) -> _FakeBlobClient:
        return _FakeBlobClient(self, name)


class FakeBlobServiceClient:
    def __init__(self) -> None:
        self.containers: dict[str, _FakeContainer] = {}

    def get_container_client(self, name: str) -> _FakeContainer:
        return self.containers.setdefault(name, _FakeContainer(name))
