"""Azure Blob Storage-backed artifact store.

Implements :class:`~ophelian.stores.base.ArtifactStore` over Azure Blob
Storage using the ``azure-storage-blob`` SDK. Mirror of
:mod:`ophelian.stores.s3` so the Azure provider can use the exact same
artifact contract as the AWS provider.

`azure-storage-blob` is an optional dependency; importing this module
without it raises :class:`MissingAzureDependencies` with an actionable
hint.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger("ophelian.stores.azure_blob")


class MissingAzureDependencies(ImportError):
    """Raised when ``azure-storage-blob`` / ``azure-identity`` are not installed."""


def _require_blob() -> Any:
    try:
        from azure.storage.blob import BlobServiceClient
    except ImportError as exc:  # pragma: no cover - exercised in CI variants
        raise MissingAzureDependencies(
            "Azure Blob support requires the `azure` extra: install "
            "`ophelian[azure]` or `pip install azure-storage-blob azure-identity`."
        ) from exc
    return BlobServiceClient


def parse_az_uri(uri: str) -> tuple[str, str, str]:
    """Split an ``az://account/container/key`` URI into ``(account, container, key)``.

    We use the ``az://`` pseudo-scheme — it's terser than the official
    ``https://{account}.blob.core.windows.net/{container}/{key}`` form
    used by Azure SDKs and matches the spirit of the ``s3://`` /
    ``gs://`` prefixes the rest of Ophelian uses. The store knows how
    to translate either form when needed.
    """
    parsed = urlparse(uri)
    if parsed.scheme != "az":
        raise ValueError(f"Not an az:// URI: {uri!r}")
    account = parsed.netloc
    if not account:
        raise ValueError(f"az URI is missing an account: {uri!r}")
    parts = parsed.path.lstrip("/").split("/", 1)
    if len(parts) < 2 or not parts[0]:
        raise ValueError(f"az URI must include container and key: {uri!r}")
    container, key = parts[0], parts[1]
    return account, container, key


class AzureBlobArtifactStore:
    """Read/write artifacts under a single ``az://account/container/prefix`` location.

    Parameters
    ----------
    account:
        Storage account name.
    container:
        Blob container.
    prefix:
        Optional blob name prefix — every ``put``/``get`` is rooted here.
    credential:
        Anything the ``BlobServiceClient`` understands — connection
        string, ``AzureCliCredential``, ``DefaultAzureCredential``, etc.
        Falls back to ``DefaultAzureCredential`` when omitted.
    client:
        Optional pre-built ``BlobServiceClient`` (used by tests).
    cache_dir:
        Local directory used as scratch space when callers ask for an
        on-disk path. Defaults to a fresh temp dir.
    ensure_container:
        When True (default), create the container if it does not yet
        exist.
    """

    scheme: str = "az"

    def __init__(
        self,
        account: str,
        container: str,
        *,
        prefix: str = "",
        credential: Any | None = None,
        client: Any | None = None,
        cache_dir: str | Path | None = None,
        ensure_container: bool = True,
    ) -> None:
        if not account:
            raise ValueError("AzureBlobArtifactStore requires a non-empty account name")
        if not container:
            raise ValueError("AzureBlobArtifactStore requires a non-empty container name")
        self._account = account
        self._container_name = container
        self._prefix = prefix.strip("/")
        self._credential = credential
        self._service_client = (
            client if client is not None else self._build_client(account, credential)
        )
        self._container = self._service_client.get_container_client(container)
        self._cache_dir = (
            Path(cache_dir)
            if cache_dir is not None
            else Path(tempfile.mkdtemp(prefix="ophelian-az-cache-"))
        )
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        if ensure_container:
            self._ensure_container()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_client(account: str, credential: Any | None) -> Any:
        BlobServiceClient = _require_blob()
        if credential is None:
            try:
                from azure.identity import DefaultAzureCredential

                credential = DefaultAzureCredential()
            except ImportError as exc:  # pragma: no cover
                raise MissingAzureDependencies(
                    "Azure auth requires `azure-identity`: install `ophelian[azure]`."
                ) from exc
        url = f"https://{account}.blob.core.windows.net"
        return BlobServiceClient(account_url=url, credential=credential)

    def _ensure_container(self) -> None:
        import contextlib

        with contextlib.suppress(Exception):  # pragma: no cover - permission-dependent
            self._container.create_container()

    # ------------------------------------------------------------------
    # Key helpers
    # ------------------------------------------------------------------

    @property
    def account(self) -> str:
        return self._account

    @property
    def container(self) -> str:
        return self._container_name

    @property
    def prefix(self) -> str:
        return self._prefix

    @property
    def cache_dir(self) -> Path:
        return self._cache_dir

    def _full_key(self, key: str) -> str:
        key = key.lstrip("/")
        if self._prefix:
            return f"{self._prefix}/{key}"
        return key

    def uri(self, key: str) -> str:
        return f"az://{self._account}/{self._container_name}/{self._full_key(key)}"

    # ------------------------------------------------------------------
    # ArtifactStore interface
    # ------------------------------------------------------------------

    def put(self, key: str, source: str | Path) -> str:
        source_path = Path(source)
        if not source_path.exists():
            raise FileNotFoundError(f"Cannot upload missing path: {source_path}")
        full_key = self._full_key(key)
        if source_path.is_dir():
            uploaded = 0
            for path in source_path.rglob("*"):
                if path.is_file():
                    rel = path.relative_to(source_path).as_posix()
                    sub_key = f"{full_key}/{rel}" if full_key else rel
                    with path.open("rb") as fh:
                        self._container.upload_blob(sub_key, fh, overwrite=True)
                    uploaded += 1
            if uploaded == 0:
                self._container.upload_blob(full_key + "/", b"", overwrite=True)
        else:
            with source_path.open("rb") as fh:
                self._container.upload_blob(full_key, fh, overwrite=True)
        return self.uri(key)

    def put_bytes(self, key: str, payload: bytes) -> str:
        full_key = self._full_key(key)
        self._container.upload_blob(full_key, payload, overwrite=True)
        return self.uri(key)

    def get_bytes(self, key: str) -> bytes:
        full_key = self._full_key(key)
        try:
            blob = self._container.download_blob(full_key)
            return bytes(blob.readall())
        except Exception as exc:
            raise FileNotFoundError(
                f"No Azure blob at az://{self._account}/{self._container_name}/{full_key}"
            ) from exc

    def get(self, key: str, destination: str | Path | None = None) -> Path:
        full_key = self._full_key(key)
        if destination is None:
            destination = self._cache_dir / key
        dst = Path(destination)
        dst.parent.mkdir(parents=True, exist_ok=True)
        listed = list(self._iter_keys(full_key))
        if not listed:
            raise FileNotFoundError(
                f"No Azure blob at az://{self._account}/{self._container_name}/{full_key}"
            )
        if len(listed) == 1 and listed[0] == full_key:
            with dst.open("wb") as fh:
                fh.write(self._container.download_blob(full_key).readall())
            return dst
        if dst.exists() and dst.is_file():
            dst.unlink()
        dst.mkdir(parents=True, exist_ok=True)
        for blob_name in listed:
            rel = blob_name[len(full_key) :].lstrip("/")
            if not rel:
                continue
            target = dst / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("wb") as fh:
                fh.write(self._container.download_blob(blob_name).readall())
        return dst

    def exists(self, key: str) -> bool:
        full_key = self._full_key(key)
        try:
            if self._container.get_blob_client(full_key).exists():
                return True
        except Exception:
            pass
        return any(True for _ in self._iter_keys(full_key))

    def delete(self, key: str) -> None:
        import contextlib

        full_key = self._full_key(key)
        keys = list(self._iter_keys(full_key))
        if not keys:
            with contextlib.suppress(Exception):
                self._container.delete_blob(full_key)
            return
        for blob_name in keys:
            with contextlib.suppress(Exception):
                self._container.delete_blob(blob_name)

    def list(self, prefix: str = "") -> Iterable[str]:
        full_prefix = self._full_key(prefix) if prefix else self._prefix
        for blob_name in self._iter_keys(full_prefix):
            rel = blob_name[len(self._prefix) :].lstrip("/") if self._prefix else blob_name
            yield rel

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _iter_keys(self, prefix: str) -> Iterable[str]:
        for blob in self._container.list_blobs(name_starts_with=prefix):
            name = getattr(blob, "name", None) or blob["name"]
            yield str(name)


def az_store_from_uri(uri: str, **kwargs: Any) -> AzureBlobArtifactStore:
    """Build an :class:`AzureBlobArtifactStore` rooted at ``az://account/container/prefix``."""
    account, container, prefix = parse_az_uri(uri)
    return AzureBlobArtifactStore(account=account, container=container, prefix=prefix, **kwargs)


def cleanup_local_cache(store: AzureBlobArtifactStore) -> None:
    """Best-effort removal of the store's on-disk cache."""
    import contextlib

    with contextlib.suppress(Exception):  # pragma: no cover - defensive
        shutil.rmtree(store.cache_dir, ignore_errors=True)


__all__ = [
    "AzureBlobArtifactStore",
    "MissingAzureDependencies",
    "az_store_from_uri",
    "cleanup_local_cache",
    "parse_az_uri",
]
