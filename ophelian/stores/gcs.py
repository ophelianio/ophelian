"""Google Cloud Storage-backed artifact store.

Implements :class:`~ophelian.stores.base.ArtifactStore` over GCS using the
official ``google-cloud-storage`` SDK. Mirror of :mod:`ophelian.stores.s3`
so the GCP provider can use the exact same artifact contract as the AWS
provider — including spot/preemptible checkpoints uploaded by
:mod:`ophelian.runtime.step_runner`.

`google-cloud-storage` is an optional dependency; importing this module
without it raises :class:`MissingGCPDependencies` with an actionable hint.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger("ophelian.stores.gcs")


class MissingGCPDependencies(ImportError):
    """Raised when ``google-cloud-storage`` is not installed."""


def _require_gcs() -> Any:
    try:
        from google.cloud import storage
    except ImportError as exc:  # pragma: no cover - exercised in CI variants
        raise MissingGCPDependencies(
            "GCS support requires the `gcp` extra: install `ophelian[gcp]` "
            "or `pip install google-cloud-storage`."
        ) from exc
    return storage


def parse_gs_uri(uri: str) -> tuple[str, str]:
    """Split a ``gs://bucket/prefix/key`` URI into ``(bucket, key)``."""
    parsed = urlparse(uri)
    if parsed.scheme != "gs":
        raise ValueError(f"Not a gs:// URI: {uri!r}")
    if not parsed.netloc:
        raise ValueError(f"gs URI is missing a bucket: {uri!r}")
    return parsed.netloc, parsed.path.lstrip("/")


class GCSArtifactStore:
    """Read/write artifacts under a single ``gs://bucket/prefix`` location.

    Parameters
    ----------
    bucket:
        GCS bucket name.
    prefix:
        Optional object name prefix — every ``put``/``get`` is rooted here.
    project:
        GCP project. Falls back to the default credentials chain.
    client:
        Optional pre-built ``google.cloud.storage.Client`` (used by tests
        with a stub).
    cache_dir:
        Local directory used as scratch space when callers ask for an
        on-disk path. Defaults to a fresh temp dir.
    ensure_bucket:
        When True (default), create the bucket if it does not yet exist.
    location:
        Only used when ``ensure_bucket=True`` and the bucket has to be
        created — the GCS region (e.g. ``us-central1``).
    """

    scheme: str = "gs"

    def __init__(
        self,
        bucket: str,
        *,
        prefix: str = "",
        project: str | None = None,
        client: Any | None = None,
        cache_dir: str | Path | None = None,
        ensure_bucket: bool = True,
        location: str | None = None,
    ) -> None:
        if not bucket:
            raise ValueError("GCSArtifactStore requires a non-empty bucket name")
        self._bucket_name = bucket
        self._prefix = prefix.strip("/")
        self._project = project
        self._location = location
        self._client = client if client is not None else self._build_client(project)
        self._cache_dir = (
            Path(cache_dir)
            if cache_dir is not None
            else Path(tempfile.mkdtemp(prefix="ophelian-gcs-cache-"))
        )
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._bucket = self._client.bucket(bucket)
        if ensure_bucket:
            self._ensure_bucket()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_client(project: str | None) -> Any:
        storage = _require_gcs()
        kwargs: dict[str, Any] = {}
        if project:
            kwargs["project"] = project
        return storage.Client(**kwargs)

    def _ensure_bucket(self) -> None:
        try:
            if self._bucket.exists():
                return
        except Exception:  # pragma: no cover - depends on backend
            pass
        try:
            self._client.create_bucket(self._bucket, location=self._location)
        except Exception as exc:  # pragma: no cover - permission-dependent
            logger.debug("create_bucket failed for %s: %s", self._bucket_name, exc)

    # ------------------------------------------------------------------
    # Key helpers
    # ------------------------------------------------------------------

    @property
    def bucket(self) -> str:
        return self._bucket_name

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
        return f"gs://{self._bucket_name}/{self._full_key(key)}"

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
                    self._bucket.blob(sub_key).upload_from_filename(str(path))
                    uploaded += 1
            if uploaded == 0:
                self._bucket.blob(full_key + "/").upload_from_string(b"")
        else:
            self._bucket.blob(full_key).upload_from_filename(str(source_path))
        return self.uri(key)

    def put_bytes(self, key: str, payload: bytes) -> str:
        full_key = self._full_key(key)
        self._bucket.blob(full_key).upload_from_string(payload)
        return self.uri(key)

    def get_bytes(self, key: str) -> bytes:
        full_key = self._full_key(key)
        blob = self._bucket.blob(full_key)
        try:
            return bytes(blob.download_as_bytes())
        except Exception as exc:
            raise FileNotFoundError(
                f"No GCS object at gs://{self._bucket_name}/{full_key}"
            ) from exc

    def get(self, key: str, destination: str | Path | None = None) -> Path:
        full_key = self._full_key(key)
        if destination is None:
            destination = self._cache_dir / key
        dst = Path(destination)
        dst.parent.mkdir(parents=True, exist_ok=True)
        listed = list(self._iter_keys(full_key))
        if not listed:
            raise FileNotFoundError(f"No GCS object at gs://{self._bucket_name}/{full_key}")
        if len(listed) == 1 and listed[0] == full_key:
            self._bucket.blob(full_key).download_to_filename(str(dst))
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
            self._bucket.blob(blob_name).download_to_filename(str(target))
        return dst

    def exists(self, key: str) -> bool:
        full_key = self._full_key(key)
        try:
            if self._bucket.blob(full_key).exists():
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
                self._bucket.blob(full_key).delete()
            return
        for blob_name in keys:
            with contextlib.suppress(Exception):
                self._bucket.blob(blob_name).delete()

    def list(self, prefix: str = "") -> Iterable[str]:
        full_prefix = self._full_key(prefix) if prefix else self._prefix
        for blob_name in self._iter_keys(full_prefix):
            rel = blob_name[len(self._prefix) :].lstrip("/") if self._prefix else blob_name
            yield rel

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _iter_keys(self, prefix: str) -> Iterable[str]:
        for blob in self._client.list_blobs(self._bucket_name, prefix=prefix):
            yield str(blob.name)


def gcs_store_from_uri(uri: str, **kwargs: Any) -> GCSArtifactStore:
    """Build a :class:`GCSArtifactStore` rooted at *uri* (``gs://bucket/prefix``)."""
    bucket, prefix = parse_gs_uri(uri)
    return GCSArtifactStore(bucket=bucket, prefix=prefix, **kwargs)


def fetch_gcs_object_to_tempfile(
    uri: str,
    *,
    client: Any | None = None,
    project: str | None = None,
    cache_dir: str | Path | None = None,
) -> Path:
    """Download a single ``gs://`` object to a local temp file and return it."""
    bucket, key = parse_gs_uri(uri)
    if not key:
        raise ValueError(f"gs URI must include an object key: {uri!r}")
    if client is None:
        client = GCSArtifactStore._build_client(project)
    cache = (
        Path(cache_dir) if cache_dir is not None else Path(tempfile.mkdtemp(prefix="ophelian-gcs-"))
    )
    cache.mkdir(parents=True, exist_ok=True)
    safe_name = key.replace("/", "__") or "object"
    dst = cache / safe_name
    try:
        client.bucket(bucket).blob(key).download_to_filename(str(dst))
    except Exception as exc:
        raise FileNotFoundError(f"Could not download {uri}: {exc}") from exc
    return dst


def cleanup_local_cache(store: GCSArtifactStore) -> None:
    """Best-effort removal of the store's on-disk cache."""
    import contextlib

    with contextlib.suppress(Exception):  # pragma: no cover - defensive
        shutil.rmtree(store.cache_dir, ignore_errors=True)


__all__ = [
    "GCSArtifactStore",
    "MissingGCPDependencies",
    "cleanup_local_cache",
    "fetch_gcs_object_to_tempfile",
    "gcs_store_from_uri",
    "parse_gs_uri",
]
