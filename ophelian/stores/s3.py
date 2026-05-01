"""S3-backed artifact store.

Implements :class:`~ophelian.stores.base.ArtifactStore` over Amazon S3 (or
any S3-compatible endpoint). Used by the AWS provider to persist datasets,
models, checkpoints and step results so they survive across the ephemeral
EC2 instances that actually run training.

The store keeps a small local "working" directory so callers that hand back
:class:`pathlib.Path` objects (most adapters) do not have to know they are
actually streaming bytes from S3. Files are uploaded with multipart support
via boto3 and downloaded streaming to disk so that very large checkpoints
work without buffering the whole payload in memory.

`boto3` and `botocore` are optional dependencies. Importing
:class:`S3ArtifactStore` without them raises :class:`MissingAWSDependencies`
with an actionable hint.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

if TYPE_CHECKING:  # pragma: no cover
    pass

logger = logging.getLogger("ophelian.stores.s3")


class MissingAWSDependencies(ImportError):
    """Raised when boto3/botocore are not installed."""


def _require_boto3() -> Any:
    try:
        import boto3
    except ImportError as exc:  # pragma: no cover - exercised in CI variants
        raise MissingAWSDependencies(
            "S3 support requires the `aws` extra: install `ophelian[aws]` "
            "or `pip install boto3 botocore`."
        ) from exc
    return boto3


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Split an ``s3://bucket/prefix/key`` URI into ``(bucket, key)``."""
    parsed = urlparse(uri)
    if parsed.scheme != "s3":
        raise ValueError(f"Not an s3:// URI: {uri!r}")
    if not parsed.netloc:
        raise ValueError(f"s3 URI is missing a bucket: {uri!r}")
    return parsed.netloc, parsed.path.lstrip("/")


class S3ArtifactStore:
    """Read/write artifacts under a single ``s3://bucket/prefix`` location.

    Parameters
    ----------
    bucket:
        S3 bucket name. Created lazily when ``ensure_bucket=True``.
    prefix:
        Optional key prefix — every ``put``/``get`` is rooted here.
    region:
        AWS region. Falls back to the boto3 default chain.
    client:
        Optional pre-built boto3 S3 client (used by tests with moto).
    cache_dir:
        Local directory used as scratch space when callers ask for an
        on-disk path. Defaults to a fresh temp dir.
    ensure_bucket:
        When True (default), create the bucket if it does not yet exist.
        This is convenient for first-run / examples; production users
        usually set ``ensure_bucket=False`` and pre-create with their own
        IaC.
    """

    scheme: str = "s3"

    def __init__(
        self,
        bucket: str,
        *,
        prefix: str = "",
        region: str | None = None,
        client: Any | None = None,
        cache_dir: str | Path | None = None,
        ensure_bucket: bool = True,
    ) -> None:
        if not bucket:
            raise ValueError("S3ArtifactStore requires a non-empty bucket name")
        self._bucket = bucket
        self._prefix = prefix.strip("/")
        self._region = region
        self._client = client if client is not None else self._build_client(region)
        self._cache_dir = (
            Path(cache_dir)
            if cache_dir is not None
            else Path(tempfile.mkdtemp(prefix="ophelian-s3-cache-"))
        )
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        if ensure_bucket:
            self._ensure_bucket()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_client(region: str | None) -> Any:
        boto3 = _require_boto3()
        kwargs: dict[str, Any] = {}
        if region:
            kwargs["region_name"] = region
        return boto3.client("s3", **kwargs)

    def _ensure_bucket(self) -> None:
        try:
            self._client.head_bucket(Bucket=self._bucket)
            return
        except Exception:  # pragma: no cover - depends on boto3 error class
            pass
        kwargs: dict[str, Any] = {"Bucket": self._bucket}
        if self._region and self._region != "us-east-1":
            kwargs["CreateBucketConfiguration"] = {"LocationConstraint": self._region}
        try:
            self._client.create_bucket(**kwargs)
        except Exception as exc:  # pragma: no cover - permission-dependent
            logger.debug("create_bucket failed for %s: %s", self._bucket, exc)

    # ------------------------------------------------------------------
    # Key helpers
    # ------------------------------------------------------------------

    @property
    def bucket(self) -> str:
        return self._bucket

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
        return f"s3://{self._bucket}/{self._full_key(key)}"

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
                    self._client.upload_file(str(path), self._bucket, sub_key)
                    uploaded += 1
            if uploaded == 0:
                # Mark empty directories with a placeholder so `exists`
                # still returns True.
                self._client.put_object(Bucket=self._bucket, Key=full_key + "/", Body=b"")
        else:
            self._client.upload_file(str(source_path), self._bucket, full_key)
        return self.uri(key)

    def put_bytes(self, key: str, payload: bytes) -> str:
        full_key = self._full_key(key)
        self._client.put_object(Bucket=self._bucket, Key=full_key, Body=payload)
        return self.uri(key)

    def get_bytes(self, key: str) -> bytes:
        full_key = self._full_key(key)
        try:
            response = self._client.get_object(Bucket=self._bucket, Key=full_key)
        except Exception as exc:
            raise FileNotFoundError(f"No S3 object at s3://{self._bucket}/{full_key}") from exc
        body = response["Body"].read()
        return bytes(body)

    def get(self, key: str, destination: str | Path | None = None) -> Path:
        full_key = self._full_key(key)
        if destination is None:
            destination = self._cache_dir / key
        dst = Path(destination)
        dst.parent.mkdir(parents=True, exist_ok=True)
        # Try as a directory first by listing matching keys.
        listed = list(self._iter_keys(full_key))
        if not listed:
            raise FileNotFoundError(f"No S3 object at s3://{self._bucket}/{full_key}")
        if len(listed) == 1 and listed[0] == full_key:
            self._client.download_file(self._bucket, full_key, str(dst))
            return dst
        # Multi-key prefix → reconstruct as a directory under dst.
        if dst.exists() and dst.is_file():
            dst.unlink()
        dst.mkdir(parents=True, exist_ok=True)
        for s3_key in listed:
            rel = s3_key[len(full_key) :].lstrip("/")
            if not rel:
                continue
            target = dst / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            self._client.download_file(self._bucket, s3_key, str(target))
        return dst

    def exists(self, key: str) -> bool:
        full_key = self._full_key(key)
        try:
            self._client.head_object(Bucket=self._bucket, Key=full_key)
            return True
        except Exception:
            pass
        return any(True for _ in self._iter_keys(full_key))

    def delete(self, key: str) -> None:
        import contextlib

        full_key = self._full_key(key)
        keys = list(self._iter_keys(full_key))
        if not keys:
            with contextlib.suppress(Exception):  # pragma: no cover
                self._client.delete_object(Bucket=self._bucket, Key=full_key)
            return
        for s3_key in keys:
            self._client.delete_object(Bucket=self._bucket, Key=s3_key)

    def list(self, prefix: str = "") -> Iterable[str]:
        full_prefix = self._full_key(prefix) if prefix else self._prefix
        for s3_key in self._iter_keys(full_prefix):
            rel = s3_key[len(self._prefix) :].lstrip("/") if self._prefix else s3_key
            yield rel

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _iter_keys(self, prefix: str) -> Iterable[str]:
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
            for obj in page.get("Contents", []) or []:
                yield obj["Key"]


def s3_store_from_uri(uri: str, **kwargs: Any) -> S3ArtifactStore:
    """Build an :class:`S3ArtifactStore` rooted at *uri* (``s3://bucket/prefix``)."""
    bucket, prefix = parse_s3_uri(uri)
    return S3ArtifactStore(bucket=bucket, prefix=prefix, **kwargs)


def fetch_s3_object_to_tempfile(
    uri: str,
    *,
    client: Any | None = None,
    region: str | None = None,
    cache_dir: str | Path | None = None,
) -> Path:
    """Download a single ``s3://`` object to a local temp file and return it.

    Used by the data loader so ``Data(source="s3://...")`` works without
    threading an :class:`S3ArtifactStore` through the call stack.
    """
    bucket, key = parse_s3_uri(uri)
    if not key:
        raise ValueError(f"s3 URI must include an object key: {uri!r}")
    if client is None:
        client = S3ArtifactStore._build_client(region)
    cache = Path(cache_dir) if cache_dir is not None else Path(tempfile.mkdtemp(prefix="ophelian-s3-"))
    cache.mkdir(parents=True, exist_ok=True)
    safe_name = key.replace("/", "__") or "object"
    dst = cache / safe_name
    try:
        client.download_file(bucket, key, str(dst))
    except Exception as exc:
        raise FileNotFoundError(f"Could not download {uri}: {exc}") from exc
    return dst


def cleanup_local_cache(store: S3ArtifactStore) -> None:
    """Best-effort removal of the store's on-disk cache."""
    import contextlib

    with contextlib.suppress(Exception):  # pragma: no cover - defensive
        shutil.rmtree(store.cache_dir, ignore_errors=True)


__all__ = [
    "MissingAWSDependencies",
    "S3ArtifactStore",
    "cleanup_local_cache",
    "fetch_s3_object_to_tempfile",
    "parse_s3_uri",
    "s3_store_from_uri",
]


def _env_or_none(name: str) -> str | None:
    value = os.environ.get(name)
    return value or None
