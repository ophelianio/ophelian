"""Artifact stores — abstractions over where models, datasets and reports live.

The local filesystem store is always available. Cloud-backed stores
(S3 / GCS / Azure Blob) are part of optional dependency extras and are
lazy-imported so installing ``ophelian`` without the corresponding
extra still works.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ophelian.stores.base import ArtifactStore
from ophelian.stores.local import LocalArtifactStore

if TYPE_CHECKING:  # pragma: no cover
    from ophelian.stores.azure_blob import AzureBlobArtifactStore
    from ophelian.stores.gcs import GCSArtifactStore
    from ophelian.stores.s3 import S3ArtifactStore


def __getattr__(name: str) -> Any:
    if name == "S3ArtifactStore":
        from ophelian.stores.s3 import S3ArtifactStore

        return S3ArtifactStore
    if name == "GCSArtifactStore":
        from ophelian.stores.gcs import GCSArtifactStore

        return GCSArtifactStore
    if name == "AzureBlobArtifactStore":
        from ophelian.stores.azure_blob import AzureBlobArtifactStore

        return AzureBlobArtifactStore
    raise AttributeError(f"module 'ophelian.stores' has no attribute {name!r}")


__all__ = [
    "ArtifactStore",
    "AzureBlobArtifactStore",
    "GCSArtifactStore",
    "LocalArtifactStore",
    "S3ArtifactStore",
]
