"""Artifact stores — abstractions over where models, datasets and reports live.

The local filesystem store is always available. The S3 store is part of the
optional :pypi:`boto3` dependency and is lazy-imported so installing
``ophelian`` without the ``aws`` extra still works.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ophelian.stores.base import ArtifactStore
from ophelian.stores.local import LocalArtifactStore

if TYPE_CHECKING:  # pragma: no cover
    from ophelian.stores.s3 import S3ArtifactStore


def __getattr__(name: str) -> Any:
    if name == "S3ArtifactStore":
        from ophelian.stores.s3 import S3ArtifactStore

        return S3ArtifactStore
    raise AttributeError(f"module 'ophelian.stores' has no attribute {name!r}")


__all__ = ["ArtifactStore", "LocalArtifactStore", "S3ArtifactStore"]
