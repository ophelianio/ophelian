"""Artifact stores — abstractions over where models and reports live.

v0.1 only ships the local filesystem store; cloud stores (S3, GCS, Azure Blob)
follow in the multi-cloud milestone.
"""

from __future__ import annotations

from ophelian.stores.local import LocalArtifactStore

__all__ = ["LocalArtifactStore"]
