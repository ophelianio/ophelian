"""Common interface every artifact store implements.

`Provider` implementations and the `Data` loader treat artifact storage
through this `ArtifactStore` Protocol so they can stay agnostic of where
bytes actually live — local filesystem today, S3 / GCS / Azure Blob next.

The contract is intentionally minimal — put / get / exists / delete are
enough to back checkpointing, model persistence and the resume logic the
spot-instance handler relies on.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class ArtifactStore(Protocol):
    """Read/write a flat namespace of byte-stream artifacts."""

    def put(self, key: str, source: str | Path) -> str:
        """Upload *source* (file or directory) to *key*; return the canonical URI."""
        ...

    def get(self, key: str, destination: str | Path | None = None) -> Path:
        """Materialise *key* on local disk and return the local path.

        If *destination* is provided, write there; otherwise the store may
        return an existing local path (useful for the LocalArtifactStore that
        is already on-disk).
        """
        ...

    def exists(self, key: str) -> bool:
        """Return True iff *key* has been written."""
        ...

    def delete(self, key: str) -> None:
        """Remove *key* (idempotent — missing keys are a no-op)."""
        ...

    def list(self, prefix: str = "") -> Iterable[str]:
        """Yield keys with the given prefix (empty string lists everything)."""
        ...

    def uri(self, key: str) -> str:
        """Return the canonical URI for *key* (``file://``, ``s3://``, ...)."""
        ...


__all__ = ["ArtifactStore"]
