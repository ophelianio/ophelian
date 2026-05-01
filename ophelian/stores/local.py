"""Local-filesystem artifact store."""

from __future__ import annotations

import shutil
from collections.abc import Iterable
from pathlib import Path


class LocalArtifactStore:
    """Read/write artifacts under a single root directory.

    Implements the :class:`~ophelian.stores.base.ArtifactStore` Protocol.
    """

    scheme: str = "file"

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        return self._root

    def path_for(self, key: str) -> Path:
        target = self._root / key
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def put(self, key: str, source: str | Path) -> str:
        destination = self.path_for(key)
        source_path = Path(source)
        if source_path.is_dir():
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(source_path, destination)
        else:
            shutil.copy2(source_path, destination)
        return self.uri(key)

    def get(self, key: str, destination: str | Path | None = None) -> Path:
        path = self._root / key
        if not path.exists():
            raise FileNotFoundError(f"No artifact at {key}")
        if destination is None:
            return path
        dst = Path(destination)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if path.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(path, dst)
        else:
            shutil.copy2(path, dst)
        return dst

    def exists(self, key: str) -> bool:
        return (self._root / key).exists()

    def delete(self, key: str) -> None:
        target = self._root / key
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()

    def list(self, prefix: str = "") -> Iterable[str]:
        base = self._root / prefix if prefix else self._root
        if not base.exists():
            return []
        keys: list[str] = []
        for path in base.rglob("*"):
            if path.is_file():
                keys.append(str(path.relative_to(self._root)))
        return keys

    def uri(self, key: str) -> str:
        return (self._root / key).as_uri()
