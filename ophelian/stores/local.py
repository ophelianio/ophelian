"""Local-filesystem artifact store."""

from __future__ import annotations

import shutil
from pathlib import Path


class LocalArtifactStore:
    """Read/write artifacts under a single root directory."""

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

    def put(self, key: str, source: str | Path) -> Path:
        destination = self.path_for(key)
        source_path = Path(source)
        if source_path.is_dir():
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(source_path, destination)
        else:
            shutil.copy2(source_path, destination)
        return destination

    def get(self, key: str) -> Path:
        path = self._root / key
        if not path.exists():
            raise FileNotFoundError(f"No artifact at {key}")
        return path

    def exists(self, key: str) -> bool:
        return (self._root / key).exists()

    def delete(self, key: str) -> None:
        target = self._root / key
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
