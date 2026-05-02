"""Local-filesystem artifact store."""

from __future__ import annotations

import os
import shutil
from collections.abc import Iterable
from pathlib import Path


class LocalArtifactStore:
    """Read/write artifacts under a single root directory.

    Implements the :class:`~ophelian.stores.base.ArtifactStore` Protocol.

    Keys are treated as **relative paths inside the configured root**.
    Any key that resolves outside the root — whether through ``..``
    components, an absolute path, or a pre-existing symlink under the
    root — is rejected with :class:`ValueError`. This is enforced for
    both reads and writes so the store cannot be coerced into reading
    or writing arbitrary files on the host.
    """

    scheme: str = "file"

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        # Resolve once at construction so symlinks in the prefix don't
        # repeatedly skew containment checks.
        self._root_resolved = self._root.resolve(strict=True)

    @property
    def root(self) -> Path:
        return self._root

    def _safe_path(self, key: str, *, must_exist: bool = False) -> Path:
        """Translate a user-supplied key to an absolute path inside the
        root, or raise ``ValueError`` if the key escapes.

        Containment is checked against the **resolved** target so that
        ``..`` segments, absolute paths, and pre-existing symlinks that
        point outside the root are all caught. When the target does not
        yet exist (write path) we resolve its closest existing ancestor,
        which still rejects ``..`` escapes without requiring the file
        to be present.
        """
        if not key or key in {".", "/"}:
            raise ValueError(f"Invalid artifact key: {key!r}")
        # Reject absolute keys explicitly — `Path(root) / "/abs"` would
        # silently drop the root on POSIX.
        if os.path.isabs(key) or key.startswith(("/", "\\")):
            raise ValueError(f"Artifact key must be relative, got {key!r}")

        candidate = (self._root / key).absolute()

        # ``os.path.realpath`` resolves *all* symlinks along the path,
        # including for components that do not yet exist (the missing
        # tail is left literal). That lets us catch a symlink under the
        # root that points outside, even when the leaf write target has
        # not been created yet.
        resolved = Path(os.path.realpath(candidate))

        try:
            resolved.relative_to(self._root_resolved)
        except ValueError as exc:
            raise ValueError(
                f"Artifact key {key!r} escapes store root {self._root_resolved}"
            ) from exc

        if must_exist and not resolved.exists():
            raise FileNotFoundError(f"No artifact at {key}")
        return resolved

    def path_for(self, key: str) -> Path:
        target = self._safe_path(key)
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
        path = self._safe_path(key, must_exist=True)
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
        try:
            return self._safe_path(key).exists()
        except (ValueError, FileNotFoundError):
            return False

    def delete(self, key: str) -> None:
        try:
            target = self._safe_path(key)
        except (ValueError, FileNotFoundError):
            return
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()

    def list(self, prefix: str = "") -> Iterable[str]:
        if prefix:
            try:
                base = self._safe_path(prefix)
            except (ValueError, FileNotFoundError):
                return []
        else:
            base = self._root_resolved
        if not base.exists():
            return []
        keys: list[str] = []
        for path in base.rglob("*"):
            if path.is_file():
                keys.append(str(path.relative_to(self._root_resolved)))
        return keys

    def uri(self, key: str) -> str:
        return self._safe_path(key).as_uri()

    def put_bytes(self, key: str, payload: bytes) -> str:
        destination = self.path_for(key)
        destination.write_bytes(payload)
        return self.uri(key)

    def get_bytes(self, key: str) -> bytes:
        path = self._safe_path(key, must_exist=True)
        if path.is_dir():
            raise FileNotFoundError(f"No artifact at {key}")
        return path.read_bytes()
