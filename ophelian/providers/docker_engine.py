"""Docker engine abstractions used by the Standalone provider.

The provider talks to Docker through a small :class:`DockerEngine` protocol so
that:

* :class:`RealDockerEngine` shells out to the ``docker`` CLI (works on any
  machine with a running Docker daemon).
* :class:`FakeDockerEngine` records the calls that would have been made,
  enabling unit tests that exercise the provider's containerization logic
  without requiring a daemon.

The engine is intentionally minimal — it's only what v0.1 needs: build an
image once, run a one-shot container that mounts the workspace, run a
detached container with a published port for ``Deploy`` steps, and tear them
down.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@dataclass
class ContainerHandle:
    """Outcome of running a container."""

    container_id: str
    exit_code: int = 0
    logs: str = ""
    detached: bool = False
    published_ports: dict[int, int] = field(default_factory=dict)


@runtime_checkable
class DockerEngine(Protocol):
    """Minimal contract the Standalone provider relies on."""

    def ping(self) -> bool: ...

    def build_image(self, *, context: Path, dockerfile: str, tag: str) -> str: ...

    def run_container(
        self,
        *,
        image: str,
        command: list[str],
        volumes: Mapping[Path, str] | None = None,
        environment: Mapping[str, str] | None = None,
        ports: Mapping[int, int] | None = None,
        detach: bool = False,
        name: str | None = None,
        timeout: float | None = None,
    ) -> ContainerHandle: ...

    def stop(self, container_id: str) -> None: ...

    def remove(self, container_id: str, *, force: bool = False) -> None: ...

    def logs(self, container_id: str) -> str: ...


# ---------------------------------------------------------------------------
# Real engine — subprocess wrapper around the docker CLI.
# ---------------------------------------------------------------------------


class DockerUnavailableError(RuntimeError):
    """Raised when the docker CLI or daemon is not reachable."""


class RealDockerEngine:
    """A :class:`DockerEngine` backed by the local ``docker`` CLI."""

    def __init__(self, *, binary: str = "docker") -> None:
        self._binary = binary

    def ping(self) -> bool:
        if shutil.which(self._binary) is None:
            return False
        try:
            result = subprocess.run(
                [self._binary, "info", "--format", "{{.ServerVersion}}"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        return result.returncode == 0 and bool(result.stdout.strip())

    def build_image(self, *, context: Path, dockerfile: str, tag: str) -> str:
        dockerfile_path = context / "Dockerfile"
        dockerfile_path.write_text(dockerfile)
        proc = self._run_cli(
            ["build", "-t", tag, "-f", str(dockerfile_path), str(context)],
            check=True,
        )
        # docker build prints the image id at the end of stdout when using
        # the legacy builder; either way the tag is what callers reference.
        del proc
        return tag

    def run_container(
        self,
        *,
        image: str,
        command: list[str],
        volumes: Mapping[Path, str] | None = None,
        environment: Mapping[str, str] | None = None,
        ports: Mapping[int, int] | None = None,
        detach: bool = False,
        name: str | None = None,
        timeout: float | None = None,
    ) -> ContainerHandle:
        argv: list[str] = ["run"]
        if detach:
            argv.append("-d")
        else:
            argv.append("--rm")
        if name:
            argv += ["--name", name]
        for host_path, container_path in (volumes or {}).items():
            argv += ["-v", f"{Path(host_path).resolve()}:{container_path}"]
        for key, value in (environment or {}).items():
            argv += ["-e", f"{key}={value}"]
        published: dict[int, int] = {}
        for host_port, container_port in (ports or {}).items():
            argv += ["-p", f"{host_port}:{container_port}"]
            published[host_port] = container_port
        argv += [image, *command]
        proc = self._run_cli(argv, check=False, timeout=timeout)
        container_id = proc.stdout.strip().splitlines()[-1] if detach and proc.stdout else ""
        return ContainerHandle(
            container_id=container_id or (name or ""),
            exit_code=proc.returncode,
            logs=(proc.stdout or "") + (proc.stderr or ""),
            detached=detach,
            published_ports=published,
        )

    def stop(self, container_id: str) -> None:
        if not container_id:
            return
        self._run_cli(["stop", container_id], check=False, timeout=15)

    def remove(self, container_id: str, *, force: bool = False) -> None:
        if not container_id:
            return
        argv = ["rm"]
        if force:
            argv.append("-f")
        argv.append(container_id)
        self._run_cli(argv, check=False, timeout=15)

    def logs(self, container_id: str) -> str:
        if not container_id:
            return ""
        proc = self._run_cli(["logs", container_id], check=False, timeout=10)
        return (proc.stdout or "") + (proc.stderr or "")

    def _run_cli(
        self,
        argv: list[str],
        *,
        check: bool,
        timeout: float | None = None,
    ) -> subprocess.CompletedProcess[str]:
        if shutil.which(self._binary) is None:
            raise DockerUnavailableError(f"{self._binary!r} CLI not found on PATH")
        try:
            proc = subprocess.run(
                [self._binary, *argv],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise DockerUnavailableError(
                f"docker {' '.join(argv)} timed out after {timeout}s"
            ) from exc
        if check and proc.returncode != 0:
            raise DockerUnavailableError(
                f"docker {' '.join(argv)} failed (exit={proc.returncode}): "
                f"{proc.stderr.strip() or proc.stdout.strip()}"
            )
        return proc


# ---------------------------------------------------------------------------
# Fake engine — used by unit tests to verify *what* the provider would run.
# ---------------------------------------------------------------------------


@dataclass
class _RecordedAction:
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)


class FakeDockerEngine:
    """In-memory :class:`DockerEngine` for unit tests.

    Records build/run/stop calls and lets tests register canned per-image
    side effects (e.g. write a result file into the mounted workspace) so
    the provider can be exercised end-to-end without a real daemon.
    """

    def __init__(self) -> None:
        self.actions: list[_RecordedAction] = []
        self._next_id = 0
        self._handlers: dict[str, list[Any]] = {}
        self._reachable = True
        self.removed: list[str] = []
        self.stopped: list[str] = []
        self._logs: dict[str, str] = {}

    def set_reachable(self, reachable: bool) -> None:
        self._reachable = reachable

    def register_handler(self, image: str, handler: Any) -> None:
        """Register a callable invoked on every ``run_container`` for *image*.

        The handler receives the full payload dict (image, command, volumes,
        environment, ports, detach) and may return a partial dict that will
        be merged into the resulting :class:`ContainerHandle` (``exit_code``,
        ``logs``). It can also write files into the mounted volumes to
        simulate the container producing artifacts.
        """
        self._handlers.setdefault(image, []).append(handler)

    def ping(self) -> bool:
        return self._reachable

    def _new_id(self, prefix: str) -> str:
        self._next_id += 1
        return f"{prefix}-{self._next_id:04d}"

    def build_image(self, *, context: Path, dockerfile: str, tag: str) -> str:
        self.actions.append(
            _RecordedAction(
                kind="build",
                payload={"context": str(context), "dockerfile": dockerfile, "tag": tag},
            )
        )
        return tag

    def run_container(
        self,
        *,
        image: str,
        command: list[str],
        volumes: Mapping[Path, str] | None = None,
        environment: Mapping[str, str] | None = None,
        ports: Mapping[int, int] | None = None,
        detach: bool = False,
        name: str | None = None,
        timeout: float | None = None,
    ) -> ContainerHandle:
        del timeout
        container_id = name or self._new_id("ctr")
        payload = {
            "image": image,
            "command": list(command),
            "volumes": {str(k): v for k, v in (volumes or {}).items()},
            "environment": dict(environment or {}),
            "ports": dict(ports or {}),
            "detach": detach,
            "container": container_id,
        }
        self.actions.append(_RecordedAction(kind="run", payload=payload))
        result: dict[str, Any] = {"exit_code": 0, "logs": ""}
        for handler in self._handlers.get(image, []):
            override = handler(payload) or {}
            result.update(override)
        self._logs[container_id] = result.get("logs", "")
        return ContainerHandle(
            container_id=container_id,
            exit_code=int(result.get("exit_code", 0)),
            logs=result.get("logs", ""),
            detached=detach,
            published_ports=dict(ports or {}),
        )

    def stop(self, container_id: str) -> None:
        self.stopped.append(container_id)
        self.actions.append(_RecordedAction(kind="stop", payload={"container": container_id}))

    def remove(self, container_id: str, *, force: bool = False) -> None:
        self.removed.append(container_id)
        self.actions.append(
            _RecordedAction(kind="remove", payload={"container": container_id, "force": force})
        )

    def logs(self, container_id: str) -> str:
        return self._logs.get(container_id, "")


def write_step_result(workspace: Path, payload: dict[str, Any]) -> None:
    """Helper used by FakeDockerEngine handlers to simulate step output."""
    (workspace / "result.json").write_text(json.dumps(payload, default=str))


__all__ = [
    "ContainerHandle",
    "DockerEngine",
    "DockerUnavailableError",
    "FakeDockerEngine",
    "RealDockerEngine",
    "write_step_result",
]
