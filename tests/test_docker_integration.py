"""End-to-end integration tests against a real Docker daemon.

These tests build the ``ophelian-runtime`` image, run real containers, and
hit the deployed FastAPI app on its published host port. They are
**auto-skipped** when the Docker daemon is not reachable (so the suite
stays green in environments without Docker — including the local Replit
sandbox), and exercised in CI where the daemon is available.
"""

from __future__ import annotations

import time
from pathlib import Path

import httpx
import pytest
from ophelian import Data, Deploy, Pipeline, Train
from ophelian.providers.docker_engine import RealDockerEngine
from ophelian.providers.standalone import StandaloneProvider

_X: list[list[float]] = [[0.0, 0.1], [0.1, 0.2], [0.9, 1.0], [1.0, 0.9]]
_Y: list[int] = [0, 0, 1, 1]


def _docker_available() -> bool:
    return RealDockerEngine().ping()


pytestmark = pytest.mark.skipif(
    not _docker_available(),
    reason="Docker daemon not reachable on this host — skipping integration tests.",
)


def _wait_for_health(url: str, *, timeout: float = 30.0) -> dict[str, str]:
    deadline = time.monotonic() + timeout
    last_exc: Exception | None = None
    while time.monotonic() < deadline:
        try:
            response = httpx.get(url, timeout=2.0)
            if response.status_code == 200:
                return response.json()
        except Exception as exc:  # pragma: no cover - timing dependent
            last_exc = exc
        time.sleep(1.0)
    raise AssertionError(f"Endpoint {url!r} never became ready: {last_exc!r}")


@pytest.fixture()
def provider(tmp_path: Path) -> StandaloneProvider:
    p = StandaloneProvider(local=True, workspace=tmp_path / "ws", container=True)
    yield p
    p.cleanup()


def test_container_pipeline_runs_train_and_deploy(provider: StandaloneProvider) -> None:
    pipeline = Pipeline(
        [
            Data(
                name="ds",
                source="memory://toy",
                format="inline",
                options={"X": _X, "y": _Y},
            ),
            Train(
                name="trainer",
                framework="sklearn",
                model="sklearn.linear_model.LogisticRegression",
                data="ds",
            ),
            Deploy(name="serve", model="trainer", port=8000),
        ],
        name="docker-end-to-end",
    )

    result = pipeline.run(env=provider)
    assert result.succeeded, [s.error for s in result.steps if s.error]

    serve = result.step("serve")
    assert serve.info["mode"] == "container"
    health = _wait_for_health(serve.info["health"])
    assert health["status"] == "ok"

    response = httpx.post(serve.info["predict"], json={"inputs": _X}, timeout=10.0)
    assert response.status_code == 200
    assert response.json()["prediction"] == _Y
