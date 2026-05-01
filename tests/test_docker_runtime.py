"""Container-mode unit tests for the StandaloneProvider.

These tests inject a :class:`FakeDockerEngine` so they can verify the
provider issues the right ``docker build`` / ``docker run`` calls
(image tag, mounts, environment, ports, detach) without requiring a real
Docker daemon. Real-daemon integration coverage lives in
``tests/test_docker_integration.py`` and is auto-skipped when no daemon
is reachable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from ophelian import Data, Deploy, Pipeline, Train
from ophelian.providers.docker_engine import FakeDockerEngine, write_step_result
from ophelian.providers.standalone import StandaloneProvider

_X: list[list[float]] = [[0.0, 0.1], [0.1, 0.2], [0.9, 1.0], [1.0, 0.9]]
_Y: list[int] = [0, 0, 1, 1]


def _data_node(name: str = "ds") -> Data:
    return Data(name=name, source="memory://toy", format="inline", options={"X": _X, "y": _Y})


def _train_node(name: str = "trainer", *, data: str = "ds") -> Train:
    return Train(
        name=name,
        framework="sklearn",
        model="sklearn.linear_model.LogisticRegression",
        data=data,
    )


def _make_step_handler(provider: StandaloneProvider) -> Any:
    """Make a handler that runs the step in-process and writes result.json.

    The FakeDockerEngine doesn't really execute the container — but we want
    the host code path (spec writing, mount mapping, result parsing) to be
    exercised end-to-end. So we install a handler that, on each ``docker
    run``, reads the spec from the mounted workspace, runs the appropriate
    in-process handler, and writes ``result.json`` back.
    """

    def handler(payload: dict[str, Any]) -> dict[str, Any]:
        # Deploy containers run uvicorn directly — no spec/result.json round-trip.
        if payload["detach"] or payload["command"][0:3] == ["python", "-m", "uvicorn"]:
            return {"exit_code": 0, "logs": ""}
        # Locate the host workspace for this step from the mount table.
        host_work = next(
            Path(host) for host, container in payload["volumes"].items() if container == "/work"
        )
        spec = json.loads((host_work / "step.json").read_text())
        kind = spec["kind"]
        node_payload = spec["node"]
        # Translate container artifact paths back to host paths so the
        # in-process handlers (which run on the host) can read them.
        upstream: dict[str, dict[str, str]] = {}
        for name, art in (spec.get("artifacts") or {}).items():
            mount = f"/work-{name}"
            host_dir = next(Path(h) for h, c in payload["volumes"].items() if c == mount)
            upstream[name] = {k: v.replace(mount, str(host_dir)) for k, v in art.items()}
        from ophelian.core.nodes import Train as _Train

        if kind == "data":
            node = Data(**node_payload)
            result = provider._handle_data(node, host_work)
        elif kind == "train":
            train_node = _Train(**node_payload)
            result = provider._handle_train(train_node, host_work, upstream)
        else:
            raise AssertionError(f"unexpected kind: {kind!r}")
        # Rewrite host paths back to container paths in the result so the
        # provider's path translation (container → host) is exercised too.
        rewritten_artifacts = {
            k: v.replace(str(host_work), "/work") for k, v in result.artifacts.items()
        }
        write_step_result(
            host_work,
            {
                "name": result.name,
                "kind": result.kind,
                "status": result.status,
                "metrics": result.metrics,
                "artifacts": rewritten_artifacts,
                "info": result.info,
                "error": result.error,
            },
        )
        return {"exit_code": 0, "logs": "ok"}

    return handler


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    return tmp_path / "ws"


def test_provider_picks_container_mode_when_engine_pings_true(workspace: Path) -> None:
    engine = FakeDockerEngine()
    provider = StandaloneProvider(local=True, workspace=workspace, docker_engine=engine)
    assert provider.mode == "container"


def test_provider_falls_back_to_inprocess_when_engine_unreachable(workspace: Path) -> None:
    engine = FakeDockerEngine()
    engine.set_reachable(False)
    provider = StandaloneProvider(local=True, workspace=workspace, docker_engine=engine)
    assert provider.mode == "inprocess"


def test_container_true_with_no_daemon_raises(workspace: Path) -> None:
    from ophelian.providers.docker_engine import DockerUnavailableError

    engine = FakeDockerEngine()
    engine.set_reachable(False)
    with pytest.raises(DockerUnavailableError):
        StandaloneProvider(local=True, workspace=workspace, docker_engine=engine, container=True)


def test_container_pipeline_records_build_and_run_calls(workspace: Path) -> None:
    engine = FakeDockerEngine()
    provider = StandaloneProvider(local=True, workspace=workspace, docker_engine=engine)
    engine.register_handler(provider.runtime_image, _make_step_handler(provider))

    pipeline = Pipeline([_data_node(), _train_node()], name="container-train")
    result = pipeline.run(env=provider)

    assert result.succeeded, [s.error for s in result.steps if s.error]

    builds = [a for a in engine.actions if a.kind == "build"]
    runs = [a for a in engine.actions if a.kind == "run"]

    # Image is built exactly once (idempotent across all step containers).
    assert len(builds) == 1
    assert builds[0].payload["tag"] == provider.runtime_image
    assert "FROM python" in builds[0].payload["dockerfile"]

    # Two one-shot containers — one per step.
    assert len(runs) == 2
    for run in runs:
        assert run.payload["image"] == provider.runtime_image
        assert run.payload["command"][:3] == ["python", "-m", "ophelian.runtime.step_runner"]
        assert run.payload["detach"] is False
        assert "/work" in run.payload["volumes"].values()

    # The Train container also mounts the upstream Data step's directory so
    # the runner can read the dataset through the in-container path.
    train_run = runs[1].payload
    assert any(container.startswith("/work-ds") for container in train_run["volumes"].values())


def test_container_deploy_runs_detached_with_published_port(workspace: Path) -> None:
    engine = FakeDockerEngine()
    provider = StandaloneProvider(local=True, workspace=workspace, docker_engine=engine)
    engine.register_handler(provider.runtime_image, _make_step_handler(provider))

    pipeline = Pipeline(
        [
            _data_node(),
            _train_node(),
            Deploy(name="serve", model="trainer", port=9000),
        ],
        name="container-deploy",
    )
    result = pipeline.run(env=provider)
    assert result.succeeded, [s.error for s in result.steps if s.error]

    deploy_runs = [a for a in engine.actions if a.kind == "run" and a.payload.get("detach") is True]
    assert len(deploy_runs) == 1
    payload = deploy_runs[0].payload
    assert payload["command"][0:3] == ["python", "-m", "uvicorn"]
    # Uvicorn must be invoked with the env-driven factory so the
    # container starts without keyword-arg injection.
    assert "ophelian.runtime.fastapi_runtime:app_from_env" in payload["command"]
    assert payload["environment"]["OPHELIAN_FRAMEWORK"] == "sklearn"
    assert payload["environment"]["OPHELIAN_MODEL_PATH"] == "/model"
    # The mounted host directory must be the model dir itself (containing
    # ``ophelian.json``), not its parent.
    [(host_model_dir, container_model_dir)] = [
        (h, c) for h, c in payload["volumes"].items() if c == "/model"
    ]
    assert (Path(host_model_dir) / "ophelian.json").exists()
    assert container_model_dir == "/model"
    # Container port from the DSL is published to a host port.
    assert 9000 in payload["ports"].values()
    [host_port] = list(payload["ports"].keys())

    info = result.step("serve").info
    assert info["mode"] == "container"
    assert info["host_port"] == host_port
    assert info["container_port"] == 9000
    assert info["served"] is True
    assert info["predict"].endswith("/predict")


def test_runtime_image_extras_extend_with_pipeline_frameworks(workspace: Path) -> None:
    """The runtime image must install the extras for every framework the
    pipeline actually uses, not just the default ``sklearn`` extra. This
    pins the fix for the reviewer's report that pytorch/xgboost/hf
    pipelines couldn't run in container mode."""
    engine = FakeDockerEngine()
    provider = StandaloneProvider(
        local=True,
        workspace=workspace,
        docker_engine=engine,
        runtime_extras=("sklearn",),
    )
    engine.register_handler(provider.runtime_image, _make_step_handler(provider))

    pipeline = Pipeline(
        [
            _data_node(),
            Train(
                name="trainer",
                framework="xgboost",
                model="xgboost.XGBClassifier",
                data="ds",
            ),
        ],
        name="extras-extend",
    )
    # Don't actually run training (no xgboost in handler) — we only care
    # that ``execute`` extended the extras and that the next image build
    # would install xgboost. Inspect through a direct call instead.
    from ophelian.core.compiler import GraphCompiler

    provider._extend_runtime_extras_from_plan(GraphCompiler().compile(pipeline))
    provider._ensure_runtime_image()

    [build] = [a for a in engine.actions if a.kind == "build"]
    dockerfile = build.payload["dockerfile"]
    assert "[sklearn,xgboost]" in dockerfile


def test_provider_cleanup_stops_and_removes_deploy_containers(workspace: Path) -> None:
    engine = FakeDockerEngine()
    provider = StandaloneProvider(local=True, workspace=workspace, docker_engine=engine)
    engine.register_handler(provider.runtime_image, _make_step_handler(provider))

    pipeline = Pipeline(
        [_data_node(), _train_node(), Deploy(name="serve", model="trainer", port=9100)],
        name="cleanup",
    )
    pipeline.run(env=provider)
    assert provider.containers, "Deploy must register a container handle"
    deploy_id = provider.containers["serve"].container_id

    provider.cleanup()
    assert deploy_id in engine.stopped
    assert deploy_id in engine.removed
