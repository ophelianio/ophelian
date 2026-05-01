"""Entry point used inside Standalone-provider containers.

The Standalone provider, when running in container mode, mounts the
per-step workspace at ``/work`` and writes a small ``step.json`` describing
the node and any upstream artifact paths (also mapped under ``/work``).

The container then runs::

    python -m ophelian.runtime.step_runner /work/step.json

This module loads the spec, dispatches to the same in-process handlers used
by the in-process executor, and writes ``/work/result.json`` for the host
to read back.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from ophelian.core.nodes import Data, Deploy, Eval, Train, Tune
from ophelian.providers import standalone as _standalone

_NODE_TYPES: dict[str, type[Any]] = {
    "data": Data,
    "train": Train,
    "tune": Tune,
    "eval": Eval,
    "deploy": Deploy,
}


def run(spec_path: Path) -> int:
    spec = json.loads(spec_path.read_text())
    kind = spec["kind"]
    node_payload = spec["node"]
    artifacts = spec.get("artifacts", {})
    work_dir = spec_path.parent

    node_cls = _NODE_TYPES[kind]
    node = node_cls(**node_payload)

    # Reuse the in-process handlers; we instantiate a provider in
    # in-process mode rooted at the mounted workspace so paths line up.
    provider = _standalone.StandaloneProvider(
        local=True,
        workspace=work_dir,
        container=False,
    )
    if kind == "data":
        result = provider._handle_data(node, work_dir)
    elif kind == "train":
        result = provider._handle_train(node, work_dir, artifacts)
    elif kind == "tune":
        result = provider._handle_tune(node, work_dir, artifacts)
    elif kind == "eval":
        result = provider._handle_eval(node, work_dir, artifacts)
    elif kind == "deploy":
        # Deploy in container mode is handled directly by the host (we run
        # uvicorn as the container's entrypoint), so the runner should not
        # be asked to handle it. Guard explicitly.
        raise RuntimeError("Deploy steps run uvicorn directly, not via step_runner")
    else:
        raise ValueError(f"Unknown step kind: {kind!r}")

    payload = {
        "name": result.name,
        "kind": result.kind,
        "status": result.status,
        "metrics": result.metrics,
        "artifacts": result.artifacts,
        "info": result.info,
        "error": result.error,
    }
    (work_dir / "result.json").write_text(json.dumps(payload, default=str))
    return 0 if result.status == "success" else 1


def main(argv: list[str] | None = None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])
    if not argv:
        print("usage: python -m ophelian.runtime.step_runner <step.json>", file=sys.stderr)
        return 2
    return run(Path(argv[0]))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
