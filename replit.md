# Ophelian

> Declarative open-source ML framework. Write your pipeline once — run it anywhere.

## Project Overview

Ophelian is a Python framework being rebuilt from scratch (greenfield reset of an
older PySpark-focused codebase). The new vision is a small, focused, 100% open
source library that lets users define an ML training + serving pipeline as a
declarative DAG, then run it on local Docker, AWS, GCP, Azure, or Kubernetes
without changing the pipeline code.

Target product API:

```python
from ophelian.core import Pipeline, Train, Deploy
from ophelian.envs import AWS

env = AWS(region="us-east-1", instance="g4dn.xlarge")

pipe = Pipeline([
    Train(model="pytorch_resnet"),
    Deploy(replicas=3),
])

pipe.run(env=env)
```

End goal: publish on PyPI as `pip install ophelian`. Differentiator vs Modal /
SkyPilot / Metaflow / ZenML: clean Pythonic DSL + true multi-cloud + open source
+ spot-aware + cost-router (`Auto(cheapest_gpu="A100")`).

## Roadmap (Project Tasks)

The work is split into three sequential milestones, tracked as project tasks:

- **Task #1 — Reset & Foundation v0.1** (MERGED)
  - Greenfield clean-up of the old PySpark code
  - New repo skeleton with `uv` + `hatchling` + `ruff` + `mypy strict` + `pytest`
  - Core DSL (`Pipeline`, `Train`, `Deploy`, `Tune`, `Eval`, `Data`) as
    immutable Pydantic v2 models
  - Graph compiler with `dry-run` mode
  - `Standalone(local=True)` provider with dual-mode execution: local Docker
    when a daemon is reachable, in-process fallback otherwise
  - Model adapters: PyTorch, HuggingFace, sklearn, XGBoost (plugin entry-points)
  - FastAPI inference runtime served from inside the container at `Deploy` time
  - CLI `ophelian` (run, dry-run, version)
  - GitHub Actions CI on Python 3.11 and 3.12
- **Task #2 — AWS Provider Profundo v0.5** (IN PROGRESS, isolated task agent)
  - EC2 / EKS drivers, S3 store, spot-aware with auto-resume
- **Task #3 — Multi-Cloud + Auto Router + v1.0** (PENDING)
  - GCP + Azure providers, `Auto()` cost router, observability,
    three viral demos, mkdocs site, **publish to PyPI as `ophelian`**

## Tech Stack (final, post-Task #1)

- Python 3.11+ (also tested on 3.12)
- Package manager: `uv` (lockfile in `uv.lock`)
- Build backend: `hatchling`
- Validation: `pydantic` v2
- CLI: `typer` + `rich`
- Inference runtime: `fastapi` + `uvicorn`
- Containers: Docker CLI subprocess wrapper (see `ophelian/providers/docker_engine.py`)
- Lint / format / typecheck: `ruff` + `mypy --strict`
- Tests: `pytest` + `pytest-cov`
- Optional adapter extras: `pytorch`, `huggingface`, `sklearn`, `xgboost`, `all`

## Repo Layout

```
ophelian/
├── ophelian/
│   ├── core/              # Pipeline, Train, Deploy, Tune, Eval, Data DSL
│   │                      # plus the graph compiler that turns a Pipeline
│   │                      # into an ExecutionPlan
│   ├── envs/              # Standalone (today); AWS/GCP/Azure/Auto next
│   ├── providers/         # base.py (Provider protocol), docker_engine.py
│   │                      # (CLI subprocess wrapper), standalone.py (executor)
│   ├── models/            # pytorch / huggingface / sklearn / xgboost adapters
│   ├── stores/            # local FS today; S3 / GCS / Blob coming
│   ├── data/              # tiny built-in sample datasets used by examples/tests
│   ├── runtime/           # fastapi_runtime.py (in-container inference server)
│   │                      # and step_runner.py (in-container step entrypoint)
│   ├── cli/               # typer-based `ophelian` CLI
│   └── observability/     # placeholder; structured logging in Task #3
├── tests/                 # 45 unit tests, 1 docker-integration (CI-only), 2 extras-only
├── examples/              # quickstart scripts per workload
├── docs/                  # placeholder; mkdocs site lands in Task #3
├── scripts/post-merge.sh  # uv sync after every task merge
├── pyproject.toml
├── uv.lock
├── README.md
├── CHANGELOG.md
└── LICENSE                # Apache-2.0
```

## Common Commands

```bash
uv sync --extra dev               # install everything for development
uv run pytest                     # run the test suite
uv run ruff check                 # lint
uv run ruff format --check        # format check
uv run mypy ophelian              # strict typecheck
uv run python -m build            # build sdist + wheel
uv run twine check dist/*         # validate the wheel before PyPI upload
```

## Replit Environment Notes

- This Repl is not a Node monorepo anymore — the legacy pnpm workspace files
  (`package.json`, `pnpm-lock.yaml`, `pnpm-workspace.yaml`, `node_modules/`)
  have been removed and the `nodejs-24` Replit module has been uninstalled.
  The only active runtime is `python-3.12`.
- `scripts/post-merge.sh` runs `uv sync --extra dev` after every task merge.
  Timeout is 120 s.
- No long-running workflow is configured: this is a library, not a server. Use
  `uv run pytest` and the example scripts to validate changes.
- **Known cosmetic noise in `.replit`** (cannot be edited directly via tools, no
  functional impact): a leftover `[deployment.postBuild]` running `pnpm store
  prune`, an `[agent] stack = "PNPM_WORKSPACE"` value, and three port mappings
  (5173, 8080, 8081) from previous artifacts. None of these execute because
  there is no `package.json`, no deployment configured, and no service binding
  to those ports. Leave as-is.
- Task #2 is being implemented by an isolated task agent in a separate
  environment. Do not edit AWS/EC2/S3-related files in this Repl until that
  task is merged — changes here will be discarded by the merge.

## User Preferences

- Communicates in Spanish (English fine for code/docs).
- Prefers honest, opinionated technical critique over polite hedging.
- Wants the framework to look like the screenshot in
  `attached_assets/image_1777617847318.png`: minimal Pythonic DSL, multi-cloud,
  100% open source.
- Final distribution target: PyPI (`pip install ophelian`).
