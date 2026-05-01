# Ophelian

> Declarative ML framework. Write your pipeline once — run it anywhere.

Ophelian is a small, opinionated Python framework for taking ML / AI prototypes
to production without rewriting them every time the runtime changes. You write
a **Pipeline** with declarative nodes (`Data`, `Train`, `Tune`, `Eval`,
`Deploy`) and the framework compiles it into an execution plan that an **env**
(a backend) knows how to run. Today there are two envs: `Standalone(local=True)`
runs every step **in a local Docker container** when a Docker daemon is
available — and falls back to running the same logic in-process when it isn't,
so notebooks, CI without Docker, and quick experiments still work without any
configuration. `AWS(region=..., instance=..., spot=True)` runs the same
pipeline on real EC2 (or EKS) workers with S3-backed artifacts and automatic
resume on spot interruption — see [`docs/aws.md`](docs/aws.md). GCP / Azure
envs land in upcoming milestones; pipelines do not change.

## Vision

```text
              ┌──────────────────────────┐
              │     Pipeline (DSL)       │   declarative, immutable
              │  Train · Tune · Eval ·   │   Pydantic v2 models
              │       Deploy · Data      │
              └────────────┬─────────────┘
                           │ compile
                           ▼
              ┌──────────────────────────┐
              │   GraphCompiler →        │   topological order
              │   ExecutionPlan          │   dry-run friendly
              └────────────┬─────────────┘
                           │ execute
                           ▼
   ┌───────────────────────┴───────────────────────────┐
   │                    Provider                       │
   ├───────────────────────────────────────────────────┤
   │   Standalone(local=True)  ← v0.1 (local Docker;   │
   │                             auto-fallback to       │
   │                             in-process if no       │
   │                             daemon)                │
   │   AWS(region=…, spot=True) ← v0.5 (EC2/EKS, S3,   │
   │                              spot-resume)          │
   │   GCP / Azure             ← v0.6+                 │
   │   Multi-cloud + auto-router ← v1.0                │
   └───────────────────────────────────────────────────┘
```

## Install

```bash
pip install -e .                  # core framework
pip install -e '.[sklearn]'       # add a specific framework adapter
pip install -e '.[aws]'           # AWS env (boto3 + paramiko)
pip install -e '.[aws,eks]'       # AWS env + Kubernetes/EKS backend
pip install -e '.[all,dev]'       # everything, including dev tooling
```

Requires Python 3.11 or 3.12.

## Quickstart — sklearn (fully runnable)

This is the canonical end-to-end example and is exercised by CI:

```python
from ophelian import Data, Deploy, Pipeline, Standalone, Train

pipe = Pipeline([
    Data(name="ds", source="synthetic://iris"),
    Train(
        name="trainer",
        framework="sklearn",
        model="sklearn.linear_model.LogisticRegression",
        data="ds",
        hyperparameters={"max_iter": 200},
    ),
    Deploy(name="serve", model="trainer", port=8080),
])

if __name__ == "__main__":
    result = pipe.run(env=Standalone(local=True))
    print("Endpoint:", result.step("serve").info["predict"])
```

Run it:

```bash
ophelian run examples/sklearn_pipeline.py
```

## Quickstart — AWS

The same pipeline runs on real EC2 by swapping the env. Artifacts are written
to S3, the worker is torn down in a `finally` block, and spot interruptions
are checkpointed so you can resume from where you left off.

```python
from ophelian import AWS, Data, Eval, Pipeline, Train

env = AWS(
    region="us-east-1",
    instance="g4dn.xlarge",
    spot=True,
    artifact_bucket="my-ophelian-bucket",
    iam_role="ophelian-worker",
)

result = Pipeline([
    Data(name="ds", source="s3://my-ophelian-bucket/datasets/iris.jsonl", format="jsonl"),
    Train(name="trainer", framework="sklearn",
          model="sklearn.linear_model.LogisticRegression", data="ds"),
    Eval(name="ev", model="trainer", data="ds", metrics=("accuracy",)),
]).run(env=env)

if not result.succeeded:
    failed = next(s for s in result.steps if s.status == "failed")
    if failed.info.get("resumable"):
        env = env.with_resume(run_id=failed.info["run_id"])
        result = Pipeline([...]).run(env=env)   # picks up after the last completed step
```

See [`docs/aws.md`](docs/aws.md) for credentials, the minimal IAM policy,
the spot/resume protocol, the optional EKS backend, and per-workload cost
estimates. End-to-end demos for XGBoost, ResNet, and HuggingFace LLMs live
in [`examples/aws/`](examples/aws/).

## Other frameworks

The same `Pipeline` shape works with the other built-in adapters. They are all
covered by save/load round-trip tests in CI; the snippets below assume you have
the optional deps installed (`pip install -e '.[pytorch]'`, `'.[xgboost]'`,
`'.[huggingface]'`).

### XGBoost (tabular)

```python
from ophelian import Data, Pipeline, Standalone, Train

pipe = Pipeline([
    Data(
        name="ds",
        source="inline://",
        format="inline",
        options={
            "X": [[0, 0], [0, 1], [1, 0], [1, 1]],
            "y": [0, 1, 1, 0],
        },
    ),
    Train(
        name="trainer",
        framework="xgboost",
        model="XGBClassifier",
        data="ds",
        hyperparameters={"max_depth": 3, "n_estimators": 30, "verbosity": 0},
    ),
])

pipe.run(env=Standalone(local=True))
```

### PyTorch (tabular regression)

The PyTorch adapter accepts a fully-qualified class path and runs a small
tabular training loop (MSE for float targets, cross-entropy for int targets):

```python
from ophelian import Data, Pipeline, Standalone, Train

pipe = Pipeline([
    Data(
        name="ds",
        source="inline://",
        format="inline",
        options={"X": [[1.0], [2.0], [3.0]], "y": [2.0, 4.0, 6.0]},
    ),
    Train(
        name="trainer",
        framework="pytorch",
        model="torch.nn.Linear",
        data="ds",
        hyperparameters={"in_features": 1, "out_features": 1, "lr": 0.05},
        epochs=200,
    ),
])

pipe.run(env=Standalone(local=True))
```

### HuggingFace

The HuggingFace adapter is a thin wrapper around `transformers`. It is
**network- and disk-heavy** (downloads model weights and tokenizers), so it is
not exercised end-to-end in CI. With `transformers` installed it follows the
same shape as the other adapters; pass a hub model id as `model=` and a
`Dataset` (or a dict your tokenizer understands) as `data=`.

## Dry-run a pipeline

You can inspect the compiled plan without executing anything:

```bash
ophelian dry-run examples/sklearn_pipeline.py
```

Or programmatically:

```python
pipe.dry_run()
```

## CLI

```bash
ophelian version
ophelian dry-run path/to/pipeline.py
ophelian run     path/to/pipeline.py
```

## Developer setup

```bash
pip install -e '.[all,dev]'
pre-commit install
pytest
ruff check .
mypy ophelian
```

CI runs lint (ruff), typecheck (mypy strict) and tests (pytest) on every PR
across Python 3.11 and 3.12.

## Architecture at a glance

| Subpackage              | Responsibility                                              |
|-------------------------|-------------------------------------------------------------|
| `ophelian.core`         | Declarative DSL nodes + graph compiler                      |
| `ophelian.envs`         | Public env constructors (`Standalone(...)`)                 |
| `ophelian.providers`    | Backend implementations (Standalone in-process today)       |
| `ophelian.models`       | Framework adapters + plugin registry                        |
| `ophelian.runtime`      | Inference runtimes (FastAPI today)                          |
| `ophelian.stores`       | Artifact stores (local FS today, cloud later)               |
| `ophelian.data`         | Data loaders (inline, synthetic, csv/json/jsonl/parquet)    |
| `ophelian.cli`          | `ophelian` command                                          |
| `ophelian.observability`| Logging surface (OTel later)                                |

## Execution modes

The Standalone provider exposes two execution modes selected by the
`container` argument:

| `container=` | Behavior                                                                    |
|--------------|-----------------------------------------------------------------------------|
| `True`       | Force Docker. Each step runs in an `ophelian-runtime` container; `Deploy` runs detached with its port published. Raises if no daemon is reachable. |
| `False`      | Force in-process. Useful in notebooks, on machines without Docker, and in tests where you want to assert against the produced FastAPI app via `TestClient`. |
| `"auto"` (default) | Probe the daemon and pick `container` if reachable, otherwise `inprocess`. |

In container mode `result.step("serve").info` includes `container_id`,
`host_port`, `container_port`, and the real `predict`/`health` URLs bound
to the host. CI exercises this end-to-end: `tests/test_docker_integration.py`
builds the runtime image, starts a real container, and `curl`s `/health`
and `/predict` against the host port.

## Status

This is the **v0.1 foundation**. It establishes the API and architecture
end-to-end on a single machine — in a real Docker container when a daemon
is available, and in-process as a transparent fallback otherwise. The
Standalone provider trains via the model adapters, persists artifacts to a
local store, and serves `Deploy` steps as a real FastAPI app (mounted in
your test client when in-process; running in a detached container with a
published port when in Docker). The next milestones add real cloud
execution (AWS first), spot/auto-scaling, multi-cloud and a cost-aware
auto-router.

## License

Apache-2.0 — see `LICENSE`.
