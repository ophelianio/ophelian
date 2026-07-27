# Tutorial: Test Ophelian and Serve a Live Prediction

This tutorial sets up Ophelian from a fresh clone, runs its test suite,
and then serves a model over HTTP and sends it a real prediction —
**using Ophelian's own commands, not hand-written plumbing**.

Everything runs **locally**. No cloud account or credentials needed.
Every command below was actually run; the outputs are real.

---

## 0. What you need

- `git`
- [`uv`](https://docs.astral.sh/uv/) — installs Python and dependencies for you.
- No cloud keys. Ophelian is local-first: every cloud path has a local
  mirror, so the whole framework works offline.

> **Python version.** Ophelian supports Python **3.11, 3.12, 3.13**. If
> your system Python is newer (e.g. 3.14), let `uv` install a supported
> one — step 2 does that.

---

## 1. Get the code

```bash
git clone https://github.com/ophelianio/ophelian.git
cd ophelian
git checkout v1.1.1        # the version published on PyPI
```

---

## 2. Create the environment

```bash
uv python install 3.12
uv python pin 3.12
uv sync --frozen --extra dev
```

`uv sync` reads `uv.lock`, so you get the exact same package versions
every time. `--extra dev` adds the test and lint tools.

Check it:

```bash
uv run ophelian version          # ophelian 1.1.1
uv run ophelian --help           # version | dry-run | run | costs
```

---

## 3. Run the tests

```bash
uv run pytest -q
# 406 passed, 4 skipped
```

The 4 skips are **optional**, not failures: AWS integration (needs real
keys), Docker integration (needs the Docker daemon), and two model tests
that need `torch` / `xgboost`.

Unlock the two model tests by adding the ML extras:

```bash
uv sync --frozen --extra dev --extra pytorch --extra xgboost --extra huggingface
uv run pytest -q
# 409 passed, 2 skipped
```

Now only the AWS and Docker integration tests are skipped — expected on
a laptop without those services.

---

## 4. The mental model

An Ophelian program is a **`Pipeline`** of steps (`Data`, `Train`,
`Eval`, `Deploy`, `Tune`). You pick an **`env`** and call `run`. The
same pipeline runs locally or on any cloud — only the `env` changes.

```python
from ophelian import Pipeline, Data, Train, Deploy, Standalone

pipe = Pipeline([
    Data(name="ds", source="synthetic://iris", format="synthetic",
         options={"name": "iris"}),
    Train(name="trainer", framework="sklearn",
          model="sklearn.linear_model.LogisticRegression",
          data="ds", hyperparameters={"max_iter": 200}),
    Deploy(name="serve", model="trainer", port=8080),
], name="sklearn-quickstart")
```

See the plan first — this never runs anything:

```bash
uv run ophelian dry-run examples/sklearn_pipeline.py -a pipe
```

```
# │ Node    │ Kind   │ Depends on
0 │ ds      │ data   │ —
1 │ trainer │ train  │ ds
2 │ serve   │ deploy │ trainer
```

---

## 5. Serve a model and predict — the idiomatic way

The goal: a **live HTTP server** answering a **real prediction**. Do
**not** launch uvicorn yourself or copy model files around — Ophelian's
`Deploy` step does all of that. There are two idiomatic ways.

### Way A — one CLI command (recommended)

`ophelian run` with `--serve` runs the pipeline and, for every `Deploy`
step, starts a real uvicorn server bound to that step's port:

```bash
uv run ophelian run examples/sklearn_pipeline.py -a pipe --serve
```

You get the per-step status table, and the `serve` step's server keeps
running in the background. Now hit it from another shell:

```bash
curl -s http://127.0.0.1:8080/health
# {"status":"ok","framework":"sklearn","model":".../trainer/model"}

curl -s -X POST http://127.0.0.1:8080/predict \
  -H 'Content-Type: application/json' \
  -d '{"inputs": [[5.1,3.5,1.4,0.2],[6.7,3.0,5.2,2.3],[6.0,2.7,5.1,1.6]]}'
# {"prediction":[0,2,2]}
```

Class `0` = *Setosa*, class `2` = *Virginica* — correct for those
measurements. Stop the server with:

```bash
kill "$(lsof -ti:8080)"
```

The request body is always `{"inputs": [...]}`. For iris, each row is
`[sepal_length, sepal_width, petal_length, petal_width]`.

### Way B — from Python

Same thing without the CLI: pass `serve_deploys=True` to the local env.
The `Deploy` step starts the server as part of `pipe.run`:

```python
from ophelian import Standalone

result = pipe.run(env=Standalone(local=True, serve_deploys=True))

serve = result.step("serve")
print(serve.info["predict"])   # http://localhost:8080/predict
print(serve.info["served"])    # True
```

While the process is alive, `POST /predict` works exactly as above. This
is the right hook when you want to serve from inside a larger script or
a notebook.

### Endpoints exposed by every Deploy

| Route      | Method | Purpose                                  |
|------------|--------|------------------------------------------|
| `/health`  | GET    | Liveness + which model is loaded         |
| `/predict` | POST   | Run inference (`{"inputs": [...]}`)       |
| `/metrics` | GET    | Prometheus metrics (set `OPHELIAN_PROMETHEUS=1`) |

> **Advanced / escape hatch.** If you ever need to serve a saved model
> *outside* a pipeline run, Ophelian exposes a zero-arg factory:
> `uvicorn --factory ophelian.runtime.fastapi_runtime:app_from_env`
> with `OPHELIAN_FRAMEWORK` and `OPHELIAN_MODEL_PATH` set. You should
> not need this for normal use — prefer Way A or Way B.

---

## 6. The cost ledger

Every finished run appends one row to `~/.ophelian/ledger.jsonl`. Read it
back in several formats:

```bash
uv run ophelian costs                      # rich table
uv run ophelian costs --by pipeline        # grouped totals
uv run ophelian costs --format json        # machine-readable
uv run ophelian costs --format csv
uv run ophelian costs --format markdown
uv run ophelian costs --since 2026-01-01 --until 2026-12-31
```

---

## 7. The cost router (optional, no cloud calls)

`Auto` picks the cheapest GPU across clouds. Offline it uses built-in
fallback prices, so you can see the decision without any account:

```python
from ophelian import Auto

Auto(cheapest_gpu="A100", project="demo-project")
# Auto router selected gcp/us-central1 a2-highgpu-1g @ 1.400 USD/h (spot)

Auto(cheapest_gpu="T4", project="demo-project")
# Auto router selected gcp/us-central1 n1-standard-4+t4 @ 0.130 USD/h (spot)
```

Building the chosen provider needs that cloud's extra
(`ophelian[gcp]`, `ophelian[aws]`, …) plus a project or credentials. If
they are missing, Ophelian fails fast with a message telling you what to
install.

---

## 8. Verified idiomatic surface

Every row below was exercised on a laptop, offline, for this tutorial.

| Surface | Idiomatic entry point | Result |
|---------|----------------------|--------|
| Install | `uv sync --frozen --extra dev` | ready |
| Test suite | `uv run pytest -q` | 406 → 409 passed (with ML extras) |
| Plan | `uv run ophelian dry-run <file> -a pipe` | DAG printed |
| Local run | `uv run ophelian run <file> -a pipe` | steps succeed |
| **Serve (CLI)** | `uv run ophelian run <file> -a pipe --serve` | live server |
| **Serve (Python)** | `Standalone(local=True, serve_deploys=True)` | live server |
| Predict | `POST /predict {"inputs":[...]}` | `{"prediction":[0,2,2]}` |
| Health | `GET /health` | `{"status":"ok",...}` |
| Metrics | `GET /metrics` (`OPHELIAN_PROMETHEUS=1`) | Prometheus text |
| Ledger | `uv run ophelian costs [--by --format --since]` | table/json/csv/md |
| Cost router | `Auto(cheapest_gpu=...)` | routes A100 / T4 offline |
| Adapters | `framework="sklearn|xgboost|pytorch|huggingface"` | train E2E locally |
| Cloud envs | `AWS(...) / GCP(...) / Azure(...)` | 154 offline tests green |

### Adapter examples

```bash
uv run python examples/sklearn_pipeline.py       # logistic regression
uv run python examples/xgboost_pipeline.py       # gradient boosting
uv run python examples/pytorch_pipeline.py       # torch.nn.Linear
uv run python examples/huggingface_pipeline.py   # tiny-gpt2, inline corpus
```

The first three train end-to-end locally with no downloads. The
HuggingFace one needs the `huggingface` extra and does a one-time ~5 MB
`tiny-gpt2` model download, but its training data is inline — no dataset
download.

---

## Recap

You installed Ophelian, ran its test suite, and served a trained model
over HTTP with a single command — `ophelian run ... --serve` — then got
a real prediction back. No manual uvicorn, no copying model files: the
framework's `Deploy` step handled the serving for you.
