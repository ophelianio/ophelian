<p align="center">
  <a href="https://ophelianio.github.io/ophelian/">
    <img src="https://raw.githubusercontent.com/ophelianio/ophelian/dev/docs/assets/ophelian.png" alt="Ophelian" width="120" height="120">
  </a>
</p>

<h1 align="center">Ophelian</h1>

<p align="center">
  <em>Declarative, multi-cloud ML pipelines that route to the cheapest GPU.</em>
</p>

<p align="center">
  <a href="https://pypi.org/project/ophelian/"><img src="https://img.shields.io/pypi/v/ophelian.svg" alt="PyPI Latest Release"/></a>
  <a href="https://pypi.org/project/ophelian/"><img src="https://img.shields.io/pypi/pyversions/ophelian.svg" alt="Python versions"/></a>
  <a href="https://pypi.org/project/ophelian/"><img src="https://img.shields.io/pypi/dm/ophelian.svg" alt="PyPI Downloads"/></a>
  <a href="https://github.com/ophelianio/ophelian/blob/dev/LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" alt="License: Apache 2.0"/></a>
  <a href="https://github.com/ophelianio/ophelian/actions/workflows/ci.yml"><img src="https://github.com/ophelianio/ophelian/actions/workflows/ci.yml/badge.svg" alt="CI"/></a>
  <a href="https://github.com/ophelianio/ophelian"><img src="https://img.shields.io/badge/typed-mypy%20strict-1f6feb.svg" alt="mypy strict"/></a>
</p>

<p align="center">
  <a href="https://ophelianio.github.io/ophelian/">User guide</a> ·
  <a href="https://github.com/ophelianio/ophelian/blob/dev/docs/quickstart.md">Quickstart</a> ·
  <a href="https://github.com/ophelianio/ophelian/blob/dev/docs/concepts.md">Concepts</a> ·
  <a href="https://github.com/ophelianio/ophelian/blob/dev/docs/cookbook.md">Cookbook</a> ·
  <a href="https://github.com/ophelianio/ophelian/blob/dev/docs/cli.md">CLI</a> ·
  <a href="https://github.com/ophelianio/ophelian/tree/dev/examples">Examples</a> ·
  <a href="https://github.com/ophelianio/ophelian/blob/dev/CHANGELOG.md">Changelog</a> ·
  <a href="https://github.com/ophelianio/ophelian/discussions">Discussions</a>
</p>

---

## Overview

Ophelian is a small, opinionated Python framework for taking ML and AI prototypes to production without rewriting them every time the runtime changes. Declare a `Pipeline`, pick an `env`, and Ophelian compiles and runs it — locally, on AWS, GCP, or Azure.

## Quick example

```python
from ophelian import Pipeline, Train, Auto

pipe = Pipeline([
    Train(
        model="meta-llama/Llama-3.2-1B",
        data="s3://my-bucket/dataset.jsonl",
        epochs=3,
    ),
])

# Picks the cheapest A100 across AWS / GCP / Azure right now.
pipe.run(env=Auto(cheapest_gpu="A100"))
```

The same source runs on your laptop, on EC2 Spot, on a GCE preemptible VM, or on an Azure Spot VM. Ophelian handles checkpointing, artifact persistence (S3 / GCS / Azure Blob), structured logs, and a rich summary at the end.

More copy-paste recipes — LLM fine-tuning, distributed training, deploy targets, custom envs — live in the [Cookbook](https://github.com/ophelianio/ophelian/blob/dev/docs/cookbook.md).

## Highlights

- **One pipeline, every cloud.** The same `Pipeline(...)` runs on local Docker, AWS (EC2 / EKS), GCP (GCE), and Azure (Azure VM) with no code change.
- **Cost router.** `Auto(cheapest_gpu="A100")` consults live pricing across AWS, GCP, and Azure and selects the cheapest provider/region for the requested GPU class, with a static fallback when APIs are unreachable.
- **Spot-aware.** Checkpointing and auto-resume work identically on EC2 Spot, GCE preemptible, and Azure Spot. Re-run with `OPHELIAN_RESUME_FROM=<run_id>`.
- **Pluggable adapters.** PyTorch, HuggingFace Transformers, scikit-learn, and XGBoost are first-class; third-party adapters register via entry-points.
- **Local-first.** Every cloud code path has a `Local*Driver` mirror, so the full test suite exercises the framework offline — no cloud credentials required.
- **Honest cost telemetry.** Every terminal run appends one row to `~/.ophelian/ledger.jsonl`; `ophelian costs` renders showback as a rich table, Markdown, JSON, or CSV.
- **Strictly typed.** Pydantic v2 throughout; the entire codebase passes `mypy --strict`.
- **Apache 2.0**, no vendor lock-in, Python **3.11+**.

## Five envs, one API

```python
from ophelian import Standalone, AWS, GCP, Azure, Auto

Standalone(local=True)                                     # local Docker / in-process

AWS(region="us-east-1", instance="g5.xlarge", spot=True)   # EC2 / S3

GCP(project="p", region="us-central1",
    machine_type="n1-standard-4", gpu_type="nvidia-tesla-t4",
    preemptible=True)                                      # GCE / GCS

Azure(subscription_id=..., resource_group="ml",
      region="eastus", vm_size="Standard_NC6s_v3",
      spot=True)                                           # Azure VM / Blob

Auto(cheapest_gpu="A100",
     regions=["us-east-1", "us-central1", "eastus"])       # cost router
```

The same `Pipeline(...)` runs on every one of them.

## Provenance and cost

The auto-router reports exactly where each price came from — live API, on-disk cache, or static fallback table — so you can audit the decision before spending money on a GPU:

```text
Auto router selected azure/eastus Standard_NC24ads_A100_v4 @ 0.735 USD/h (spot)
  | data: aws=cached@2h gcp=static azure=live | considered: 8 quotes
```

Need a hard guarantee? `Auto(..., require_live=["azure"])` raises instead of silently falling back to a stale or static price.

Every terminal run appends a row to `~/.ophelian/ledger.jsonl`. Query it with:

```bash
ophelian costs --by team --since 2026-01-01 --format markdown
```

| team | runs |   hours | actual_usd |
| :--- | ---: | ------: | ---------: |
| ml   |   42 | 18.7000 |     $12.34 |
| nlp  |   11 |  3.2000 |      $4.10 |

The ledger schema is versioned and append-only; downstream dashboards, invoicing, and internal showback tools tail the file without us imposing a backend.

## Why not just use SageMaker / Vertex AI / Azure ML?

| Capability                               | Ophelian | SageMaker | Vertex AI | Azure ML | Bare cloud SDKs |
| :--------------------------------------- | :------: | :-------: | :-------: | :------: | :-------------: |
| Single API across AWS + GCP + Azure      |    ✅    |    ❌     |    ❌     |    ❌    |       ❌        |
| Write pipeline once, run anywhere        |    ✅    |    ❌     |    ❌     |    ❌    |       ❌        |
| Auto-router that picks the cheapest GPU  |    ✅    |    ❌     |    ❌     |    ❌    |       ❌        |
| Spot / preemptible resume                |    ✅    |  partial  |  partial  | partial  |       DIY       |
| Local-first dev (no cloud auth)          |    ✅    |    ❌     |    ❌     |    ❌    |       ❌        |
| Apache-2.0, no vendor lock-in            |    ✅    |    ❌     |    ❌     |    ❌    |      mixed      |

## Installation

```bash
pip install ophelian                  # core, runs locally
pip install 'ophelian[aws]'           # + EC2 / S3
pip install 'ophelian[gcp]'           # + GCE / GCS
pip install 'ophelian[azure]'         # + Azure VM / Blob
pip install 'ophelian[huggingface]'   # + Transformers + PyTorch
pip install 'ophelian[all]'           # every adapter and provider
```

Requires Python **3.11+**. Full extras list (`pytorch`, `sklearn`, `xgboost`, `otel`, …) in [`pyproject.toml`](https://github.com/ophelianio/ophelian/blob/dev/pyproject.toml).

Verify the install and confirm the CLI is on your `$PATH`:

```bash
ophelian version
```

## Documentation

The hosted user guide lives at **<https://ophelianio.github.io/ophelian/>**. Per-topic pages:

- [Quickstart](https://github.com/ophelianio/ophelian/blob/dev/docs/quickstart.md) — install and run your first pipeline.
- [Concepts](https://github.com/ophelianio/ophelian/blob/dev/docs/concepts.md) — pipelines, nodes, envs, providers, drivers, stores, run_id.
- [Envs](https://github.com/ophelianio/ophelian/tree/dev/docs/envs) — per-cloud reference (AWS, GCP, Azure, Standalone, Auto).
- [Cookbook](https://github.com/ophelianio/ophelian/blob/dev/docs/cookbook.md) — copy-paste recipes for common patterns.
- [CLI](https://github.com/ophelianio/ophelian/blob/dev/docs/cli.md) — `ophelian run`, `dry-run`, `costs`, `version`.
- [Cost ledger](https://github.com/ophelianio/ophelian/blob/dev/docs/ledger.md) — schema, showback, integrations.
- [Observability](https://github.com/ophelianio/ophelian/tree/dev/docs/observability) — JSON logs, OpenTelemetry, lifecycle events.
- [Troubleshooting](https://github.com/ophelianio/ophelian/blob/dev/docs/troubleshooting.md) — when it doesn't work.

Runnable end-to-end examples in [`examples/`](https://github.com/ophelianio/ophelian/tree/dev/examples). Release history in [`CHANGELOG.md`](https://github.com/ophelianio/ophelian/blob/dev/CHANGELOG.md).

## Contributing

Contributions are welcome — from bug reports to new envs and adapters. Start with the [contributing guide](https://github.com/ophelianio/ophelian/blob/dev/CONTRIBUTING.md) to set up a local development environment, then browse [good first issues](https://github.com/ophelianio/ophelian/labels/good%20first%20issue). We follow the [Contributor Covenant 2.1](https://github.com/ophelianio/ophelian/blob/dev/CODE_OF_CONDUCT.md).

## Community

- **Questions** → [GitHub Discussions](https://github.com/ophelianio/ophelian/discussions)
- **Bugs and features** → [Issues](https://github.com/ophelianio/ophelian/issues)
- **Security** → see [`SECURITY.md`](https://github.com/ophelianio/ophelian/blob/dev/SECURITY.md) — please **do not** file a public issue

## Citation

```bibtex
@software{ophelian_2026,
  author  = {Falva, Luis and the Ophelian contributors},
  title   = {{Ophelian: a declarative, multi-cloud ML pipeline framework}},
  year    = {2026},
  version = {1.1.0},
  license = {Apache-2.0},
  url     = {https://github.com/ophelianio/ophelian},
}
```

## License

Copyright © 2024–2026 Luis Falva and the Ophelian contributors. Licensed under the **Apache License, Version 2.0** — see [`LICENSE`](https://github.com/ophelianio/ophelian/blob/dev/LICENSE) and [`NOTICE`](https://github.com/ophelianio/ophelian/blob/dev/NOTICE) for third-party attributions.