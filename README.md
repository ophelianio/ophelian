# Ophelian

> **Write your ML pipeline once. Run it anywhere. Pay the lowest GPU price on the market.**

[![PyPI](https://img.shields.io/pypi/v/ophelian.svg)](https://pypi.org/project/ophelian/)
[![Python](https://img.shields.io/pypi/pyversions/ophelian.svg)](https://pypi.org/project/ophelian/)
[![License](https://img.shields.io/pypi/l/ophelian.svg)](LICENSE)
[![CI](https://github.com/LuisFalva/ophelian/actions/workflows/ci.yml/badge.svg)](https://github.com/LuisFalva/ophelian/actions/workflows/ci.yml)
[![Docs](https://github.com/LuisFalva/ophelian/actions/workflows/docs.yml/badge.svg)](https://luisfalva.github.io/ophelian/)

Ophelian is a small, opinionated Python framework for taking ML / AI
prototypes to production without rewriting them every time the runtime
changes. You declare a `Pipeline` of nodes (`Data`, `Train`, `Tune`,
`Eval`, `Deploy`), pick an `env`, and Ophelian compiles + runs it.

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

That's it. Same source runs on your laptop, on EC2 spot, on a GCE
preemptible VM, or on an Azure Spot VM — Ophelian handles checkpointing,
artifact persistence (S3 / GCS / Azure Blob), structured logs, and a
rich summary at the end.

## Why Ophelian

| | Ophelian | SageMaker | Vertex AI | Azure ML | Bare cloud SDKs |
|---|:---:|:---:|:---:|:---:|:---:|
| Single API across AWS + GCP + Azure | ✅ | ❌ | ❌ | ❌ | ❌ |
| Write pipeline once, run anywhere | ✅ | ❌ | ❌ | ❌ | ❌ |
| Auto-router that picks the cheapest GPU | ✅ | ❌ | ❌ | ❌ | ❌ |
| Spot / preemptible / Azure Spot resume | ✅ | partial | partial | partial | DIY |
| Local-first dev (Docker, no cloud auth) | ✅ | ❌ | ❌ | ❌ | ❌ |
| Structured JSON logs + run_id | ✅ | partial | partial | partial | DIY |
| Apache-2.0, OSS, no vendor SDK lock-in | ✅ | ❌ | ❌ | ❌ | mixed |
| `pip install` and go | ✅ | ❌ | ❌ | ❌ | partial |

## Install

```bash
pip install ophelian            # core, runs locally
pip install 'ophelian[aws]'     # + EC2 / S3
pip install 'ophelian[gcp]'     # + GCE / GCS
pip install 'ophelian[azure]'   # + Azure VM / Blob
pip install 'ophelian[all]'     # every extra
```

Python **3.11+**.

## Three demos in three minutes

All three live under [`examples/`](examples/) and default to
`Auto(cheapest_gpu=...)`. Set `OPHELIAN_DRY_RUN=1` to print the
chosen provider/region/price without spinning anything up.

### 1. Llama-3 fine-tune on the cheapest A100

```bash
python examples/llama_finetune.py
```

### 2. ResNet-50 on ImageNet — preemptible-friendly

```bash
python examples/resnet_train.py
```

### 3. XGBoost tabular — CPU only, sub-$0.05/run

```bash
python examples/xgboost_tabular.py
```

## Envs at a glance

```python
from ophelian import Standalone, AWS, GCP, Azure, Auto

Standalone(local=True)                                     # local Docker / in-process
AWS(region="us-east-1", instance="g5.xlarge", spot=True)   # EC2 / S3
GCP(project="my-proj", region="us-central1",
    machine_type="n1-standard-4", gpu_type="nvidia-tesla-t4",
    preemptible=True)                                      # GCE / GCS
Azure(subscription_id=..., resource_group="ml",
      region="eastus", vm_size="Standard_NC6s_v3",
      spot=True)                                           # Azure VM / Blob
Auto(cheapest_gpu="A100",
     regions=["us-east-1", "us-central1", "eastus"])       # cost router
```

The same `Pipeline(...)` runs on every one of them.

## Observability

```python
from ophelian.observability import configure_logging
configure_logging(json=True)
```

Every step emits structured JSON with `run_id`, `step`, `env`,
`instance`, `duration_s`, `cost_estimate_usd`. At the end you get a
rich table summary you can paste into a Slack thread.

## Docs

Full docs at **<https://luisfalva.github.io/ophelian/>**:

- [Quickstart](https://luisfalva.github.io/ophelian/quickstart/)
- [Concepts](https://luisfalva.github.io/ophelian/concepts/)
- Envs: [AWS](https://luisfalva.github.io/ophelian/envs/aws/) ·
  [GCP](https://luisfalva.github.io/ophelian/envs/gcp/) ·
  [Azure](https://luisfalva.github.io/ophelian/envs/azure/) ·
  [Standalone](https://luisfalva.github.io/ophelian/envs/standalone/) ·
  [Auto](https://luisfalva.github.io/ophelian/envs/auto/)
- [Cookbook](https://luisfalva.github.io/ophelian/cookbook/) ·
  [Troubleshooting](https://luisfalva.github.io/ophelian/troubleshooting/)

## Status

`v1.0.0` — public API is stable. Breaking changes from this point
forward require a major version bump and a migration note in
`CHANGELOG.md`.

## Contributing

PRs welcome. See [`CONTRIBUTING.md`](CONTRIBUTING.md) and the
[Code of Conduct](CODE_OF_CONDUCT.md). The bar for new public API is
*"this makes pipelines more portable, more honest, or more pleasant to
use across every supported env."*

## License

Apache-2.0. See [`LICENSE`](LICENSE).
