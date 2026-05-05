<p align="center">
  <img src="https://raw.githubusercontent.com/ophelianio/ophelian/dev/docs/assets/ophelian.png" alt="Ophelian" width="96" height="96">
</p>

<h1 align="center">Ophelian</h1>

<p align="center">
  <strong>Write your ML pipeline once. Run it anywhere. Route to the cheapest available GPU across AWS, GCP, and Azure.</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/ophelian/"><img src="https://img.shields.io/pypi/v/ophelian.svg" alt="PyPI"></a>
  <a href="https://pypi.org/project/ophelian/"><img src="https://img.shields.io/pypi/pyversions/ophelian.svg" alt="Python versions"></a>
  <a href="https://github.com/ophelianio/ophelian/blob/v1.0.1/LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" alt="License: Apache 2.0"></a>
  <a href="https://github.com/ophelianio/ophelian/actions/workflows/ci.yml"><img src="https://github.com/ophelianio/ophelian/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://ophelianio.github.io/ophelian/"><img src="https://img.shields.io/badge/docs-online-4baaaa.svg" alt="Docs"></a>
</p>

---

Ophelian is a small, opinionated Python framework for taking ML / AI
prototypes to production without rewriting them every time the runtime
changes. Declare a `Pipeline`, pick an `env`, and Ophelian compiles +
runs it — locally in Docker, on AWS, GCP, or Azure, or routed
automatically to the cheapest cloud for the GPU you need.

## Install

```bash
pip install ophelian                  # core, runs locally
pip install 'ophelian[aws]'           # + EC2 / S3
pip install 'ophelian[gcp]'           # + GCE / GCS
pip install 'ophelian[azure]'         # + Azure VM / Blob
pip install 'ophelian[huggingface]'   # + Transformers + PyTorch
pip install 'ophelian[all]'           # every adapter and provider
```

Python **3.11+**. Full extras list (`pytorch`, `sklearn`, `xgboost`,
`otel`, ...) in [`pyproject.toml`](https://github.com/ophelianio/ophelian/blob/v1.0.1/pyproject.toml).

## Hello, pipeline

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

The same source runs on your laptop, on EC2 spot, on a GCE preemptible
VM, or on an Azure Spot VM — Ophelian handles checkpointing, artifact
persistence (S3 / GCS / Azure Blob), structured logs, and a rich
summary at the end.

## Envs

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

The same `Pipeline(...)` runs on every one of them. Each cloud env has
a `Local*Driver` mirror so the test suite (and any contributor without
cloud creds) exercises the full framework offline.

## Why Ophelian

| | Ophelian | SageMaker | Vertex AI | Azure ML | Bare cloud SDKs |
|---|:---:|:---:|:---:|:---:|:---:|
| Single API across AWS + GCP + Azure | ✅ | ❌ | ❌ | ❌ | ❌ |
| Write pipeline once, run anywhere | ✅ | ❌ | ❌ | ❌ | ❌ |
| Auto-router that picks the cheapest GPU | ✅ | ❌ | ❌ | ❌ | ❌ |
| Spot / preemptible resume | ✅ | partial | partial | partial | DIY |
| Local-first dev (no cloud auth) | ✅ | ❌ | ❌ | ❌ | ❌ |
| Apache-2.0, no vendor lock-in | ✅ | ❌ | ❌ | ❌ | mixed |

## Documentation

Full docs live at **<https://ophelianio.github.io/ophelian/>**:
[Quickstart](https://ophelianio.github.io/ophelian/quickstart/) ·
[Concepts](https://ophelianio.github.io/ophelian/concepts/) ·
[Envs](https://ophelianio.github.io/ophelian/envs/auto/) ·
[Cookbook](https://ophelianio.github.io/ophelian/cookbook/) ·
[Troubleshooting](https://ophelianio.github.io/ophelian/troubleshooting/).

Runnable examples in [`examples/`](https://github.com/ophelianio/ophelian/tree/v1.0.1/examples).
Roadmap and release history in [`CHANGELOG.md`](https://github.com/ophelianio/ophelian/blob/v1.0.1/CHANGELOG.md).

## Community

- **Questions** → [GitHub Discussions](https://github.com/ophelianio/ophelian/discussions)
- **Bugs / features** → [Issues](https://github.com/ophelianio/ophelian/issues)
- **Security** → see [`SECURITY.md`](https://github.com/ophelianio/ophelian/blob/v1.0.1/SECURITY.md) — please **do not** file a public issue
- **Contributing** → [`CONTRIBUTING.md`](https://github.com/ophelianio/ophelian/blob/v1.0.1/CONTRIBUTING.md) (we follow the [Contributor Covenant 2.1](https://github.com/ophelianio/ophelian/blob/v1.0.1/CODE_OF_CONDUCT.md))

## Citation

```bibtex
@software{ophelian_2026,
  author    = {Falva, Luis and the Ophelian contributors},
  title     = {{Ophelian: a declarative, multi-cloud ML pipeline framework}},
  year      = {2026},
  version   = {1.0.1},
  license   = {Apache-2.0},
  url       = {https://github.com/ophelianio/ophelian},
}
```

## License

Copyright © 2024–2026 Luis Falva and the Ophelian contributors.
Licensed under the **Apache License, Version 2.0** — see
[`LICENSE`](https://github.com/ophelianio/ophelian/blob/v1.0.1/LICENSE)
and [`NOTICE`](https://github.com/ophelianio/ophelian/blob/v1.0.1/NOTICE)
for third-party attributions.
