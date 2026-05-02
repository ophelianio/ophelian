<div align="center">

# Ophelian

**Write your ML pipeline once. Run it anywhere. Pay the lowest GPU price on the market.**

<!-- Status -->
[![PyPI](https://img.shields.io/pypi/v/ophelian.svg)](https://pypi.org/project/ophelian/)
[![Python](https://img.shields.io/pypi/pyversions/ophelian.svg)](https://pypi.org/project/ophelian/)
[![Downloads](https://static.pepy.tech/badge/ophelian/month)](https://pepy.tech/project/ophelian)
[![License](https://img.shields.io/pypi/l/ophelian.svg)](LICENSE)

<!-- Quality -->
[![CI](https://github.com/LuisFalva/ophelian/actions/workflows/ci.yml/badge.svg)](https://github.com/LuisFalva/ophelian/actions/workflows/ci.yml)
[![CodeQL](https://github.com/LuisFalva/ophelian/actions/workflows/codeql.yml/badge.svg)](https://github.com/LuisFalva/ophelian/actions/workflows/codeql.yml)
[![Security](https://github.com/LuisFalva/ophelian/actions/workflows/security.yml/badge.svg)](https://github.com/LuisFalva/ophelian/actions/workflows/security.yml)
[![Docs](https://github.com/LuisFalva/ophelian/actions/workflows/docs.yml/badge.svg)](https://luisfalva.github.io/ophelian/)

<!-- Community -->
[![Code of Conduct](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)
[![Discussions](https://img.shields.io/github/discussions/LuisFalva/ophelian)](https://github.com/LuisFalva/ophelian/discussions)
[![Issues](https://img.shields.io/github/issues/LuisFalva/ophelian)](https://github.com/LuisFalva/ophelian/issues)

</div>

---

Ophelian is a small, opinionated Python framework for taking ML / AI
prototypes to production without rewriting them every time the runtime
changes. You declare a `Pipeline` of nodes (`Data`, `Train`, `Tune`,
`Eval`, `Deploy`), pick an `env`, and Ophelian compiles + runs it —
locally inside Docker, on AWS, GCP, or Azure, or routed automatically
to the cheapest cloud for the GPU you need.

## Table of contents

- [Hello, pipeline](#hello-pipeline)
- [Why Ophelian](#why-ophelian)
- [Install](#install)
- [Architecture](#architecture)
- [Envs at a glance](#envs-at-a-glance)
- [Three demos in three minutes](#three-demos-in-three-minutes)
- [Use cases](#use-cases)
- [Observability](#observability)
- [Compatibility](#compatibility)
- [Documentation](#documentation)
- [Roadmap](#roadmap)
- [Status & versioning](#status--versioning)
- [Community & support](#community--support)
- [Governance & maintainers](#governance--maintainers)
- [Security](#security)
- [Contributing](#contributing)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)
- [License](#license)

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

That's it. The same source runs on your laptop, on EC2 spot, on a GCE
preemptible VM, or on an Azure Spot VM — Ophelian handles
checkpointing, artifact persistence (S3 / GCS / Azure Blob),
structured logs, and a rich summary at the end.

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

## Architecture

Ophelian draws a hard line between *what* a pipeline does and *where*
it runs. Nodes are declarative. Envs pick a provider. Providers pick a
driver. Drivers persist artifacts to a store. The cost router is the
only component that crosses providers.

```
                 +--------------------------------------------+
                 |              Pipeline (DAG)                |
                 |  Data → Train → Tune → Eval → Deploy       |
                 +--------------------------------------------+
                                     |
                                     v
                 +--------------------------------------------+
                 |  Env  (Standalone | AWS | GCP | Azure |    |
                 |        Auto)                               |
                 +--------------------------------------------+
                                     |
                  +------------------+--------------------+
                  |                  |                    |
                  v                  v                    v
          +---------------+  +---------------+   +-----------------+
          |  Provider     |  |  Provider     |   |  Provider       |
          |  (AWS, GCP,   |  |  (Standalone) |   |  (chosen by     |
          |   Azure)      |  |               |   |   Auto router)  |
          +---------------+  +---------------+   +-----------------+
                  |
       +----------+----------+
       |                     |
       v                     v
  +---------+         +-------------+         +-------------------+
  | Driver  |  ...    |   Driver    |  <----  |   Cost router     |
  | (EC2,   |         |  (LocalEC2, |         |  (live spot +     |
  |  EKS,   |         |   FakeGCE…) |         |   retail prices,  |
  |  GCE,   |         |             |         |   24h disk cache) |
  |  AzureVM|         +-------------+         +-------------------+
  +---------+
       |
       v
  +-----------------------------------------------------+
  |  ArtifactStore  (Local | S3 | GCS | Azure Blob)     |
  |   put/get/exists/delete/list  — same Protocol       |
  +-----------------------------------------------------+
```

Every cloud code path has a `Local*Driver` mirror, so the test suite
(and any contributor without cloud creds) exercises the full
framework offline.

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

## Three demos in three minutes

All three live under [`examples/`](examples/) and default to
`Auto(cheapest_gpu=...)`. Set `OPHELIAN_DRY_RUN=1` to print the
chosen provider/region/price without spinning anything up.

```bash
python examples/llama_finetune.py    # Llama-3 fine-tune on the cheapest A100
python examples/resnet_train.py      # ResNet-50 on ImageNet, preemptible-friendly
python examples/xgboost_tabular.py   # XGBoost tabular, CPU only, sub-$0.05/run
```

## Use cases

| Scenario | Why Ophelian fits | Snippet |
|---|---|---|
| **Cost-driven LLM fine-tuning** — you want an A100 right now and don't care which cloud sells it cheapest | `Auto(cheapest_gpu=...)` queries live AWS spot, Azure retail, and GCP billing prices, picks the winner, and resumes from checkpoint if the spot is reclaimed | `pipe.run(env=Auto(cheapest_gpu="A100"))` |
| **Local → cloud without rewrites** — prototype on your laptop, ship the same `pipe` to production | Every cloud code path mirrors a `Local*Driver`, so the same `Pipeline` object runs in-process, in Docker, on EC2, on GCE, or on an Azure VM | `pipe.run(env=Standalone(local=True))` then `pipe.run(env=AWS(...))` |
| **Multi-cloud failover for batch inference** — your primary region runs out of GPU capacity at 3am | List acceptable regions in `Auto(...)`; the router falls through to the next cheapest provider with capacity, transparently | `Auto(cheapest_gpu="L4", regions=["us-east-1","us-central1","eastus"])` |
| **Reproducible academic benchmarks** — you need to publish numbers another lab can re-run | `run_id`-tagged structured logs, deterministic artifact layout, pinned price table, Apache-2.0 license, [citable](#citation) | `configure_logging(json=True)` + cite the version you used |

## Observability

```python
from ophelian.observability import configure_logging
configure_logging(json=True)
```

Every step emits structured JSON with `run_id`, `step`, `env`,
`instance`, `duration_s`, `cost_estimate_usd`. At the end you get a
rich table summary you can paste into a Slack thread.

## Compatibility

Tested in CI on every push:

| Python | Linux (Ubuntu) | macOS | Windows |
|:------:|:--------------:|:-----:|:-------:|
| 3.11   | ✅ full         | community | community |
| 3.12   | ✅ full         | ✅ full | community |
| 3.13   | ✅ full         | community | community |

Cloud provider support:

| Provider | Compute | Artifact store | Spot / preemptible | Status |
|---|---|---|:---:|:---:|
| **Standalone** | local Docker / in-process | local FS | n/a | ✅ stable |
| **AWS** | EC2 (default), EKS (`[eks]` extra) | S3 | ✅ EC2 spot | ✅ stable |
| **GCP** | GCE | GCS | ✅ preemptible | ✅ stable (GKE planned) |
| **Azure** | Azure VM | Azure Blob | ✅ Azure Spot | ✅ stable (AKS planned) |
| **Auto router** | picks any of the above | inherits from chosen provider | ✅ | ✅ stable |

Third-party clouds plug in through the `ophelian.envs` entry-point
group — see [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Documentation

Full docs at **<https://luisfalva.github.io/ophelian/>**:

- [Quickstart](https://luisfalva.github.io/ophelian/quickstart/)
- [Concepts](https://luisfalva.github.io/ophelian/concepts/)
- Envs:
  [AWS](https://luisfalva.github.io/ophelian/envs/aws/) ·
  [GCP](https://luisfalva.github.io/ophelian/envs/gcp/) ·
  [Azure](https://luisfalva.github.io/ophelian/envs/azure/) ·
  [Standalone](https://luisfalva.github.io/ophelian/envs/standalone/) ·
  [Auto](https://luisfalva.github.io/ophelian/envs/auto/)
- [Cookbook](https://luisfalva.github.io/ophelian/cookbook/) ·
  [Troubleshooting](https://luisfalva.github.io/ophelian/troubleshooting/)

## Roadmap

Tracked in [`CHANGELOG.md`](CHANGELOG.md) under `## [Unreleased]`
and in the [GitHub project board](https://github.com/LuisFalva/ophelian/issues).
Headline items currently on deck:

- **GKE driver** — production-grade Kubernetes backend for GCP
  (`gcp_backend='gke'` is reserved and raises a clear
  `NotImplementedError` today).
- **AKS driver** — same, for Azure (`azure_backend='aks'`).
- **Real-AWS smoke run in nightly CI** — opt-in integration tests
  promoted from local-only to a scheduled workflow.
- **Pre-launch cost preview** — print estimated $/run before any
  AWS pipeline actually provisions infrastructure.
- **Live pricing for more GPU classes** — extend the router beyond
  the current T4 / L4 / V100 / A10G / A100 / H100 set.

Anything that breaks the public API requires a major version bump and
a migration note — see [Status & versioning](#status--versioning).

## Status & versioning

`v1.0.0` — public API is stable. Ophelian follows
[Semantic Versioning 2.0](https://semver.org/spec/v2.0.0.html):

- **MAJOR** — breaking changes to the public surface re-exported
  from `ophelian.__init__` or to documented env / provider kwargs.
- **MINOR** — backwards-compatible additions (new envs, new node
  fields with safe defaults, new optional extras).
- **PATCH** — bug fixes, documentation, dependency bumps, perf.

Every breaking change gets a migration note in
[`CHANGELOG.md`](CHANGELOG.md). Deprecated symbols stay importable
with a `DeprecationWarning` for at least one minor release before
removal.

## Community & support

Pick the right channel:

| You want to... | Go to |
|---|---|
| Ask a usage question | [GitHub Discussions](https://github.com/LuisFalva/ophelian/discussions) |
| Report a reproducible bug | [Issues → Bug report](https://github.com/LuisFalva/ophelian/issues/new?template=bug_report.yml) |
| Request a feature | [Issues → Feature request](https://github.com/LuisFalva/ophelian/issues/new?template=feature_request.yml) |
| Report a security vulnerability | See [Security](#security) — please **do not** file a public issue |
| Show what you built with it | [Discussions → Show and tell](https://github.com/LuisFalva/ophelian/discussions/categories/show-and-tell) |

We follow the
[Contributor Covenant 2.1](CODE_OF_CONDUCT.md) in every space we
maintain.

## Governance & maintainers

Ophelian is currently maintained by [@LuisFalva](https://github.com/LuisFalva)
under a lightweight BDFL model: the maintainer has final say on
architecture and API decisions, contributors propose changes via PRs,
and every public-API change goes through a CHANGELOG-gated review.

A formal `GOVERNANCE.md` and `MAINTAINERS.md` are planned for the
1.x series as the contributor base grows. The intended trajectory is
the standard meritocratic open-source model: contributors who make
sustained, high-quality contributions are invited to become committers,
and committers vote on new committers. PRs that move the project in
that direction are welcome.

## Security

Please **do not** report security vulnerabilities via public GitHub
issues. We take security seriously and want a chance to ship a fix
before the bug is public.

- See [`SECURITY.md`](SECURITY.md) for the full disclosure policy
  (coverage, supported versions, expected response time).
- Coordinated disclosure is preferred. Patched releases ship with
  a `Security Advisory` published on the GitHub repo.
- The CI pipeline runs **CodeQL**, **pip-audit**, and **gitleaks**
  on every push; GitHub Actions are pinned to commit SHAs and
  Dependabot watches for supply-chain regressions.

## Contributing

PRs welcome. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for dev setup
(uv-based), the test slicing strategy, the third-party env plug-in
contract, and the pricing-table refresh process. The bar for new
public API is *"this makes pipelines more portable, more honest, or
more pleasant to use across every supported env."*

By participating you agree to abide by the
[Code of Conduct](CODE_OF_CONDUCT.md).

## Citation

If Ophelian shows up in research output (papers, theses, technical
reports), please cite the specific version you used:

```bibtex
@software{ophelian_2026,
  author    = {Falva, Luis and the Ophelian contributors},
  title     = {{Ophelian: a declarative, multi-cloud ML pipeline framework}},
  year      = {2026},
  version   = {1.0.0},
  license   = {Apache-2.0},
  url       = {https://github.com/LuisFalva/ophelian},
  publisher = {GitHub},
}
```

For other versions, swap `version` and check the matching tag at
<https://github.com/LuisFalva/ophelian/releases>.

## Acknowledgments

Ophelian stands on a lot of upstream work and would not exist without
the ecosystems behind:

- **HuggingFace Transformers** and **PyTorch** — model adapters and
  the training stack that the framework wraps.
- **scikit-learn** and **XGBoost** — the tabular adapters that keep
  the framework honest for non-LLM workloads.
- **boto3 / google-cloud-* / azure-sdk-for-python** — the cloud
  SDKs that make the multi-cloud surface possible.
- **pydantic v2**, **typer**, **rich**, **FastAPI**, **uv**,
  **hatchling**, **mypy**, **ruff** — the small Python-tooling
  stack that the codebase is built on.

And to every contributor who has filed an issue, opened a PR,
refreshed the price table, or kicked the tyres on a real cloud:
**thank you.** Run `git shortlog -sn --no-merges` for the full list.

## License

Copyright © 2024–2026 Luis Falva and the Ophelian contributors.

Licensed under the **Apache License, Version 2.0** — see
[`LICENSE`](LICENSE) for the full text. A `NOTICE` file enumerating
attributions for redistributed third-party components is planned
alongside the next packaging audit.

You may not use this project except in compliance with the License.
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an **"AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND**, either express or
implied. See the License for the specific language governing
permissions and limitations under the License.
