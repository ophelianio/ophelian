# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2026-05-01

### Added — Multi-cloud parity

- **`GCP(project=..., region=..., machine_type=..., gpu_type=..., preemptible=True)` env**
  with `GCEDriver` (default) — same `Pipeline` source as AWS /
  Standalone, no node changes. A `GKEDriver` is on the post-1.0
  roadmap (the `gcp_backend='gke'` switch raises a clear
  `NotImplementedError` today).
- **`Azure(subscription_id=..., resource_group=..., region=..., vm_size=..., spot=True)` env**
  with `AzureVMDriver` (default). An `AKSDriver` is on the post-1.0
  roadmap (the `azure_backend='aks'` switch raises a clear
  `NotImplementedError` today).
- **GCS artifact store** (`ophelian.stores.gcs.GCSArtifactStore`) and
  **Azure Blob artifact store**
  (`ophelian.stores.azure_blob.AzureBlobArtifactStore`) sharing the
  `ArtifactStore` Protocol with the existing local + S3 stores.
- **Spot / preemptible parity.** GCP preemptible and Azure Spot VMs
  reuse the existing `Checkpoint` / `SpotInterruption` machinery.
  `step_runner` now uploads checkpoints to S3, GCS, or Azure Blob
  depending on `OPHELIAN_ARTIFACT_BACKEND`.
- **`Auto(cheapest_gpu="A100", regions=[...])` cost router.** Picks
  the cheapest configured provider/region for the requested GPU and
  returns a real provider you can pass straight to `pipeline.run(...)`.
  `dry_run=True` prints the decision and returns a no-op provider so
  CI can exercise the routing logic without provisioning anything.
- **`ophelian.pricing`**: `PriceQuote`, `RouterDecision`,
  `lookup_cheapest(gpu, regions, providers)`, JSON file cache with
  TTL, and a hand-curated static fallback table for T4 / L4 / V100 /
  A10G / A100 / H100 across the three clouds.
- **Live pricing across all three clouds** (`fetch_live`, opt-in via
  `allow_live=True`, on by default for `Auto`):
  - **AWS** — `ec2.describe_spot_price_history` (free, public, needs
    boto3 creds).
  - **Azure** — public **Retail Prices API**
    (`https://prices.azure.com/api/retail/prices`); no auth, no API
    key required. Returns true on-demand and Spot prices.
  - **GCP** — Cloud Billing Catalog API; opt-in via `GOOGLE_API_KEY`
    env var. The catalog only exposes per-SKU prices, so we publish a
    transparent *live GPU rate × `gpu_count` + static compute base*
    aggregate. Without the key, GCP gracefully falls back to the
    static table.
  - All three share one 24 h on-disk cache
    (`OPHELIAN_PRICING_CACHE_DIR`).

### Added — Observability

- **Structured JSON logging** (`ophelian.observability.configure_logging(json=True)`)
  with a `run_id` `ContextVar` propagated through every step.
- **Rich summary table** at the end of every `pipeline.run(...)`:
  step name, env, instance, duration, cost estimate, status,
  artifact URI.
- Optional **OpenTelemetry** scaffolding via `emit_event(...)` —
  no-ops when `opentelemetry` isn't installed.

### Added — Demos, docs, community

- Three viral demos under `examples/`: `llama_finetune.py`,
  `resnet_train.py`, `xgboost_tabular.py`. All default to
  `Auto(...)` and respect `OPHELIAN_DRY_RUN` for offline runs.
- **mkdocs-material site** under `docs/` (`mkdocs.yml`) — Quickstart,
  Concepts, one page per env, Models, Stores, CLI, Cookbook,
  Troubleshooting. Auto-deployed to GitHub Pages from `main`.
- Top-level `README.md` rewritten around the v1.0 hero, comparison
  table, and the three demos.
- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, GitHub issue forms, PR
  template.

### Added — Packaging

- New extras: `gcp`, `azure`, `pricing`, `docs`, `otel`, and `all`
  (everything). The `gke` / `aks` extras are reserved for the
  post-1.0 GKE / AKS drivers.
- `release.yml` publishes to PyPI via OIDC Trusted Publishing on
  `vX.Y.Z` tags.
- `docs.yml` builds + deploys the docs site.
- Third-party clouds plug in via the `ophelian.envs` entry-point
  group (see `CONTRIBUTING.md`).

### Changed

- `from ophelian import AWS, GCP, Azure, Auto, Standalone` is now the
  one true import surface for envs.
- `step_runner` refactored: `_detect_backend` chooses the artifact
  store from env vars, every persistence call (`_persist_artifacts_*`,
  `_upload_checkpoint_*`, `_materialise_upstream_*`) has a multi-cloud
  variant.

### Deprecated

- The S3-only helpers (`_persist_artifacts_to_s3`, etc.) still work
  but are deprecated aliases over the multi-cloud variants. They will
  be removed in 2.0.

## [0.5.0] - 2026-05-01

### Added — AWS provider

- **`AWS(region=..., instance=..., spot=True)` env** that runs any
  existing `Pipeline` on real EC2 workers with no pipeline changes.
  Lives in `ophelian.envs.aws` / `ophelian.providers.aws`.
- **S3 artifact store** (`ophelian.stores.s3.S3ArtifactStore`) plus a
  shared `ArtifactStore` Protocol so the local and S3 stores are
  interchangeable. `Data` nodes now transparently fetch `s3://` sources
  through the new store.
- **EC2 driver** (default) provisions per-step instances, runs the
  `step_runner` over user-data, polls S3 for `result.json`, and
  always terminates the worker in a `finally` block.
- **EKS driver** (optional, `pip install 'ophelian[eks]'`) submits each
  step as a Kubernetes Job — same artifact contract. The container
  command is a `sh -c` bootstrap script that pip-installs
  `ophelian[<extras>,aws]` into the default `python:3.12-slim` image,
  decodes `OPHELIAN_STEP_SPEC_B64` into `/work/step.json`, and execs
  `python -m ophelian.runtime.step_runner /work/step.json`. Production
  users who care about cold-start latency should pre-bake an image
  with `ophelian` already installed (passed via `runtime_image=`) —
  the bootstrap pip step then becomes a no-op. `Deploy` steps follow
  the same pattern but exec `uvicorn --factory
  ophelian.runtime.fastapi_runtime:app_from_env` instead — the
  Deployment manifest sets `OPHELIAN_FRAMEWORK` (resolved from the
  upstream Train step's `info` map threaded through
  `StepRequest.upstream_info`) and `OPHELIAN_MODEL_URI` env vars, and
  the bootstrap script `aws s3 sync`s the model URI into `/work/model`
  before exporting `OPHELIAN_MODEL_PATH` so the runtime gets a local
  filesystem path.
- **Spot interruption + resume**: workers monitor the EC2 instance
  metadata service, and the driver *also* polls
  `DescribeInstances` and treats any spot worker that transitions to
  `shutting-down`/`stopping`/`stopped`/`terminated` before producing a
  `result.json` as a reclamation. On interruption the provider
  checkpoints the run to S3; `AWS(...).with_resume(run_id=...)` (or
  `resume_run_id=...`) skips already-completed steps and re-runs only
  the interrupted step plus its descendants.
- **Deploy on EC2**: `Deploy` steps now run end-to-end on a real EC2
  worker — `step_runner` builds the FastAPI app, the user-data script
  reads the model artifact URI back out of `result.json` (which
  `_persist_artifacts_to_s3` rewrites to `s3://...`), syncs the
  prefix to `/work/model` via `aws s3 sync`, points
  `OPHELIAN_MODEL_PATH` at the resulting local file or directory
  (so the FastAPI runtime's `Path(...)` call hits a real local path
  rather than crashing on an unsupported `s3://` scheme), exports
  `OPHELIAN_FRAMEWORK`, and `exec`s
  `uvicorn --factory ophelian.runtime.fastapi_runtime:app_from_env`,
  the driver leaves the instance running and rewrites
  `result.info["predict"]`, `result.info["health"]` and
  `result.artifacts["endpoint_url"]` to point at the worker's public
  DNS instead of `localhost`. Deploy workers are tracked separately
  (`EC2Driver._keep_alive_instances`) so the regular post-run
  `teardown()` skips them; tear them down explicitly with
  `provider.cleanup_deploys()` (or via the AWS console). When no
  `security_group=` is provided, the driver auto-creates
  `ophelian-deploy-{port}` with TCP ingress on the deploy port so the
  returned public endpoint is actually reachable from the internet
  (and fails fast with a `CredentialError` if the caller IAM lacks
  `ec2:CreateSecurityGroup` / `ec2:AuthorizeSecurityGroupIngress`,
  rather than silently returning a black-holed URL).
- **Cross-step artifact persistence on EC2**: `step_runner`, when it
  detects `OPHELIAN_ARTIFACT_BUCKET` + `OPHELIAN_RUN_ID` in the
  environment (always set by the EC2 user-data script), now (a)
  downloads upstream `s3://...` artifact URIs into `/work/_upstream`
  before invoking the in-process handlers and (b) uploads each
  produced artifact to `s3://{bucket}/runs/{run_id}/{step}/artifacts/{name}`,
  rewriting `result.artifacts` to those S3 URIs. This means a fresh
  EC2 instance running step N can actually read what step N-1 produced
  on a different instance — without it, multi-step pipelines on EC2
  silently failed at the second step. The container-mode contract is
  unchanged (no env vars → no S3 calls → local paths).
- **In-flight Train checkpoint resume**: every Train step now runs with
  a `step_dir/checkpoint` directory wired through to the adapter as
  `checkpoint_dir=` (PyTorch writes per-epoch snapshots there;
  HuggingFace points `TrainingArguments.output_dir` at it). On SIGTERM
  the EC2 worker uploads the directory to
  `s3://{bucket}/{prefix}/runs/{run_id}/checkpoints/{step}` and exits
  ``75`` (`EX_TEMPFAIL`). When `AWS(...).with_resume(run_id=...)` retries
  the interrupted Train step, the provider resolves that URI and sets
  `OPHELIAN_RESUME_FROM`; `step_runner` downloads it into `/work/_resume`
  and forwards it as `_handle_train(..., resume_from=...)` so the
  adapter can pick up at the last completed epoch instead of restarting
  from scratch. sklearn / xgboost adapters log+ignore the kwarg (no
  in-fit checkpoints to resume from).
- **Auto-provisioned default S3 bucket**: omitting `artifact_bucket=`
  no longer raises — `AWSProvider` derives a deterministic
  `ophelian-artifacts-{account_id}-{region}` name via STS and creates
  it lazily, then writes the resolved name back into `config` so the
  EC2 user-data sees the same value. The store is built *before* the
  driver in `AWSProvider.__init__` so the driver's captured config
  observes the resolved bucket — without that ordering the
  `OPHELIAN_ARTIFACT_BUCKET` env var serialised into user-data would
  be empty and result polling would never find `result.json`.
  Production users who want strict separation can still pass
  `artifact_bucket=` explicitly.
- **`AWS(...).with_resume(run_id)`**: provider-level mirror of
  `AWSConfig.with_resume(...)` so the public surface documented in the
  README works directly on the env returned by `AWS(...)`.
- **Default AMIs resolved via SSM Parameter Store**: when `ami=` is not
  set, `EC2Driver` looks up the latest Amazon Linux 2023 AMI from
  `/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-{x86_64,arm64}`
  for the configured region — no more hard-coded placeholder IDs. GPU
  users still get a warning telling them to pin a Deep Learning AMI for
  out-of-the-box CUDA support.
- **Region availability check**: `AWSConfig.validate_instance_type_in_region`
  calls `ec2:DescribeInstanceTypeOfferings` once per driver lifetime
  and raises a friendly `ValueError` if the requested instance type
  isn't offered in the configured region (a common foot-gun for
  GPU-class types). Best-effort — IAM denial is swallowed so the run
  isn't blocked by a missing read-only permission.
- **moto-mocked test suite** (`tests/test_aws_provider.py`,
  `tests/test_s3_store.py`) covering provisioning, teardown, S3
  artifact round-trips, spot interruption, and resume — runs offline.
- **Opt-in real-AWS integration tests** (`tests/test_aws_integration.py`,
  marker `aws_integration`, env flag `OPHELIAN_AWS_INTEGRATION_TESTS=1`).
- **Examples**: `examples/aws/xgboost_tabular.py`,
  `examples/aws/pytorch_resnet.py`,
  `examples/aws/huggingface_llm.py`.
- **Docs**: `docs/aws.md` covering credentials, the minimum IAM policy
  for caller and worker roles, the spot/resume protocol, EKS, and
  per-workload cost estimates.

### Changed

- `LocalArtifactStore.put(...)` now returns a stored URI (string) for
  cross-store consistency. The local file is still written exactly as
  before.
- New optional extras in `pyproject.toml`: `[aws]`, `[eks]`. `[dev]`
  pulls in `boto3`, `moto`, and `paramiko`.

## [0.1.0] - 2026-05-01

### Added — greenfield foundation

This is a complete rewrite. Nothing from the previous PySpark-based releases
is carried over; the API, the architecture and the dependency graph are all
new.

- **Declarative DSL** (`ophelian.core`): immutable Pydantic v2 nodes
  `Pipeline`, `Train`, `Tune`, `Eval`, `Deploy`, `Data` with strict validation
  and automatic dependency wiring.
- **Graph compiler** with topological ordering, cycle detection and a
  `dry-run` mode that pretty-prints the plan with `rich`.
- **Standalone provider** (`Standalone(local=True)`): runs every step in its
  own dynamically-built Docker container. Ships with a `FakeDockerEngine` so
  the framework can be exercised end-to-end without a Docker daemon (used by
  the test suite).
- **Model adapters** for `pytorch`, `huggingface`, `sklearn` and `xgboost`,
  discovered via the `ophelian.adapters` entry-point group so third parties
  can register their own.
- **FastAPI inference runtime** with `/health` and `/predict`, wrapped by
  the `Deploy` step.
- **Local artifact store** (`ophelian.stores.LocalArtifactStore`).
- **CLI** (`ophelian`) built with `typer`: `version`, `dry-run`, `run`.
- **Optional extras**: `ophelian[pytorch]`, `[huggingface]`, `[sklearn]`,
  `[xgboost]`, `[all]`, `[dev]`.
- **Type information shipped to consumers** via the `py.typed` marker
  (PEP 561).
- **CI**: GitHub Actions matrix on Python 3.11 and 3.12 running `ruff`,
  `mypy --strict` and `pytest`.
- **Release workflow**: tag-driven publish to PyPI through Trusted Publishing
  (OIDC, no long-lived API tokens).

### Removed

- Everything from the legacy PySpark-based codebase: `OphelianSession`,
  `ophelian_spark.*`, the SMOTE / synthetic sampler wrappers, the SHAP /
  TensorFlow / Dask dependencies, the Poetry/Makefile workflow, the old
  Dockerfile, the legacy tutorials and notebooks.

[Unreleased]: https://github.com/LuisFalva/ophelian/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/LuisFalva/ophelian/compare/v0.5.0...v1.0.0
[0.5.0]: https://github.com/LuisFalva/ophelian/compare/v0.1.0...v0.5.0
[0.1.0]: https://github.com/LuisFalva/ophelian/releases/tag/v0.1.0
