# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/LuisFalva/ophelia/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/LuisFalva/ophelia/releases/tag/v0.1.0
