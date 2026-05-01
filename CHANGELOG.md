# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
