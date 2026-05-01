# Contributing to Ophelian

First — thank you. Ophelian is small on purpose, and every PR that
makes the framework more honest, more portable, or more pleasant to
use is genuinely appreciated.

## Ground rules

1. **Pipelines must stay portable.** Anything that works on
   `Standalone(local=True)` must also work on `AWS`, `GCP`, `Azure`,
   and `Auto(...)` without changing the pipeline source. If a feature
   only works on one cloud, it goes behind an `env`-specific kwarg,
   not a top-level node field.
2. **Public API is a contract.** Anything exported from
   `ophelian.__init__` follows semantic versioning. Breaking changes
   need a major bump and a migration note in `CHANGELOG.md`.
3. **Tests over comments.** A two-line property test beats a paragraph
   of "we do X because Y". The codebase favours code that is obvious
   on inspection.
4. **No hidden cloud calls.** Every code path that hits AWS / GCP /
   Azure must be reachable through a `Local*Driver` so contributors
   without cloud credentials can run the suite.

## Dev setup

```bash
python -m pip install -e '.[dev,sklearn,xgboost]'
pre-commit install
```

Optional cloud extras (only needed when working on the matching
provider):

```bash
pip install -e '.[aws,gcp,azure]'
```

## Running the suite

The full suite includes Docker and slow integration tests. For a fast
inner-loop run, slice by keyword:

```bash
pytest -k "stores or pricing or auto"     # fast, no cloud SDKs needed
pytest -k "providers"                     # runs all *_provider tests
pytest -m 'not integration'               # skip Docker integration
```

Type-check + lint:

```bash
ruff check ophelian tests
mypy --strict ophelian
```

## Adding a third-party env

Community envs (on-prem k8s, Lambda Labs, RunPod, ...) plug in via the
`ophelian.envs` entry-point group. From your distribution's
`pyproject.toml`:

```toml
[project.entry-points."ophelian.envs"]
RunPod = "my_pkg.envs:RunPod"
```

`ophelian.envs.discover_plugin_envs()` will surface your factory at
runtime. Your env factory must return an object that satisfies
`ophelian.providers.base.Provider`.

## Pricing data

The static price table in `ophelian/pricing/__init__.py` is hand-
curated and dated — see `STATIC_PRICES_LAST_REVIEW`. PRs that refresh
the table are welcome; please bump `STATIC_PRICES_LAST_REVIEW` and
include a link to the upstream pricing page in the PR description.

## Releases

We cut releases by pushing a `vX.Y.Z` tag. The `release.yml` workflow
takes care of the build + PyPI publish via OIDC Trusted Publishing.
Don't push tags without a `CHANGELOG.md` entry.

## Code of Conduct

By participating you agree to abide by the
[Code of Conduct](CODE_OF_CONDUCT.md).
