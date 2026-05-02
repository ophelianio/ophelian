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

Ophelian uses [uv](https://docs.astral.sh/uv/) for environment and
lockfile management. The pre-commit hooks run `uv run ruff` /
`uv run mypy` directly so they always match the versions pinned in
`uv.lock` (no drift between your local hooks and CI).

Install uv (one of the following), then sync the dev environment:

```bash
# pick one
curl -LsSf https://astral.sh/uv/install.sh | sh
brew install uv
pipx install uv

# from the repo root
uv sync --frozen --extra dev --extra sklearn --extra xgboost
uv run pre-commit install
```

Optional cloud extras (only needed when working on the matching
provider):

```bash
uv sync --frozen --extra dev --extra all   # aws + gcp + azure + sklearn + xgboost
```

If you really cannot use uv, you can still install with pip
(`python -m pip install -e '.[dev,sklearn,xgboost]'`), but you must
either install uv anyway for the pre-commit hooks, or run
`ruff check`, `ruff format` and `mypy` manually before pushing.

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

## Documentation links

The `Link check` workflow runs on any PR that touches `README.md`,
top-level policy files (`CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`,
`CHANGELOG.md`, `SECURITY.md`), or `docs/**/*.md`. It runs two jobs:

- **Internal links (fatal)** — `lychee --offline` verifies that every
  relative link points at a file that actually exists. A failure here
  blocks merge and almost always means a doc was renamed without
  updating callers. Fix the broken path or revert the rename.
- **External links (warn-only)** — real HTTP fetches against
  `http(s)` URLs. Marked `continue-on-error: true` so a flaky third-
  party server does not block your PR; the warning still surfaces in
  the checks list. If the failing URL is intentionally unreachable
  from CI (private dashboard, auth wall), add it to the `exclude =
  [...]` list in `lychee.toml` with a comment explaining why.

To reproduce locally before pushing:

```bash
# install lychee once: https://github.com/lycheeverse/lychee#installation
lychee --offline README.md CONTRIBUTING.md CODE_OF_CONDUCT.md \
                 CHANGELOG.md SECURITY.md 'docs/**/*.md'
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

## Changes to `.github/` (workflows, Dependabot, CODEOWNERS)

Anything under `.github/` is owned by the release maintainers via
[`.github/CODEOWNERS`](.github/CODEOWNERS). That covers:

- `.github/workflows/*.yml` — CI, release, docs, security, CodeQL
- `.github/dependabot.yml` — the policy that keeps SHA-pinned actions
  fresh
- `.github/CODEOWNERS` itself

Practical impact for contributors:

- A PR that touches any of those paths will auto-request a review from
  the code owner; please don't merge it yourself even if you have
  write access.
- We pin every GitHub Action to an immutable 40-char commit SHA for
  supply-chain safety. If you bump an action manually (instead of
  letting Dependabot do it), include both the new SHA *and* the
  upstream tag as a trailing comment so the reviewer can verify it,
  e.g. `uses: actions/checkout@<sha>  # v4.2.2`.
  This is enforced by the `action-pin-check` job in
  `.github/workflows/security.yml`: any `uses:` line under
  `.github/workflows/` that doesn't end in `@<40-char-sha>` will fail
  CI with an inline annotation pointing at the offending line. If the
  job rejects your PR, run
  `git grep -nE 'uses:[[:space:]]+[^[:space:]]+@' .github/workflows`
  to find any `@v4` / `@main` / `@<short-sha>` style refs and replace
  them with the full commit SHA + tag comment.
- Workflow changes that add new `permissions:`, new `secrets.*`
  references, or new `pull_request_target` triggers should call that
  out explicitly in the PR description — those are the changes most
  likely to need extra scrutiny.

## Code of Conduct

By participating you agree to abide by the
[Code of Conduct](CODE_OF_CONDUCT.md).
