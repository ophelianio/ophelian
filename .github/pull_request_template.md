<!--
Thanks for the PR! A short, honest description goes a long way.
Squash your commits to a single, well-written commit on merge.
-->

## What

<!-- One or two sentences: what does this PR change? -->

## Why

<!-- Link the issue if there is one (#123). Otherwise: what hurt today? -->

## How

<!-- Notable design decisions, anything reviewers should focus on. -->

## Portability check

- [ ] Pipelines that worked on `Standalone(local=True)` still work.
- [ ] Behaviour is identical across `AWS`, `GCP`, `Azure` (or the
      change is explicitly env-scoped and documented).
- [ ] No new hidden cloud calls — every code path is reachable
      through a `Local*Driver`.

## Tests

- [ ] `pytest` passes locally.
- [ ] New code has tests, or there's a note explaining why not.

## Docs

- [ ] `CHANGELOG.md` updated under `[Unreleased]`.
- [ ] Public API change → `docs/` updated.
