# Security Policy

Thanks for helping keep Ophelian and its users safe. This document
explains which versions receive security fixes, how to report a
vulnerability privately, and what response timeline you can expect.

## Supported versions

Ophelian follows [Semantic Versioning 2.0](https://semver.org/spec/v2.0.0.html).
Security fixes are issued for the latest **minor** release on the
current major line. Older minors may receive a backport at the
maintainers' discretion when the fix is small and the impact is high.

| Version | Supported          |
|---------|--------------------|
| 1.x (latest minor) | :white_check_mark: |
| 1.x (older minors) | :grey_question: best-effort backport for high-severity issues |
| < 1.0   | :x:                |

The latest released version is published on
[PyPI](https://pypi.org/project/ophelian/) and tagged in this repo.

## Reporting a vulnerability

**Please do not open a public GitHub issue, discussion, or pull
request for a suspected security vulnerability.** Public disclosure
before a fix is available puts every Ophelian user at risk.

Use one of the private channels below, in order of preference:

1. **GitHub Private Vulnerability Reporting (preferred).**
   Open a private report at
   <https://github.com/LuisFalva/ophelian/security/advisories/new>
   — this is the "Report a vulnerability" button on the repo's
   [Security tab](https://github.com/LuisFalva/ophelian/security).
   It is end-to-end private, requires no email round-trip, and
   triages directly into a GitHub Security Advisory if accepted.

2. **Direct contact with the maintainer (only if you do not have a
   GitHub account).** Contact the maintainer
   [@LuisFalva](https://github.com/LuisFalva) privately via the
   email listed on their public GitHub profile. Please put
   `[ophelian-security]` in the subject line. PVR is the strongly
   preferred channel — email is slower to triage and lacks the
   audit trail of a GitHub Security Advisory.

When you report, it helps us a lot if you include:

- A description of the issue and its impact (what an attacker can
  do, and to whom).
- The affected version(s) — `pip show ophelian` or the commit SHA.
- A minimal reproduction: pipeline definition, env, command, and
  observed vs. expected behaviour.
- Any proof-of-concept code, logs, or screenshots (please redact
  secrets).
- Whether you would like to be credited in the published advisory,
  and under what name / handle.

## Response timeline

We aim to respond on the following schedule. These are targets, not
SLAs — Ophelian is currently maintained by a small team — but we will
keep you updated if something is going to slip.

| Stage | Target |
|---|---|
| Acknowledge receipt of your report | within **5 business days** |
| Initial triage (severity, in-scope confirmation, reproduction) | within **10 business days** |
| Status update if a fix is going to take longer | at least every **14 days** until resolved |
| Coordinated fix released for accepted, in-scope reports | targeting **90 days** from triage, sooner for high-severity issues |

After a fix ships, we will publish a
[GitHub Security Advisory](https://github.com/LuisFalva/ophelian/security/advisories)
with a CVE (when applicable), the affected version range, the fixed
version, and credit to the reporter (unless you have asked to remain
anonymous).

## Scope

In scope:

- The `ophelian` Python package published on PyPI.
- This repository's source, build configuration, and release
  workflows under `.github/workflows/`.
- Documented public API surface (anything re-exported from
  `ophelian.__init__` or documented under `docs/`).
- Provider env adapters (`local`, `aws`, `gcp`, `azure`, `router`)
  shipped from this repo.

Out of scope:

- Vulnerabilities in third-party dependencies that have not yet
  been disclosed upstream — please report those to the upstream
  project first; we will pick up fixes via Dependabot and pin bumps.
- Bugs that require an attacker who already has shell access, cloud
  credentials, or write access to the user's pipeline definitions
  (these are part of the trust model, not a vulnerability in
  Ophelian itself).
- Findings against example pipelines under `examples/` that do not
  affect the published package.
- Denial-of-service caused by user-supplied pipelines that are
  obviously expensive (this is a cost question, not a security one
  — see the cost-estimation tooling in the README).

If you are unsure whether something is in scope, please report it
privately anyway and we will help you classify it.

## Safe-harbour

We will not pursue or support legal action against researchers who:

- Make a good-faith effort to follow this policy.
- Avoid privacy violations, data destruction, and service
  disruption while testing.
- Give us a reasonable window to ship a fix before any public
  disclosure.

## Defense in depth

Reports remain the primary channel — the items below are guardrails,
not a substitute for a private report:

- **CodeQL** static analysis on every push to `main`, every PR, and
  a weekly cron.
- **pip-audit**, **bandit**, and **gitleaks** on every push and PR,
  plus a weekly cron so new CVEs surface even when the repo is quiet.
- **GitHub Actions pinned to commit SHAs**, with Dependabot watching
  for supply-chain regressions and an `action-pin-check` CI job that
  fails the build if any workflow re-introduces a mutable
  `@v4`-style reference.
- **CODEOWNERS-gated reviews** on `.github/` so workflow, Dependabot,
  and CODEOWNERS changes cannot land without maintainer review.

Thanks again for reporting responsibly.
