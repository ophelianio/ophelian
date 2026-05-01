# Troubleshooting

When things don't work, start here.

## "I get `ModuleNotFoundError: No module named 'google.cloud.storage'`"

Install the GCP extras:

```bash
pip install 'ophelian[gcp]'
```

Same pattern for AWS (`[aws]`) and Azure (`[azure]`).

## "`Auto(...)` says no provider has credentials"

`Auto` checks for credentials before adding a provider to the price
race. Quick checks:

- AWS — `aws sts get-caller-identity` should print your account.
- GCP — `gcloud auth application-default print-access-token` should
  print a token.
- Azure — `az account show` should print your subscription.

If you're in CI, set the matching env vars
(`AWS_ACCESS_KEY_ID` / `GOOGLE_APPLICATION_CREDENTIALS` /
`AZURE_CLIENT_ID`) in the secret store.

## "My spot/preemptible VM was killed and the next run started from scratch"

Resume isn't automatic across runs (we don't want a stuck pipeline to
silently retry forever). Re-run with the same `run_id`:

```bash
OPHELIAN_RESUME_FROM=<previous_run_id> python my_pipeline.py
```

The `step_runner`'s SIGTERM handler uploads the latest checkpoint to
the artifact store before exiting, so as long as that finished, the
next run picks up from there.

## "The local Docker fallback isn't kicking in"

`Standalone(local=True)` tries Docker first. If it can't reach the
daemon, it logs a one-line warning and runs the step in-process. If
you want to *force* in-process, pass `Standalone(local=True, docker=False)`.

## "JSON logs aren't propagating run_id"

Make sure you're calling `configure_logging(json=True)` **before** the
first `pipe.run(...)`. Imports that pre-configure logging (some
notebooks do this) can shadow our handler — call
`configure_logging(json=True, force=True)` in that case.

## "`pip install ophelian[all]` is slow"

It is. `all` pulls in every cloud SDK + Hugging Face + torch + xgboost
+ mkdocs. Stick to `[aws]` / `[gcp]` / `[azure]` plus the model extras
you actually use unless you're CI'ing the kitchen sink.

## "I see `STATIC_PRICES_LAST_REVIEW` is months old"

That's a smell, not a bug. PRs to refresh the table are welcome — see
`CONTRIBUTING.md`. If you have a paid pricing API you want to plug in,
subclass `ophelian.pricing.lookup_cheapest` and override `fetch_live`.

## Still stuck

Open an issue with:

1. The output of `python -c "import ophelian; print(ophelian.__version__)"`.
2. A minimal reproducer (`Pipeline(...)`).
3. The full traceback. JSON logs are extra helpful — set
   `OPHELIAN_LOG_FORMAT=json` and paste a few lines.
