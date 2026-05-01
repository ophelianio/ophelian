# CLI

Installing `ophelian` puts a single `ophelian` command on your `$PATH`.

```bash
ophelian --help
```

## Commands

### `ophelian version`

Prints the installed version.

```bash
$ ophelian version
ophelian 1.0.0
```

### `ophelian dry-run <pipeline.py>`

Imports the file, finds a `Pipeline` instance, and prints the
compiled execution plan **without** running anything. Use this to
sanity-check pipeline construction or to share the plan in a PR.

```bash
ophelian dry-run examples/xgboost_tabular.py
```

By default the loader picks up the first top-level `Pipeline` in the
file. Use `--attribute` to pick a specific one:

```bash
ophelian dry-run examples/xgboost_tabular.py --attribute pipe
```

### `ophelian run <pipeline.py>`

Loads the pipeline and executes it against the env declared inside
the file (the `pipe.run(env=...)` call).

```bash
ophelian run examples/xgboost_tabular.py
```

Useful env vars at run time:

| Var | Effect |
|---|---|
| `OPHELIAN_DRY_RUN=1` | `Auto(...)` returns a no-op provider. |
| `OPHELIAN_RUN_ID=...` | Override the auto-generated `run_id`. |
| `OPHELIAN_RESUME_FROM=<run_id>` | Resume from a previous run's checkpoints. |
| `OPHELIAN_LOG_FORMAT=json` | Force JSON-line logs (same as `configure_logging(json=True)`). |
| `OPHELIAN_PROVIDERS=aws,gcp` | Restrict which providers `Auto` considers. |
