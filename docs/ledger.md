# Cost ledger

Every terminal pipeline run appends one structured row to a local
JSONL **cost ledger**. The ledger is the source of truth for the
`ophelian costs` showback CLI and the foundation for any downstream
consumer (dashboards, invoicing) built on top of Ophelian.

## Location

| Setting | Default |
| --- | --- |
| Ledger path | `~/.ophelian/ledger.jsonl` |
| Override env var | `OPHELIAN_LEDGER_PATH` |
| Disable writes | `OPHELIAN_LEDGER_DISABLED=1` |

The file is **append-only JSONL**: one JSON object per line, written
atomically under a `fcntl.flock` sidecar so concurrent pipelines on
the same host never interleave a partial line.

## Row schema

`schema_version` carries the contract version. Today's value is `1`.
New fields will be added at the end with sensible defaults; breaking
changes bump the version.

| Field | Type | Notes |
| --- | --- | --- |
| `schema_version` | int | Always present. Currently `1`. |
| `run_id` | str | Stable run identifier (falls back to pipeline name). |
| `pipeline` | str | `Pipeline.name`. |
| `timestamp` | float | Epoch seconds (UTC) at row write time. |
| `env_class` | str | `env.name` — `"standalone"`, `"aws"`, `"gcp"`, `"azure"`. |
| `provider` | str | Same as `env_class` today; reserved for future split. |
| `region` | str | Cloud region or `"local"`. |
| `gpu_type` | str \| null | GPU SKU when known. |
| `instance` | str \| null | Cloud instance type / VM size when known. |
| `hours` | float | Total wall-clock hours summed across step durations. |
| `hourly_usd` | float \| null | Router quote (per-GPU-hour) when available. |
| `estimated_usd` | float \| null | `hourly_usd * hours`. |
| `actual_usd` | float \| null | Cost incurred (same calc; failed runs reflect cost up to failure). |
| `status` | str | `"success"` or `"failed"`. |
| `context` | object | Opaque labels propagated from `Pipeline(context={...})`. |

## Context propagation

`Pipeline(context={"team": "ml", "tenant_id": "acme"})` plumbs the
same dict through:

1. every emitted lifecycle event (`PipelineStarted`,
   `StepCompleted`, …),
2. every OpenTelemetry span attached to the run, and
3. the `context` field of the ledger row.

Downstream consumers can therefore join on `context.tenant_id` (or
any other key) across logs, traces, and showback rows without extra
plumbing.

## Showback CLI

```bash
ophelian costs                              # human-readable rich table
ophelian costs --since 2026-01-01           # filter by ISO date
ophelian costs --since 2026-01-01 --until 2026-02-01
ophelian costs --by team                    # group by a context key
ophelian costs --by provider --format markdown
ophelian costs --format json
ophelian costs --format csv
```

Filters:

- `--since` / `--until`: ISO 8601 timestamp (`2026-01-01` or
  `2026-01-01T12:34:56`) or raw epoch seconds.
- `--by`: any built-in field (`provider`, `env_class`, `region`,
  `gpu_type`, `instance`, `pipeline`, `status`, `run_id`) or any key
  inside the row's `context` dict.
- `--format`: `table` (default), `markdown`, `json`, `csv`.
- `--path`: override the ledger path (otherwise honors
  `OPHELIAN_LEDGER_PATH`).

## Why a local file?

DVC and MLflow both ship a local store that works without any
external service or signup. The ledger follows the same pattern: an
OSS user gets useful showback out of the box, and any SaaS built on
top of Ophelian can tail the file (or wrap `Pipeline.run`) to ship
rows to its own backend without us imposing one.
