# Auto

```python
from ophelian import Auto

env = Auto(
    cheapest_gpu="A100",
    regions=["us-east-1", "us-central1", "eastus"],
    providers=("aws", "gcp", "azure"),  # restrict the search
)
```

`Auto` is the **cost router**. You tell it which GPU class you need
and which regions you'll accept; it returns whichever provider is
cheapest *right now*.

## How it picks

1. Look up the requested GPU in `ophelian.pricing` for every
   `(provider, region)` pair you've allowed.
2. Filter out providers whose credentials aren't configured (you'll
   see a one-line note about each one we skipped).
3. Pick the lowest total $/hr (on-demand by default; spot price when
   `spot=True`).
4. Construct the matching env (`AWS(...)`, `GCP(...)`, or
   `Azure(...)`) and return its provider.

The decision is logged as a structured event so you have an audit
trail.

## Constructor

| Field | Default | Description |
|---|---|---|
| `cheapest_gpu` | — | One of `T4`, `L4`, `V100`, `A10G`, `A100`, `H100`. |
| `regions` | all known regions | Restrict the search. |
| `providers` | `("aws", "gcp", "azure")` | Restrict the search. |
| `spot` | `True` | Score against spot/preemptible/Spot prices. |
| `dry_run` | `False` | Print the decision and return a no-op provider. |
| `prefer` | `None` | Tie-break preference (e.g. `"gcp"`). |
| `gpu_count` | `1` | How many of the GPU class you need on one node. |
| `extra_kwargs` | `{}` | Forwarded to the chosen provider's env. |

## Dry-run

```bash
OPHELIAN_DRY_RUN=1 python examples/llama_finetune.py
```

The router prints something like:

```text
[ophelian.auto] cheapest A100 right now:
  gcp / us-central1  / a2-highgpu-1g  preemptible  $1.12/hr
  aws / us-east-1    / p4d.24xlarge   spot         $1.18/hr
  azure / eastus     / NC24ads_A100_v4 spot        $1.34/hr
=> picked gcp/us-central1
```

…and returns a no-op provider so the rest of the pipeline runs as a
no-op too. Perfect for CI or for sanity-checking before you spend
real money.

## When *not* to use Auto

- When you have committed-use discounts or reserved capacity on a
  specific cloud — pin the env directly.
- When data gravity matters (your dataset is in S3 → use AWS).
- When your security review is per-cloud — pin the env so audit logs
  are predictable.

## Pricing data

`Auto(...)` defaults to `allow_live=True`. The router consults each
cloud through the cheapest free public path it can reach:

- **AWS** — calls `ec2.describe_spot_price_history` for instance types
  in our static table that match `cheapest_gpu`. Free, public, only
  needs standard boto3 credentials. No AWS credentials? It silently
  falls back to the static table.
- **Azure** — hits the public **Retail Prices API**
  (`https://prices.azure.com/api/retail/prices`). No auth, no API
  key. Returns true on-demand and Spot prices for every Azure VM
  SKU; we filter to the SKUs in our static table and skip the
  Windows variants. Source label: `azure-retail-api`.
- **GCP** — uses the **Cloud Billing Catalog API**
  (`https://cloudbilling.googleapis.com/v1/services/6F81-5844-456A`)
  when the `GOOGLE_API_KEY` env var is set. The catalog only
  exposes per-SKU prices (compute / RAM / GPU billed separately) so
  we publish a *live GPU rate × `gpu_count` + static compute base*
  aggregate. The static-compute base comes from subtracting the
  table's implied GPU rate from each instance row. Source label:
  `gcp-billing-catalog` with a `notes` field that spells out the
  split. Without `GOOGLE_API_KEY`, GCP gracefully falls back to
  the static table.

All three live paths share one on-disk cache
(`OPHELIAN_PRICING_CACHE_DIR`, default
`~/.cache/ophelian/pricing.json`) with a 24 h TTL so subsequent
`Auto(...)` calls are fully offline.

Pass `allow_live=False` to force fully-offline routing — useful for
deterministic CI snapshots. The static table lives in
`ophelian/pricing/__init__.py` and carries a
`STATIC_PRICES_LAST_REVIEW` date; PRs to refresh it are welcome.

## Run summary: duration & cost

Every `pipe.run(...)` ends with a one-line-per-step table (or a
structured `run.summary` JSON record when
`OPHELIAN_LOG_FORMAT=json`). When the provider was constructed by
`Auto(...)` the table includes a **Duration** column and a
**Cost (USD)** column computed as `router_quote_hourly_usd * (duration / 3600)`.
GPU utilisation is **not** collected by Ophelian itself — wire up an
in-VM `nvidia-smi` sidecar if you need it; the column is reserved
(`gpu_utilization: null`) in the JSON payload so you can fill it in.
