# Examples

Three demos to copy-paste, plus the original AWS-only walkthroughs.

## Three viral demos (cloud-agnostic via `Auto`)

| Demo | What it shows | GPU |
| --- | --- | --- |
| [`llama_finetune.py`](llama_finetune.py) | LoRA-fine-tune Llama-3.1-8B + deploy a `/predict` endpoint, on whichever cloud has the cheapest A100 right now | A100 |
| [`resnet_train.py`](resnet_train.py) | Train ResNet-50 on a parquet image dataset with checkpointing & spot interruption recovery | T4 |
| [`xgboost_tabular.py`](xgboost_tabular.py) | The "no GPU required" demo — XGBoost on tabular data with a deployed endpoint | CPU |

Each demo uses `Auto(cheapest_gpu=...)` to pick the cheapest configured cloud. The demo scripts read `OPHELIAN_DRY_RUN` and `OPHELIAN_PROVIDERS` from the environment and forward them to `Auto(...)`, so you can preview the routing decision without spinning up infra:

```bash
OPHELIAN_DRY_RUN=1 python examples/llama_finetune.py
```

To pin to a specific cloud, set `OPHELIAN_PROVIDERS=aws` (or `gcp`, `azure`, or comma-separated subset).

In your own code the equivalent is `Auto(..., dry_run=True, providers=["aws"])`. These env vars are a convenience implemented by the demo scripts (see the `os.environ.get(...)` calls at the top of each one), not a framework-wide convention.

## AWS-specific walkthroughs

The original `examples/aws/` tree shows how to use the `AWS(...)` env directly when you don't want the router. Same DSL, different env constructor — the demos above are the recommended starting point for v1.0.
