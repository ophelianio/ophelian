# Examples

Three demos to copy-paste, plus the original AWS-only walkthroughs.

## Three viral demos (cloud-agnostic via `Auto`)

| Demo | What it shows | GPU |
| --- | --- | --- |
| [`llama_finetune.py`](llama_finetune.py) | LoRA-fine-tune Llama-3.1-8B + deploy a `/predict` endpoint, on whichever cloud has the cheapest A100 right now | A100 |
| [`resnet_train.py`](resnet_train.py) | Train ResNet-50 on a parquet image dataset with checkpointing & spot interruption recovery | T4 |
| [`xgboost_tabular.py`](xgboost_tabular.py) | The "no GPU required" demo — XGBoost on tabular data with a deployed endpoint | CPU |

Each demo uses `Auto(cheapest_gpu=...)` to pick the cheapest configured cloud. Set `OPHELIAN_DRY_RUN=1` to just print the routing decision without spinning up infra:

```bash
OPHELIAN_DRY_RUN=1 python examples/llama_finetune.py
```

To pin to a specific cloud, set `OPHELIAN_PROVIDERS=aws` (or `gcp`, `azure`, or comma-separated subset).

## AWS-specific walkthroughs

The original `examples/aws/` tree shows how to use the `AWS(...)` env directly when you don't want the router. Same DSL, different env constructor — the demos above are the recommended starting point for v1.0.
