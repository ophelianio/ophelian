# Ophelian AWS examples

Three end-to-end pipelines that demonstrate the AWS env. Pick the one that
matches your workload, set a handful of environment variables, and run it.

| Script | Workload | Default instance | Notes |
| --- | --- | --- | --- |
| `xgboost_tabular.py` | Tabular classification with XGBoost | `m5.large` (CPU) | Cheapest; good for smoke-testing your AWS setup |
| `pytorch_resnet.py`  | ResNet image classification | `g4dn.xlarge` (T4 GPU) | Demonstrates spot + resume on GPU |
| `huggingface_llm.py` | LoRA fine-tuning of a chat LM | `g5.2xlarge` (A10G GPU) | Heaviest workload, fully spot-friendly |

## Common setup

```bash
pip install 'ophelian[aws,xgboost,pytorch,huggingface]'

aws configure       # or rely on env vars / IAM role
aws s3 mb s3://my-ophelian-bucket --region us-east-1

export OPHELIAN_AWS_REGION=us-east-1
export OPHELIAN_AWS_BUCKET=my-ophelian-bucket
export OPHELIAN_AWS_INSTANCE_PROFILE=ophelian-worker  # IAM instance profile
```

See [`docs/aws.md`](../../docs/aws.md) for the minimal IAM policy and
permission set the instance profile must have.

## Resuming a spot interruption

Each script reads ``OPHELIAN_AWS_RESUME_RUN_ID`` and forwards it to
``AWS(...)``. When a step fails with ``status=failed`` and
``info["resumable"] is True``, copy ``info["run_id"]`` into that env var
and re-run — the provider will pick up from the last checkpoint instead of
starting over.
