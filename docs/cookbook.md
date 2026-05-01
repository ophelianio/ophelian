# Cookbook

Short, copy-paste recipes for the patterns people hit most often.

## Fine-tune a Llama on the cheapest A100

```python
from ophelian import Pipeline, Train, Auto

Pipeline([
    Train(
        model="meta-llama/Llama-3.2-1B",
        data="s3://my-bucket/sft.jsonl",
        epochs=3, lr=2e-5, lora=True,
    ),
]).run(env=Auto(cheapest_gpu="A100"))
```

## Train a ResNet on preemptible GPUs and resume on eviction

```python
from ophelian import Pipeline, Train, GCP

env = GCP(
    project="my-proj", region="us-central1",
    machine_type="n1-standard-8",
    gpu_type="nvidia-tesla-t4", gpu_count=1,
    preemptible=True,
    artifact_bucket="my-ophelian-artifacts",
)

Pipeline([
    Train(
        model="torchvision.models.resnet50",
        data="gs://my-bucket/imagenet/",
        epochs=90, batch_size=256, checkpoint_every="5min",
    ),
]).run(env=env)
```

If the VM gets preempted, re-run with
`OPHELIAN_RESUME_FROM=<previous_run_id>` and Ophelian picks up at the
last checkpoint automatically.

## XGBoost on tabular data — sub-$0.05/run

```python
from ophelian import Pipeline, Data, Train, Eval, Auto

Pipeline([
    Data(source="s3://my-bucket/titanic.parquet", name="t"),
    Train(model="xgboost.XGBClassifier", data="t", target="survived",
          hyperparams={"n_estimators": 400, "tree_method": "hist"},
          name="xgb"),
    Eval(model="xgb", data="t", target="survived",
         metrics=["accuracy", "roc_auc"]),
]).run(env=Auto(cheapest_gpu=None))   # CPU-only auto-pick
```

## Pin to a single cloud (data gravity)

When your dataset lives in one cloud, you usually want the compute
there too:

```python
from ophelian import AWS

env = AWS(region="us-east-1", instance="g5.xlarge", spot=True)
```

## JSON logs + a Slack-paste-ready summary

```python
from ophelian.observability import configure_logging
configure_logging(json=True)

pipe.run(env=...)   # rich summary table is printed at the end
```

## Distributed training (multi-GPU on one node)

```python
from ophelian import GCP, Pipeline, Train

GCP(
    project="...", region="us-central1",
    machine_type="a2-highgpu-8g",
    gpu_type="nvidia-tesla-a100", gpu_count=8,
)
```

The `huggingface` adapter wires `accelerate launch` automatically when
`gpu_count > 1`. The `pytorch` adapter wires `torchrun`.

## Custom env (third-party cloud)

Register your env factory in your distribution's `pyproject.toml`:

```toml
[project.entry-points."ophelian.envs"]
RunPod = "my_pkg.envs:RunPod"
```

Once installed, `from ophelian.envs import discover_plugin_envs;
discover_plugin_envs()["RunPod"](...)` returns your env, and `Auto`
will consider it during the price race if you pass it explicitly via
`Auto(..., providers=("aws", "runpod"))`.
