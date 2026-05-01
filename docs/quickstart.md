# Quickstart

## Install

```bash
pip install ophelian              # core, runs locally
pip install 'ophelian[aws]'       # + EC2 / S3
pip install 'ophelian[gcp]'       # + GCE / GCS
pip install 'ophelian[azure]'     # + Azure VM / Blob
pip install 'ophelian[all]'       # everything
```

Python **3.11+**.

## Hello, pipeline

```python
from ophelian import Pipeline, Data, Train, Eval, Standalone

pipe = Pipeline([
    Data(source="s3://example/iris.csv", name="iris"),
    Train(model="sklearn.ensemble.RandomForestClassifier",
          data="iris", target="species", name="rf"),
    Eval(model="rf", data="iris", target="species",
         metrics=["accuracy", "f1_macro"]),
])

pipe.run(env=Standalone(local=True))
```

Run it:

```bash
python hello.py
```

You'll see a structured JSON log per step and a rich table summary at
the end with `step`, `env`, `instance`, `duration_s`, and
`cost_estimate_usd`.

## Move to the cloud — no pipeline changes

```python
from ophelian import AWS

pipe.run(env=AWS(region="us-east-1", instance="g5.xlarge", spot=True))
```

Or let Ophelian pick the cheapest GPU across clouds:

```python
from ophelian import Auto

pipe.run(env=Auto(cheapest_gpu="A100",
                  regions=["us-east-1", "us-central1", "eastus"]))
```

## Dry-run the cost router

```bash
OPHELIAN_DRY_RUN=1 python examples/llama_finetune.py
```

This prints the chosen provider/region/instance and the static price
estimate without provisioning anything.

## Next steps

- [Concepts](concepts.md) — pipelines, nodes, envs, providers, drivers,
  stores, run_id.
- [Cookbook](cookbook.md) — common patterns (fine-tuning, distributed
  training, deploy targets).
- [Troubleshooting](troubleshooting.md) — when it doesn't work.
