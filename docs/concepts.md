# Concepts

Ophelian has a small core. Five nouns and one verb.

## Pipeline

A `Pipeline` is an ordered list of nodes. It is **declarative** — it
describes *what* to do, not *where* or *how*. The same `Pipeline`
object can run on every supported env without modification.

```python
from ophelian import Pipeline, Data, Train, Eval

pipe = Pipeline([
    Data(source="s3://example/iris.csv", name="iris"),
    Train(model="sklearn.ensemble.RandomForestClassifier",
          data="iris", target="species", name="rf"),
    Eval(model="rf", data="iris", target="species",
         metrics=["accuracy"]),
])
```

## Nodes

Five built-in node types:

| Node | What it does |
|---|---|
| `Data(source=...)` | Load a dataset from local path, S3, GCS, Azure Blob, or an HTTP URL. |
| `Train(model=..., data=..., ...)` | Fit a model on a dataset. |
| `Tune(model=..., data=..., search_space=...)` | Hyperparameter search. |
| `Eval(model=..., data=..., metrics=[...])` | Evaluate a fitted model. |
| `Deploy(model=..., target=...)` | Serve a fitted model behind an endpoint. |

Nodes reference other nodes by `name`. The pipeline is a DAG; there is
no implicit ordering beyond declared dependencies.

## Envs

An `env` answers *where do I run this?* The five built-ins:

- `Standalone(local=True)` — local Docker container per step, with an
  in-process fallback when no daemon is available.
- `AWS(region=..., instance=..., spot=...)` — EC2 (default) or EKS.
- `GCP(project=..., region=..., machine_type=..., gpu_type=..., preemptible=...)`
  — GCE in v1.0 (GKE is post-1.0).
- `Azure(subscription_id=..., resource_group=..., region=..., vm_size=..., spot=...)`
  — Azure VM in v1.0 (AKS is post-1.0).
- `Auto(cheapest_gpu=..., regions=...)` — picks the cheapest provider
  for the requested GPU class.

## Providers

A **Provider** is the object the pipeline talks to. Each env factory
returns a provider (`AWSProvider`, `GCPProvider`, `AzureProvider`,
`StandaloneProvider`, or — through `Auto` — whichever of the above
won the price race).

You almost never instantiate a provider directly; you go through an
env factory.

## Drivers

A **Driver** is how a provider actually launches a step. AWS has
`EC2Driver` (default) and `EKSDriver`. GCP ships `GCEDriver` in v1.0
(`GKEDriver` is on the post-1.0 roadmap). Azure ships
`AzureVMDriver` in v1.0 (`AKSDriver` is on the post-1.0 roadmap).
Each cloud also ships a `Local*Driver` that runs the same logic
in-process — that's what the test suite uses.

## Stores

An **`ArtifactStore`** is the durable surface where step outputs and
checkpoints live. Four implementations:

- `LocalArtifactStore` — filesystem.
- `S3ArtifactStore` — AWS.
- `GCSArtifactStore` — GCP.
- `AzureBlobArtifactStore` — Azure.

All four implement the same `Protocol` (`put`, `get`, `exists`,
`delete`, `list`, `uri`, `put_bytes`, `get_bytes`), so the rest of the
framework treats them interchangeably.

## run_id

Every `pipe.run(...)` gets a unique `run_id`. It propagates through:

- The structured JSON logs (every record gets `"run_id": "..."`).
- Artifact paths (`<prefix>/<run_id>/<step>/...`).
- The end-of-run summary table.
- Spot/preemptible resume (`OPHELIAN_RESUME_FROM=<run_id>`).

You can read it from anywhere with
`ophelian.observability.get_run_id()`.
