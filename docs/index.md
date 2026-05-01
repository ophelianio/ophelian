# Ophelian

> Write your ML pipeline once. Run it anywhere. Pay the lowest GPU
> price on the market.

Ophelian is a small, opinionated Python framework for taking ML / AI
prototypes to production without rewriting them every time the runtime
changes.

```python
from ophelian import Pipeline, Train, Auto

pipe = Pipeline([
    Train(
        model="meta-llama/Llama-3.2-1B",
        data="s3://my-bucket/dataset.jsonl",
        epochs=3,
    ),
])

# Picks the cheapest A100 across AWS / GCP / Azure right now.
pipe.run(env=Auto(cheapest_gpu="A100"))
```

The same `pipe` runs on:

- `Standalone(local=True)` — local Docker, or a pure in-process
  fallback when no daemon is present.
- `AWS(...)` — EC2 or EKS, S3-backed artifacts.
- `GCP(...)` — GCE-backed (GKE is post-1.0), GCS-backed artifacts.
- `Azure(...)` — Azure-VM-backed (AKS is post-1.0), Blob-backed artifacts.
- `Auto(cheapest_gpu=...)` — picks the cheapest provider/region for
  the GPU class you ask for.

## Five things to know

1. **Pipelines are declarative.** Nodes describe *what* to do. Envs
   describe *where* to do it. The two are deliberately independent.
2. **Local-first.** Every cloud code path is reachable through a
   `Local*Driver`, so contributors and CI can run the suite without
   any cloud credentials.
3. **Spot is a first-class citizen.** Checkpoints + resume work
   identically on EC2 spot, GCE preemptible, and Azure Spot.
4. **Honest cost telemetry.** Every step records a price estimate
   pulled from the static price table (refreshed quarterly, see
   `CONTRIBUTING.md`). The end-of-run summary table tells you the
   real number.
5. **Plug-in envs.** Third-party clouds register through the
   `ophelian.envs` entry-point group — no fork required.

## Where to next

- New here? → [Quickstart](quickstart.md)
- Want to understand the model? → [Concepts](concepts.md)
- Cloud-specific docs → [AWS](envs/aws.md) · [GCP](envs/gcp.md) ·
  [Azure](envs/azure.md) · [Standalone](envs/standalone.md) ·
  [Auto](envs/auto.md)
- Stuck? → [Troubleshooting](troubleshooting.md)
