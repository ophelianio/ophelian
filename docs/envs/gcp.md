# GCP

```python
from ophelian import GCP

env = GCP(
    project="my-gcp-project",
    region="us-central1",
    zone="us-central1-a",
    machine_type="n1-standard-4",
    gpu_type="nvidia-tesla-t4",
    gpu_count=1,
    preemptible=True,
    artifact_bucket="my-ophelian-artifacts",
)
```

GCP runs your pipeline on **GCE** in v1.0. A `GKEDriver` is on the
post-1.0 roadmap; `GCP(..., gcp_backend='gke')` raises a clear
`NotImplementedError` today.

## Install

```bash
pip install 'ophelian[gcp]'    # GCE driver + GCS store
```

## Constructor

| Field | Default | Description |
|---|---|---|
| `project` | — | Required GCP project ID. |
| `region` | — | Required GCE region (e.g. `us-central1`). |
| `zone` | first zone in region | GCE zone. |
| `machine_type` | `n1-standard-4` | GCE machine type. |
| `gpu_type` | `None` | e.g. `nvidia-tesla-t4`, `nvidia-tesla-a100`. |
| `gpu_count` | `0` | Number of accelerators. |
| `preemptible` | `False` | Use a preemptible (Spot) VM. |
| `artifact_bucket` | env var | GCS bucket for artifacts. |
| `artifact_prefix` | `"ophelian"` | Object prefix inside the bucket. |
| `network` / `subnet` | `default` | VPC. |
| `service_account` | default compute SA | SA the VM runs as. |
| `runtime_image` | `python:3.12-slim` | Container image (reserved for the post-1.0 GKE driver). |
| `runtime_extras` | `()` | Pip extras installed at boot. |
| `timeout_seconds` | `3600` | Per-step ceiling. |
| `poll_interval_seconds` | `15` | How often we poll GCS for `result.json`. |
| `resume_run_id` | `None` | Resume from a previous run's checkpoints. |
| `instance_tags` | `()` | Network tags applied to the VM. |

## Drivers

- `GCEDriver` (default) — provisions a per-step GCE VM with a
  cloud-init startup script that pip-installs the right extras,
  decodes the step spec, and execs `python -m ophelian.runtime.step_runner`.
  Polls GCS for the step's `result.json` and **always** terminates the
  VM in a `finally` block.
- `LocalGCPDriver` — used by tests; runs the step in-process while
  going through the real GCS code path (with a stub client).
- `GKEDriver` — **post-1.0 roadmap**. The `gcp_backend='gke'` switch
  is wired but raises `NotImplementedError` until the driver lands.

## Spot / preemptible

`preemptible=True` tells GCE to provision a preemptible VM. The
`step_runner` installs a SIGTERM handler that uploads the latest
checkpoint to GCS before exiting, and the next run with
`OPHELIAN_RESUME_FROM=<run_id>` will pick up where the previous one
stopped — exactly the same contract as AWS spot.

## Authentication

`google-cloud-storage` and `google-cloud-compute` use Application
Default Credentials. The simplest local setup:

```bash
gcloud auth application-default login
```

In CI, mount a service account JSON and set
`GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json`.
