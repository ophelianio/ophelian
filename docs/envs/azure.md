# Azure

```python
from ophelian import Azure

env = Azure(
    subscription_id="00000000-0000-0000-0000-000000000000",
    resource_group="ml-rg",
    region="eastus",
    vm_size="Standard_NC6s_v3",
    spot=True,
    artifact_account="myophelianstore",
    artifact_container="artifacts",
)
```

Azure runs your pipeline on **Azure VMs** in v1.0. An `AKSDriver` is
on the post-1.0 roadmap; `Azure(..., azure_backend='aks')` raises a
clear `NotImplementedError` today.

## Install

```bash
pip install 'ophelian[azure]'   # VM driver + Blob store
```

## Constructor

| Field | Default | Description |
|---|---|---|
| `subscription_id` | env var | Azure subscription. |
| `resource_group` | env var | RG that owns the VM and disk. |
| `region` | — | Azure region (e.g. `eastus`). |
| `vm_size` | `Standard_NC6s_v3` | VM SKU; pick one with the GPU you need. |
| `spot` | `False` | Use an Azure Spot VM. |
| `max_spot_price` | `-1` (pay up to on-demand) | Spot price ceiling, USD/hr. |
| `artifact_account` | env var | Storage account for Blob artifacts. |
| `artifact_container` | `"ophelian"` | Container inside the storage account. |
| `artifact_prefix` | `""` | Blob prefix inside the container. |
| `vnet` / `subnet` | created on demand | Networking. |
| `runtime_image` | `python:3.12-slim` | Container image (reserved for the post-1.0 AKS driver). |
| `runtime_extras` | `()` | Pip extras installed at boot. |
| `timeout_seconds` | `3600` | Per-step ceiling. |
| `poll_interval_seconds` | `15` | How often we poll Blob for `result.json`. |
| `resume_run_id` | `None` | Resume from a previous run's checkpoints. |

## Drivers

- `AzureVMDriver` (default) — provisions a per-step VM (Spot when
  `spot=True`) with a cloud-init custom data script. Polls Blob for
  `result.json` and **always** deletes the VM + NIC + disk in a
  `finally` block.
- `LocalAzureDriver` — used by tests.
- `AKSDriver` — **post-1.0 roadmap**. The `azure_backend='aks'` switch
  is wired but raises `NotImplementedError` until the driver lands.

## Spot

`spot=True` enables Azure Spot. The same SIGTERM checkpoint upload
machinery used by AWS / GCP is wired up here too, so an evicted Spot
VM picks back up via `OPHELIAN_RESUME_FROM=<run_id>` on the next run.

## Authentication

`azure-identity` walks `DefaultAzureCredential`:

1. `AZURE_CLIENT_ID` / `AZURE_CLIENT_SECRET` / `AZURE_TENANT_ID` env
   vars (CI-friendly).
2. `az login` (local).
3. Managed identity (when running inside Azure).
