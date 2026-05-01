"""GCP env — Google Cloud backend for Ophelian pipelines.

Pipeline authors only ever interact with the friendly :func:`GCP` factory::

    from ophelian.envs import GCP

    env = GCP(
        project="my-gcp-project",
        region="us-central1",
        machine_type="n1-standard-8",
        gpu_type="nvidia-tesla-t4",
        gpu_count=1,
        spot=True,
    )
    pipe.run(env=env)

The factory builds a :class:`GCPConfig` and hands it to a
:class:`~ophelian.providers.gcp.GCPProvider`. The public API mirrors
:func:`~ophelian.envs.aws.AWS` so a user can swap clouds without changing
the pipeline definition — that's the central promise of the multi-cloud
release.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

if TYPE_CHECKING:  # pragma: no cover
    from ophelian.providers.gcp import GCPProvider


# Curated map of GCP regions used as a sanity check; anything that
# *looks* like a region (``<continent>-<location><digit>``) is also
# accepted so we stay compatible with newly-launched regions.
KNOWN_REGIONS: tuple[str, ...] = (
    "us-central1",
    "us-east1",
    "us-east4",
    "us-east5",
    "us-west1",
    "us-west2",
    "us-west3",
    "us-west4",
    "europe-west1",
    "europe-west2",
    "europe-west3",
    "europe-west4",
    "europe-west6",
    "europe-north1",
    "asia-east1",
    "asia-east2",
    "asia-northeast1",
    "asia-southeast1",
    "asia-south1",
    "australia-southeast1",
    "southamerica-east1",
    "northamerica-northeast1",
)

GPU_TYPES: tuple[str, ...] = (
    "nvidia-tesla-t4",
    "nvidia-tesla-v100",
    "nvidia-tesla-p100",
    "nvidia-tesla-p4",
    "nvidia-tesla-a100",
    "nvidia-a100-80gb",
    "nvidia-l4",
    "nvidia-h100-80gb",
)

_INSTANCE_RE = re.compile(r"^[a-z0-9]+-[a-z0-9]+(-[a-z0-9]+)*$")
_REGION_RE = re.compile(r"^[a-z]+-[a-z]+\d$")
_ZONE_RE = re.compile(r"^[a-z]+-[a-z]+\d-[a-z]$")

CloudBackend = Literal["gce", "gke"]


class GCPConfig(BaseModel):
    """Strongly-typed configuration for the GCP env.

    Mirrors :class:`~ophelian.envs.aws.AWSConfig` field-for-field where
    possible so the public API of multi-cloud pipelines stays identical.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    project: str = Field(description="GCP project id (e.g. ``my-team-prod``).")
    region: str = Field(description="GCP region (e.g. ``us-central1``).")
    zone: str | None = Field(
        default=None,
        description=(
            "Optional zone — defaults to ``{region}-a``. Required for some"
            " GPU/A2 machine types only available in specific zones."
        ),
    )
    machine_type: str = Field(
        default="n1-standard-4",
        description="GCE machine type (``n1-standard-4``, ``a2-highgpu-1g``, ...).",
    )
    gpu_type: str | None = Field(
        default=None,
        description="Accelerator type (e.g. ``nvidia-tesla-t4``). ``None`` for CPU-only.",
    )
    gpu_count: int = Field(default=0, ge=0, description="Number of accelerators to attach.")
    spot: bool = Field(
        default=False,
        description="Request a preemptible / Spot VM. Cheaper but reclaimable.",
    )
    image_family: str = Field(
        default="debian-12",
        description="Image family used to boot the worker (resolved via Compute API).",
    )
    image_project: str = Field(
        default="debian-cloud",
        description="Project that owns ``image_family``.",
    )
    network: str = Field(default="default", description="VPC network name.")
    subnet: str | None = Field(default=None, description="Optional subnet (region-scoped).")
    service_account: str | None = Field(
        default=None,
        description=(
            "Service-account email attached to the VM. Required when the"
            " worker reads/writes GCS artifacts; documented in"
            " ``docs/envs/gcp.md``."
        ),
    )
    artifact_bucket: str | None = Field(
        default=None,
        description=(
            "GCS bucket used as the artifact store. When omitted, the"
            " provider creates one named ``ophelian-artifacts-{project}``"
            " on first use."
        ),
    )
    artifact_prefix: str = Field(
        default="runs",
        description="Object name prefix under which run artifacts are written.",
    )
    gcp_backend: CloudBackend = Field(
        default="gce",
        description="``gce`` (default) provisions VMs directly. ``gke`` submits to GKE.",
    )
    gke_cluster: str | None = Field(
        default=None,
        description="GKE cluster name. Required when ``gcp_backend='gke'``.",
    )
    gke_namespace: str = Field(
        default="default",
        description="Kubernetes namespace used for GKE workloads.",
    )
    runtime_image: str = Field(
        default="python:3.12-slim",
        description="Container image executed on the worker. Override to use Artifact Registry.",
    )
    runtime_extras: tuple[str, ...] = Field(
        default=("sklearn",),
        description="Pyproject extras installed alongside ophelian inside the worker container.",
    )
    pipeline_archive_uri: str | None = Field(
        default=None,
        description=(
            "Optional ``gs://bucket/key.tar.gz`` containing the pipeline source."
            " When set, the worker downloads and ``pip install``s it."
        ),
    )
    user_data: str | None = Field(
        default=None,
        description="Extra startup-script appended to the worker's user-data (advanced).",
    )
    instance_tags: dict[str, str] = Field(
        default_factory=lambda: {"ophelian-managed": "true"},
        description="Labels applied to every GCP resource the provider creates.",
    )
    timeout_seconds: int = Field(
        default=60 * 60 * 6,
        ge=1,
        description="Hard timeout per step before the provider tears down the worker.",
    )
    poll_interval_seconds: float = Field(
        default=10.0,
        gt=0.0,
        description="Polling interval used while waiting for instance/job state changes.",
    )
    resume_run_id: str | None = Field(
        default=None,
        description=(
            "Existing run id to resume. Set automatically by the provider"
            " after a preemption so a subsequent ``pipe.run`` continues from"
            " the latest checkpoint."
        ),
    )

    # ---- validators ---------------------------------------------------

    @field_validator("region")
    @classmethod
    def _valid_region(cls, value: str) -> str:
        if not value or not isinstance(value, str):
            raise ValueError("region must be a non-empty string")
        if value not in KNOWN_REGIONS and not _REGION_RE.match(value):
            raise ValueError(
                f"region={value!r} does not look like a valid GCP region "
                f"(expected e.g. 'us-central1'). Known: {', '.join(KNOWN_REGIONS)}"
            )
        return value

    @field_validator("zone")
    @classmethod
    def _valid_zone(cls, value: str | None) -> str | None:
        if value is None:
            return value
        if not _ZONE_RE.match(value):
            raise ValueError(
                f"zone={value!r} is not a valid GCP zone (expected e.g. 'us-central1-a')."
            )
        return value

    @field_validator("machine_type")
    @classmethod
    def _valid_machine(cls, value: str) -> str:
        if not _INSTANCE_RE.match(value):
            raise ValueError(
                f"machine_type={value!r} is not a valid GCP machine type "
                "(expected e.g. 'n1-standard-4', 'a2-highgpu-1g')."
            )
        return value

    @field_validator("gpu_type")
    @classmethod
    def _valid_gpu(cls, value: str | None) -> str | None:
        if value is None:
            return value
        if not value.startswith("nvidia-"):
            raise ValueError(
                f"gpu_type={value!r} must start with 'nvidia-' (known: {', '.join(GPU_TYPES)})"
            )
        return value

    @field_validator("artifact_prefix")
    @classmethod
    def _valid_prefix(cls, value: str) -> str:
        return value.strip("/")

    @field_validator("runtime_extras")
    @classmethod
    def _valid_extras(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(v.strip() for v in value if v.strip())

    @model_validator(mode="after")
    def _backend_consistency(self) -> GCPConfig:
        if self.gcp_backend == "gke" and not self.gke_cluster:
            raise ValueError(
                "gcp_backend='gke' requires `gke_cluster=<name>` so the provider can submit Jobs."
            )
        if self.spot and self.gcp_backend == "gke":
            raise ValueError(
                "Preemptible scheduling is managed by the GKE node pool, not by "
                "the Ophelian provider — drop `spot=True` when targeting GKE."
            )
        if self.gpu_count > 0 and self.gpu_type is None:
            raise ValueError("gpu_count > 0 requires gpu_type='nvidia-...'.")
        if self.gpu_type is not None and self.gpu_count == 0:
            object.__setattr__(self, "gpu_count", 1)
        # Default zone to <region>-a if not provided.
        if self.zone is None:
            object.__setattr__(self, "zone", f"{self.region}-a")
        return self

    # ---- ergonomics ----------------------------------------------------

    @property
    def is_gpu_instance(self) -> bool:
        return self.gpu_count > 0 and self.gpu_type is not None

    @property
    def effective_zone(self) -> str:
        return self.zone or f"{self.region}-a"

    def with_resume(self, run_id: str) -> GCPConfig:
        """Return a copy of this config with ``resume_run_id`` set."""
        return self.model_copy(update={"resume_run_id": run_id})


def GCP(
    *,
    project: str,
    region: str,
    machine_type: str = "n1-standard-4",
    gpu_type: str | None = None,
    gpu_count: int = 0,
    spot: bool = False,
    preemptible: bool | None = None,
    gcp_backend: CloudBackend = "gce",
    artifact_bucket: str | None = None,
    gke_cluster: str | None = None,
    driver: Any | None = None,
    store: Any | None = None,
    **kwargs: Any,
) -> GCPProvider:
    """Build a :class:`~ophelian.providers.gcp.GCPProvider`.

    The keyword-only signature mirrors :class:`GCPConfig` for the
    parameters pipeline authors typically reach for. ``driver`` and
    ``store`` are wiring overrides reserved for tests.

    ``preemptible`` is accepted as a backward-compatible alias for
    ``spot`` so docs and examples written against either name keep
    working — Google's own console has migrated from "preemptible" to
    "Spot VMs" but both terms remain in widespread use.
    """
    from ophelian.providers.gcp import GCPProvider

    if preemptible is not None:
        if spot and preemptible != spot:
            raise ValueError(
                "GCP() received conflicting values for `spot` and "
                "`preemptible` — pass only one (they are aliases)."
            )
        spot = preemptible

    config = GCPConfig(
        project=project,
        region=region,
        machine_type=machine_type,
        gpu_type=gpu_type,
        gpu_count=gpu_count,
        spot=spot,
        gcp_backend=gcp_backend,
        artifact_bucket=artifact_bucket,
        gke_cluster=gke_cluster,
        **kwargs,
    )
    return GCPProvider(config=config, driver=driver, store=store)


__all__ = [
    "GCP",
    "GPU_TYPES",
    "KNOWN_REGIONS",
    "CloudBackend",
    "GCPConfig",
]
