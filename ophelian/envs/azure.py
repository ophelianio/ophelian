"""Azure env — Microsoft Azure backend for Ophelian pipelines.

Pipeline authors only ever interact with the friendly :func:`Azure` factory::

    from ophelian.envs import Azure

    env = Azure(
        subscription_id="00000000-0000-0000-0000-000000000000",
        resource_group="ophelian-rg",
        location="eastus",
        vm_size="Standard_NC6s_v3",
        spot=True,
    )
    pipe.run(env=env)

The factory builds an :class:`AzureConfig` and hands it to a
:class:`~ophelian.providers.azure.AzureProvider`. The public API mirrors
:func:`~ophelian.envs.aws.AWS` and :func:`~ophelian.envs.gcp.GCP`.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

if TYPE_CHECKING:  # pragma: no cover
    from ophelian.providers.azure import AzureProvider

# Most-common Azure regions (= "locations"). Anything that *looks* like
# a location is also accepted so we stay compatible with newly-launched
# regions.
KNOWN_REGIONS: tuple[str, ...] = (
    "eastus",
    "eastus2",
    "westus",
    "westus2",
    "westus3",
    "centralus",
    "northcentralus",
    "southcentralus",
    "westcentralus",
    "northeurope",
    "westeurope",
    "uksouth",
    "ukwest",
    "francecentral",
    "germanywestcentral",
    "switzerlandnorth",
    "swedencentral",
    "norwayeast",
    "polandcentral",
    "italynorth",
    "spaincentral",
    "eastasia",
    "southeastasia",
    "japaneast",
    "japanwest",
    "koreacentral",
    "australiaeast",
    "australiasoutheast",
    "centralindia",
    "southindia",
    "uaenorth",
    "southafricanorth",
    "brazilsouth",
    "canadacentral",
    "canadaeast",
)

GPU_VM_FAMILIES: tuple[str, ...] = (
    "Standard_NC",
    "Standard_ND",
    "Standard_NV",
)

_VM_SIZE_RE = re.compile(r"^Standard_[A-Za-z0-9_]+$")
_LOCATION_RE = re.compile(r"^[a-z][a-z0-9]+$")
_SUBSCRIPTION_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)

CloudBackend = Literal["vm", "aks"]


class AzureConfig(BaseModel):
    """Strongly-typed configuration for the Azure env.

    Field names follow Azure SDK conventions (``vm_size``,
    ``resource_group``, ``location``) so the surface is familiar to
    Azure users, but the mental model and lifecycle are identical to
    :class:`~ophelian.envs.aws.AWSConfig`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    subscription_id: str = Field(description="Azure subscription GUID.")
    resource_group: str = Field(description="Resource group used for every Ophelian resource.")
    location: str = Field(description="Azure location/region (e.g. ``eastus``).")
    vm_size: str = Field(
        default="Standard_D4s_v5",
        description="Azure VM size (``Standard_D4s_v5``, ``Standard_NC6s_v3``, ...).",
    )
    spot: bool = Field(
        default=False,
        description="Request a Spot VM. Cheaper but reclaimable with 30 s notice.",
    )
    spot_max_price: float | None = Field(
        default=None,
        description="Maximum spot bid in USD/hour. Defaults to the on-demand price (-1).",
    )
    image_publisher: str = Field(
        default="canonical",
        description="VM image publisher.",
    )
    image_offer: str = Field(
        default="0001-com-ubuntu-server-jammy",
        description="VM image offer.",
    )
    image_sku: str = Field(default="22_04-lts-gen2", description="VM image SKU.")
    image_version: str = Field(default="latest", description="VM image version.")
    vnet: str | None = Field(default=None, description="Virtual network name.")
    subnet: str | None = Field(default=None, description="Subnet name.")
    managed_identity: str | None = Field(
        default=None,
        description=(
            "User-assigned managed identity (resource id) attached to the"
            " VM so it can read/write Azure Blob artifacts."
        ),
    )
    artifact_account: str | None = Field(
        default=None,
        description="Storage account for the artifact container.",
    )
    artifact_container: str = Field(
        default="ophelian-artifacts",
        description="Blob container used as the artifact store.",
    )
    artifact_prefix: str = Field(
        default="runs",
        description="Blob name prefix under which run artifacts are written.",
    )
    azure_backend: CloudBackend = Field(
        default="vm",
        description="``vm`` (default) provisions VMs directly. ``aks`` submits to AKS.",
    )
    aks_cluster: str | None = Field(
        default=None,
        description="AKS cluster name. Required when ``azure_backend='aks'``.",
    )
    aks_namespace: str = Field(
        default="default",
        description="Kubernetes namespace used for AKS workloads.",
    )
    runtime_image: str = Field(
        default="python:3.12-slim",
        description="Container image executed on the worker. Override to use ACR.",
    )
    runtime_extras: tuple[str, ...] = Field(
        default=("sklearn",),
        description="Pyproject extras installed alongside ophelian inside the worker container.",
    )
    pipeline_archive_uri: str | None = Field(
        default=None,
        description=(
            "Optional ``az://account/container/key.tar.gz`` containing the pipeline source."
        ),
    )
    user_data: str | None = Field(
        default=None,
        description="Extra cloud-init script appended to the worker user-data (advanced).",
    )
    instance_tags: dict[str, str] = Field(
        default_factory=lambda: {"ophelian_managed": "true"},
        description="Tags applied to every Azure resource the provider creates.",
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
            " after a Spot eviction so a subsequent ``pipe.run`` continues"
            " from the latest checkpoint."
        ),
    )

    # ---- validators ---------------------------------------------------

    @field_validator("subscription_id")
    @classmethod
    def _valid_subscription(cls, value: str) -> str:
        if not _SUBSCRIPTION_RE.match(value):
            raise ValueError(f"subscription_id={value!r} does not look like a valid Azure GUID")
        return value

    @field_validator("location")
    @classmethod
    def _valid_location(cls, value: str) -> str:
        if not value or not isinstance(value, str):
            raise ValueError("location must be a non-empty string")
        if value not in KNOWN_REGIONS and not _LOCATION_RE.match(value):
            raise ValueError(
                f"location={value!r} does not look like a valid Azure region "
                f"(expected e.g. 'eastus'). Known: {', '.join(KNOWN_REGIONS)}"
            )
        return value

    @field_validator("vm_size")
    @classmethod
    def _valid_vm(cls, value: str) -> str:
        if not _VM_SIZE_RE.match(value):
            raise ValueError(
                f"vm_size={value!r} is not a valid Azure VM size (expected e.g. 'Standard_D4s_v5')."
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
    def _backend_consistency(self) -> AzureConfig:
        if self.azure_backend == "aks" and not self.aks_cluster:
            raise ValueError(
                "azure_backend='aks' requires `aks_cluster=<name>` so the provider can submit Jobs."
            )
        if self.spot and self.azure_backend == "aks":
            raise ValueError(
                "Spot scheduling is managed by the AKS node pool, not by the "
                "Ophelian provider — drop `spot=True` when targeting AKS."
            )
        return self

    # ---- ergonomics ----------------------------------------------------

    @property
    def is_gpu_instance(self) -> bool:
        return any(self.vm_size.startswith(p) for p in GPU_VM_FAMILIES)

    def with_resume(self, run_id: str) -> AzureConfig:
        """Return a copy of this config with ``resume_run_id`` set."""
        return self.model_copy(update={"resume_run_id": run_id})


def Azure(
    *,
    subscription_id: str,
    resource_group: str,
    location: str | None = None,
    region: str | None = None,
    vm_size: str = "Standard_D4s_v5",
    spot: bool = False,
    azure_backend: CloudBackend = "vm",
    artifact_account: str | None = None,
    artifact_container: str = "ophelian-artifacts",
    aks_cluster: str | None = None,
    driver: Any | None = None,
    store: Any | None = None,
    **kwargs: Any,
) -> AzureProvider:
    """Build an :class:`~ophelian.providers.azure.AzureProvider`.

    ``region`` is accepted as a backward-compatible alias for
    ``location`` so AWS/GCP-flavoured docs (``region=...``) and
    Azure-native docs (``location=...``) are both valid.
    """
    from ophelian.providers.azure import AzureProvider

    if location is None and region is None:
        raise TypeError("Azure() requires either `location=...` or `region=...`")
    if location is not None and region is not None and location != region:
        raise ValueError(
            "Azure() received conflicting values for `location` and `region` "
            "— pass only one (they are aliases)."
        )
    effective_location = location or region
    assert effective_location is not None  # narrowed by the checks above

    config = AzureConfig(
        subscription_id=subscription_id,
        resource_group=resource_group,
        location=effective_location,
        vm_size=vm_size,
        spot=spot,
        azure_backend=azure_backend,
        artifact_account=artifact_account,
        artifact_container=artifact_container,
        aks_cluster=aks_cluster,
        **kwargs,
    )
    return AzureProvider(config=config, driver=driver, store=store)


__all__ = [
    "GPU_VM_FAMILIES",
    "KNOWN_REGIONS",
    "Azure",
    "AzureConfig",
    "CloudBackend",
]
