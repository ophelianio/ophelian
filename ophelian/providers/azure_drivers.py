"""Drivers that actually run pipeline steps on Azure infrastructure.

Mirror of :mod:`ophelian.providers.aws_drivers` for Microsoft Azure.
The shared cloud-agnostic helpers (``StepRequest``, ``StepOutcome``,
artifact persistence, upstream materialisation) are imported from
:mod:`aws_drivers`.

We ship two drivers:

* :class:`AzureVMDriver` (default): provisions an Azure VM per step
  using ``azure-mgmt-compute``, bootstraps Docker + ophelian via
  cloud-init, runs the step inside a container, polls Azure Blob for
  the worker's ``result.json`` and tears the VM down.

* :class:`LocalAzureDriver` (used by tests): runs the step in-process
  against the same artifact store the real driver would use.
"""

from __future__ import annotations

import base64
import json
import logging
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ophelian.core.nodes import Data, Deploy, Eval, StepResult, Train, Tune
from ophelian.providers.aws_drivers import (
    StepOutcome,
    StepRequest,
    _default_workspace,
    _materialise_upstream,
    _persist_artifacts,
    _strip_store_prefix,
    artifact_key,
    persist_run_checkpoint,
    random_run_id,
    result_json_key,
    train_checkpoint_key,
)
from ophelian.runtime.spot import SpotInterruption, SpotInterruptionMonitor

if TYPE_CHECKING:  # pragma: no cover
    from ophelian.envs.azure import AzureConfig
    from ophelian.stores.base import ArtifactStore

logger = logging.getLogger("ophelian.providers.azure_drivers")


class AzureCredentialError(RuntimeError):
    """Raised when Azure credentials are missing or insufficient."""


def _missing_azure_extra(component: str) -> ImportError:
    return ImportError(
        f"Azure {component} support requires the optional `azure` extra: "
        "`pip install 'ophelian[azure]'`."
    )


# ---------------------------------------------------------------------------
# Local driver — used by tests and as a CPU fallback
# ---------------------------------------------------------------------------


class LocalAzureDriver:
    """Run steps in-process; persist artifacts through *store*."""

    name = "local-azure"

    def __init__(
        self,
        *,
        spot_monitor: SpotInterruptionMonitor | None = None,
        fail_on: tuple[str, ...] = (),
    ) -> None:
        self._monitor = spot_monitor
        self._fail_on = set(fail_on)
        self._executed: list[str] = []

    @property
    def executed(self) -> list[str]:
        return list(self._executed)

    def execute(self, request: StepRequest, store: ArtifactStore) -> StepOutcome:
        from ophelian.providers.standalone import StandaloneProvider

        if request.step_name in self._fail_on:
            raise RuntimeError(f"LocalAzureDriver: forced failure on {request.step_name!r}")

        if request.workspace is None:
            workspace = Path(_default_workspace(request.run_id, request.step_name))
        else:
            workspace = Path(request.workspace)
        workspace.mkdir(parents=True, exist_ok=True)
        provider = StandaloneProvider(local=True, workspace=workspace, container=False)
        node = request.node

        local_artifacts = _materialise_upstream(request.upstream_artifacts, store, workspace)

        if isinstance(node, Data):
            result = provider._handle_data(node, workspace)
        elif isinstance(node, Train):
            result = provider._handle_train(node, workspace, local_artifacts)
        elif isinstance(node, Tune):
            result = provider._handle_tune(node, workspace, local_artifacts)
        elif isinstance(node, Eval):
            result = provider._handle_eval(node, workspace, local_artifacts)
        elif isinstance(node, Deploy):
            result = provider._handle_deploy(node, workspace, local_artifacts)
        else:
            raise TypeError(f"LocalAzureDriver does not know how to run {type(node).__name__}")

        persisted = _persist_artifacts(
            run_id=request.run_id,
            step_name=request.step_name,
            artifacts=result.artifacts,
            store=store,
        )
        result = result.model_copy(update={"artifacts": persisted})

        self._executed.append(request.step_name)

        if self._monitor is not None and self._monitor.check():
            raise SpotInterruption(
                f"Azure spot eviction detected while running step {request.step_name!r}",
                deadline=time.time() + 30.0,
            )

        return StepOutcome(
            result=result,
            instance_id=f"local-azure-{request.step_name}",
            instance_type="local",
            spot=False,
            logs="",
        )

    def teardown(self) -> None:
        return None


# ---------------------------------------------------------------------------
# Azure VM driver (default)
# ---------------------------------------------------------------------------


class AzureVMDriver:
    """Provision an Azure VM per step and run it inside a container.

    Lifecycle:

    1. Create NIC + VM (on-demand or Spot) with cloud-init that installs
       Docker, pulls ``runtime_image`` and runs ``step_runner`` against
       ``/work/step.json``.
    2. Wait for the VM to reach ``Succeeded`` provisioning state.
    3. Poll the artifact container for the worker's ``result.json``.
    4. Delete the VM + NIC + disk — guaranteed by ``try/finally``.

    Tests can inject a fake ``compute_client`` and ``network_client`` so
    SDK calls go to stubs rather than real Azure.
    """

    name = "azure-vm"

    def __init__(
        self,
        config: AzureConfig,
        *,
        compute_client: Any | None = None,
        network_client: Any | None = None,
        sleep: Any = time.sleep,
    ) -> None:
        self._config = config
        self._compute = (
            compute_client if compute_client is not None else _build_compute_client(config)
        )
        self._network = (
            network_client if network_client is not None else _build_network_client(config)
        )
        self._sleep = sleep
        self._tracked_vms: list[str] = []
        self._tracked_nics: list[str] = []
        self._keep_alive_vms: set[str] = set()
        self._keep_alive_nics: set[str] = set()

    @property
    def tracked_vms(self) -> list[str]:
        return list(self._tracked_vms)

    @property
    def keep_alive_vms(self) -> set[str]:
        return set(self._keep_alive_vms)

    # ---- Azure SDK wrappers -------------------------------------------

    def _vm_name(self, request: StepRequest) -> str:
        suffix = uuid.uuid4().hex[:6]
        return f"oph-{request.run_id}-{request.step_name}-{suffix}"[:60].lower()

    def _resolve_subnet_id(self) -> str:
        """Return the ARM resource id of the subnet to attach the NIC to.

        Requires both ``vnet`` and ``subnet`` on the config; in real
        deployments these are pre-created so the worker VM lives in
        the customer's network with their NSGs / private endpoints.
        """
        cfg = self._config
        if not cfg.vnet or not cfg.subnet:
            raise AzureCredentialError(
                "AzureVMDriver requires `vnet=...` and `subnet=...` on AzureConfig "
                "so the worker VM can attach a NIC. Pre-create the network or pass "
                "an existing one: see docs/envs/azure.md."
            )
        return (
            f"/subscriptions/{cfg.subscription_id}"
            f"/resourceGroups/{cfg.resource_group}"
            f"/providers/Microsoft.Network/virtualNetworks/{cfg.vnet}"
            f"/subnets/{cfg.subnet}"
        )

    def _create_nic(self, vm_name: str) -> str:
        cfg = self._config
        nic_name = f"{vm_name}-nic"[:80]
        params = {
            "location": cfg.location,
            "tags": {k: str(v) for k, v in cfg.instance_tags.items()},
            "ip_configurations": [
                {
                    "name": "ipconfig1",
                    "subnet": {"id": self._resolve_subnet_id()},
                    "private_ip_allocation_method": "Dynamic",
                }
            ],
        }
        try:
            poller = self._network.network_interfaces.begin_create_or_update(
                resource_group_name=cfg.resource_group,
                network_interface_name=nic_name,
                parameters=params,
            )
            nic = poller.result()
        except Exception as exc:
            _raise_azure_credential_error(exc, "network_interfaces.begin_create_or_update")
            raise
        nic_id = getattr(nic, "id", None)
        if not nic_id:
            raise RuntimeError(
                f"AzureVMDriver: NIC {nic_name!r} was created but the SDK did not "
                "return an id we can attach to the VM."
            )
        self._tracked_nics.append(nic_name)
        return str(nic_id)

    def _delete_nic(self, name: str) -> None:
        cfg = self._config
        try:
            self._network.network_interfaces.begin_delete(
                resource_group_name=cfg.resource_group,
                network_interface_name=name,
            )
        except Exception as exc:  # pragma: no cover - permissions-dependent
            logger.warning(
                "network_interfaces.begin_delete failed for %s: %s", name, exc
            )

    def _build_vm_body(
        self, request: StepRequest, name: str, nic_id: str
    ) -> dict[str, Any]:
        cfg = self._config
        custom_data = base64.b64encode(_cloud_init(cfg, request).encode("utf-8")).decode("ascii")
        body: dict[str, Any] = {
            "location": cfg.location,
            "tags": {
                **{k: str(v) for k, v in cfg.instance_tags.items()},
                "ophelian_run_id": request.run_id,
                "ophelian_step": request.step_name,
            },
            "hardware_profile": {"vm_size": cfg.vm_size},
            "storage_profile": {
                "image_reference": {
                    "publisher": cfg.image_publisher,
                    "offer": cfg.image_offer,
                    "sku": cfg.image_sku,
                    "version": cfg.image_version,
                },
            },
            "os_profile": {
                "computer_name": name[:15],
                "admin_username": "ophelian",
                "custom_data": custom_data,
                "linux_configuration": {"disable_password_authentication": True},
            },
            "network_profile": {
                "network_interfaces": [{"id": nic_id, "primary": True}],
            },
        }
        if cfg.spot:
            body["priority"] = "Spot"
            body["eviction_policy"] = "Delete"
            if cfg.spot_max_price is not None:
                body["billing_profile"] = {"max_price": float(cfg.spot_max_price)}
            else:
                body["billing_profile"] = {"max_price": -1.0}
        if cfg.managed_identity:
            body["identity"] = {
                "type": "UserAssigned",
                "user_assigned_identities": {cfg.managed_identity: {}},
            }
        return body

    def _create_vm(self, request: StepRequest) -> str:
        cfg = self._config
        name = self._vm_name(request)
        nic_id = self._create_nic(name)
        body = self._build_vm_body(request, name, nic_id)
        try:
            self._compute.virtual_machines.begin_create_or_update(
                resource_group_name=cfg.resource_group,
                vm_name=name,
                parameters=body,
            )
        except Exception as exc:
            # Roll back the NIC we just created so we don't leak resources.
            self._delete_nic(f"{name}-nic"[:80])
            _raise_azure_credential_error(exc, "virtual_machines.begin_create_or_update")
            raise
        self._tracked_vms.append(name)
        if request.kind == "deploy":
            self._keep_alive_vms.add(name)
            self._keep_alive_nics.add(f"{name}-nic"[:80])
        return name

    def _delete_vm(self, name: str) -> None:
        cfg = self._config
        try:
            self._compute.virtual_machines.begin_delete(
                resource_group_name=cfg.resource_group, vm_name=name
            )
        except Exception as exc:  # pragma: no cover - permissions-dependent
            logger.warning("virtual_machines.begin_delete failed for %s: %s", name, exc)

    # ---- main loop -----------------------------------------------------

    def execute(self, request: StepRequest, store: ArtifactStore) -> StepOutcome:
        cfg = self._config
        name = self._create_vm(request)
        deadline = time.time() + cfg.timeout_seconds
        result_key = result_json_key(request.run_id, request.step_name)
        try:
            payload: bytes | None = None
            while time.time() < deadline:
                self._sleep(cfg.poll_interval_seconds)
                if store.exists(result_key):
                    payload = store.get_bytes(result_key)
                    break
            if payload is None:
                raise TimeoutError(
                    f"AzureVMDriver: no result.json from {name} after "
                    f"{cfg.timeout_seconds}s — check the worker logs in Azure Monitor."
                )
            data = json.loads(payload.decode("utf-8"))
            artifacts = dict(data.get("artifacts") or {})
            persisted = _persist_artifacts(
                run_id=request.run_id,
                step_name=request.step_name,
                artifacts=artifacts,
                store=store,
            )
            result = StepResult(
                name=request.step_name,
                kind=request.kind,
                status=data.get("status", "success"),
                artifacts=persisted,
                info={**(data.get("info") or {}), "vm_name": name, "cloud": "azure"},
            )
            return StepOutcome(
                result=result,
                instance_id=name,
                instance_type=cfg.vm_size,
                spot=cfg.spot,
                logs="",
            )
        finally:
            if name not in self._keep_alive_vms:
                self._delete_vm(name)
                self._delete_nic(f"{name}-nic"[:80])

    def teardown(self) -> None:
        for name in list(self._tracked_vms):
            if name in self._keep_alive_vms:
                continue
            self._delete_vm(name)
        for nic in list(self._tracked_nics):
            if nic in self._keep_alive_nics:
                continue
            self._delete_nic(nic)
        self._tracked_vms = [n for n in self._tracked_vms if n in self._keep_alive_vms]
        self._tracked_nics = [n for n in self._tracked_nics if n in self._keep_alive_nics]

    def cleanup_deploys(self) -> None:
        for name in list(self._keep_alive_vms):
            self._delete_vm(name)
        for nic in list(self._keep_alive_nics):
            self._delete_nic(nic)
        self._keep_alive_vms.clear()
        self._keep_alive_nics.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cloud_init(config: AzureConfig, request: StepRequest) -> str:
    """Cloud-init that boots the worker on an Azure VM."""
    spec_b64 = _encode_step_spec(request)
    extras_set = set(config.runtime_extras or ())
    extras_set.add("azure")
    extras = ",".join(sorted(extras_set))
    install_target = f"ophelian[{extras}]"
    account = config.artifact_account or ""
    container = config.artifact_container
    extra_user_data = f"\n{config.user_data}\n" if config.user_data else ""
    return (
        "#!/bin/bash\n"
        "set -eu\n"
        "export DEBIAN_FRONTEND=noninteractive\n"
        "apt-get update -y\n"
        "apt-get install -y python3-pip ca-certificates curl gnupg\n"
        "pip install --upgrade pip >/dev/null\n"
        f"pip install '{install_target}'\n"
        "mkdir -p /work\n"
        f"echo {spec_b64} | base64 -d > /work/step.json\n"
        f"export OPHELIAN_RUN_ID={request.run_id}\n"
        f"export OPHELIAN_ARTIFACT_ACCOUNT={account}\n"
        f"export OPHELIAN_ARTIFACT_CONTAINER={container}\n"
        f"export OPHELIAN_ARTIFACT_PREFIX={config.artifact_prefix}\n"
        "export OPHELIAN_ARTIFACT_BACKEND=azure\n"
        f"{extra_user_data}"
        "python3 -m ophelian.runtime.step_runner /work/step.json\n"
    )


def _encode_step_spec(request: StepRequest) -> str:
    spec = {
        "run_id": request.run_id,
        "pipeline_name": request.pipeline_name,
        "step_name": request.step_name,
        "kind": request.kind,
        "node": request.node.model_dump(mode="json"),
        "upstream_artifacts": request.upstream_artifacts,
        "upstream_info": request.upstream_info,
        "resume": request.resume,
        "resume_from": request.resume_from,
    }
    return base64.b64encode(json.dumps(spec).encode("utf-8")).decode("ascii")


def _build_compute_client(config: AzureConfig) -> Any:
    try:
        from azure.identity import DefaultAzureCredential
        from azure.mgmt.compute import ComputeManagementClient
    except ImportError as exc:
        raise _missing_azure_extra("compute") from exc
    return ComputeManagementClient(DefaultAzureCredential(), config.subscription_id)


def _build_network_client(config: AzureConfig) -> Any:
    try:
        from azure.identity import DefaultAzureCredential
        from azure.mgmt.network import NetworkManagementClient
    except ImportError as exc:
        raise _missing_azure_extra("network") from exc
    return NetworkManagementClient(DefaultAzureCredential(), config.subscription_id)


def _raise_azure_credential_error(exc: Exception, action: str) -> None:
    msg = str(exc).lower()
    if "credential" in msg or "unauthorized" in msg or "forbidden" in msg:
        raise AzureCredentialError(
            f"Azure {action} failed with a credentials error: {exc}. "
            "Make sure DefaultAzureCredential can authenticate (`az login` or "
            "managed identity) and the principal has Contributor on the resource group."
        ) from exc


__all__ = [
    "AzureCredentialError",
    "AzureVMDriver",
    "LocalAzureDriver",
    "StepOutcome",
    "StepRequest",
    "_default_workspace",
    "_materialise_upstream",
    "_persist_artifacts",
    "_strip_store_prefix",
    "artifact_key",
    "persist_run_checkpoint",
    "random_run_id",
    "result_json_key",
    "train_checkpoint_key",
]
