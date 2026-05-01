"""Drivers that actually run pipeline steps on GCP infrastructure.

Mirror of :mod:`ophelian.providers.aws_drivers` for Google Cloud. The
shared cloud-agnostic helpers (``StepRequest``, ``StepOutcome``,
artifact persistence, upstream materialisation) live in
:mod:`aws_drivers` because they were introduced there first; we
re-export the relevant names so the GCP code path is self-contained
from a caller's perspective.

We ship two drivers:

* :class:`GCEDriver` (default): provisions a GCE VM per step using the
  ``google-cloud-compute`` SDK, bootstraps Docker + ophelian via a
  startup script, runs the step inside a container, polls GCS for the
  worker's ``result.json`` and tears the VM down. Cleanup runs in a
  ``try/finally`` and labels every resource so a later sweeper can
  collect orphans if the host process dies mid-flight.

* :class:`LocalGCPDriver` (used by tests): runs the step in-process
  against the same artifact store the real driver would use. With a
  fake / in-memory GCS it gives end-to-end coverage without provisioning
  a real VM.
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
from ophelian.providers.aws_drivers import (  # cloud-agnostic helpers
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
    from ophelian.envs.gcp import GCPConfig
    from ophelian.stores.base import ArtifactStore

logger = logging.getLogger("ophelian.providers.gcp_drivers")


class GCPCredentialError(RuntimeError):
    """Raised when GCP credentials are missing or insufficient."""


def _missing_gcp_extra(component: str) -> ImportError:
    return ImportError(
        f"GCP {component} support requires the optional `gcp` extra: "
        "`pip install 'ophelian[gcp]'`."
    )


# ---------------------------------------------------------------------------
# Local driver — used by tests and as a CPU fallback
# ---------------------------------------------------------------------------


class LocalGCPDriver:
    """Run steps in-process; persist artifacts through *store*.

    Identical surface to :class:`ophelian.providers.aws_drivers.LocalDriver`
    but reports its driver name as ``local-gcp`` so the observability
    summary makes the cloud the run was wired up for explicit.
    """

    name = "local-gcp"

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
            raise RuntimeError(f"LocalGCPDriver: forced failure on {request.step_name!r}")

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
            raise TypeError(f"LocalGCPDriver does not know how to run {type(node).__name__}")

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
                f"Preemption detected while running step {request.step_name!r}",
                deadline=time.time() + 30.0,
            )

        return StepOutcome(
            result=result,
            instance_id=f"local-gcp-{request.step_name}",
            instance_type="local",
            spot=False,
            logs="",
        )

    def teardown(self) -> None:
        return None


# ---------------------------------------------------------------------------
# GCE driver (default)
# ---------------------------------------------------------------------------


class GCEDriver:
    """Provision a GCE VM per step and run it inside a container.

    Lifecycle:

    1. ``instances.insert`` (on-demand or preemptible) with a startup
       script that installs Docker, pulls ``runtime_image`` and runs
       ``step_runner`` against ``/work/step.json``.
    2. Wait for the instance to reach ``RUNNING`` state.
    3. Poll GCS for the ``result.json`` the worker uploads.
    4. ``instances.delete`` — guaranteed by ``try/finally``.

    Tests can inject a fake ``compute_client`` so the SDK calls go to a
    stub rather than real GCP.
    """

    name = "gce"

    def __init__(
        self,
        config: GCPConfig,
        *,
        compute_client: Any | None = None,
        sleep: Any = time.sleep,
    ) -> None:
        self._config = config
        self._compute = (
            compute_client if compute_client is not None else _build_compute_client(config)
        )
        self._sleep = sleep
        self._tracked_instances: list[str] = []
        self._keep_alive_instances: set[str] = set()

    @property
    def tracked_instances(self) -> list[str]:
        return list(self._tracked_instances)

    @property
    def keep_alive_instances(self) -> set[str]:
        return set(self._keep_alive_instances)

    # ---- SDK wrappers --------------------------------------------------

    def _instance_name(self, request: StepRequest) -> str:
        suffix = uuid.uuid4().hex[:6]
        return f"ophelian-{request.run_id}-{request.step_name}-{suffix}"[:62].lower()

    def _build_instance_body(self, request: StepRequest, name: str) -> dict[str, Any]:
        cfg = self._config
        startup = _startup_script(cfg, request)
        body: dict[str, Any] = {
            "name": name,
            "machineType": (
                f"zones/{cfg.effective_zone}/machineTypes/{cfg.machine_type}"
            ),
            "disks": [
                {
                    "boot": True,
                    "autoDelete": True,
                    "initializeParams": {
                        "sourceImage": (
                            f"projects/{cfg.image_project}/global/images/family/{cfg.image_family}"
                        ),
                    },
                }
            ],
            "networkInterfaces": [
                {
                    "network": f"global/networks/{cfg.network}",
                    "accessConfigs": [
                        {"type": "ONE_TO_ONE_NAT", "name": "External NAT"}
                    ],
                }
            ],
            "metadata": {
                "items": [
                    {"key": "startup-script", "value": startup},
                    {"key": "ophelian-run-id", "value": request.run_id},
                    {"key": "ophelian-step", "value": request.step_name},
                ]
            },
            "labels": {
                **{k: str(v) for k, v in cfg.instance_tags.items()},
                "ophelian_run_id": request.run_id.replace("_", "-"),
                "ophelian_step": request.step_name.replace("_", "-"),
            },
            "scheduling": {"preemptible": cfg.spot},
        }
        if cfg.subnet:
            body["networkInterfaces"][0]["subnetwork"] = (
                f"projects/{cfg.project}/regions/{cfg.region}/subnetworks/{cfg.subnet}"
            )
        if cfg.is_gpu_instance:
            body["guestAccelerators"] = [
                {
                    "acceleratorType": (
                        f"projects/{cfg.project}/zones/{cfg.effective_zone}"
                        f"/acceleratorTypes/{cfg.gpu_type}"
                    ),
                    "acceleratorCount": cfg.gpu_count,
                }
            ]
            body["scheduling"]["onHostMaintenance"] = "TERMINATE"
        if cfg.service_account:
            body["serviceAccounts"] = [
                {
                    "email": cfg.service_account,
                    "scopes": ["https://www.googleapis.com/auth/cloud-platform"],
                }
            ]
        return body

    def _insert_instance(self, request: StepRequest) -> str:
        cfg = self._config
        name = self._instance_name(request)
        body = self._build_instance_body(request, name)
        try:
            # google-cloud-compute's typed client takes a flattened
            # ``instance_resource=`` parameter — passing ``body=`` is the
            # discovery-client (deprecated) shape and silently no-ops on
            # the typed client.
            self._compute.insert(
                project=cfg.project,
                zone=cfg.effective_zone,
                instance_resource=body,
            )
        except Exception as exc:
            _raise_gcp_credential_error(exc, "instances.insert")
            raise
        self._tracked_instances.append(name)
        if request.kind == "deploy":
            self._keep_alive_instances.add(name)
        return name

    def _delete_instance(self, name: str) -> None:
        cfg = self._config
        try:
            self._compute.delete(project=cfg.project, zone=cfg.effective_zone, instance=name)
        except Exception as exc:  # pragma: no cover - permissions-dependent
            logger.warning("instances.delete failed for %s: %s", name, exc)

    # ---- main loop -----------------------------------------------------

    def execute(self, request: StepRequest, store: ArtifactStore) -> StepOutcome:
        cfg = self._config
        name = self._insert_instance(request)
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
                    f"GCEDriver: no result.json from {name} after "
                    f"{cfg.timeout_seconds}s — check the worker logs in Cloud Logging."
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
                info={**(data.get("info") or {}), "instance_id": name, "cloud": "gcp"},
            )
            return StepOutcome(
                result=result,
                instance_id=name,
                instance_type=cfg.machine_type,
                spot=cfg.spot,
                logs="",
            )
        finally:
            if name not in self._keep_alive_instances:
                self._delete_instance(name)

    def teardown(self) -> None:
        for name in list(self._tracked_instances):
            if name in self._keep_alive_instances:
                continue
            self._delete_instance(name)
        self._tracked_instances = [n for n in self._tracked_instances if n in self._keep_alive_instances]

    def cleanup_deploys(self) -> None:
        for name in list(self._keep_alive_instances):
            self._delete_instance(name)
        self._keep_alive_instances.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _startup_script(config: GCPConfig, request: StepRequest) -> str:
    """Cloud-init / startup-script that boots the worker on a GCE VM."""
    spec_b64 = _encode_step_spec(request)
    extras_set = set(config.runtime_extras or ())
    extras_set.add("gcp")
    extras = ",".join(sorted(extras_set))
    install_target = f"ophelian[{extras}]"
    bucket = config.artifact_bucket or ""
    archive_install = ""
    if config.pipeline_archive_uri:
        archive_install = (
            f"\ngsutil cp {config.pipeline_archive_uri} /tmp/pipeline.tar.gz\n"
            "mkdir -p /opt/ophelian-pipeline && "
            "tar xzf /tmp/pipeline.tar.gz -C /opt/ophelian-pipeline\n"
            "pip install /opt/ophelian-pipeline\n"
        )
    extra_user_data = f"\n{config.user_data}\n" if config.user_data else ""
    return (
        "#!/bin/bash\n"
        "set -eu\n"
        "export DEBIAN_FRONTEND=noninteractive\n"
        "apt-get update -y\n"
        "apt-get install -y python3-pip ca-certificates curl gnupg\n"
        "pip install --upgrade pip >/dev/null\n"
        f"pip install '{install_target}'\n"
        f"{archive_install}"
        "mkdir -p /work\n"
        f"echo {spec_b64} | base64 -d > /work/step.json\n"
        f"export OPHELIAN_RUN_ID={request.run_id}\n"
        f"export OPHELIAN_ARTIFACT_BUCKET={bucket}\n"
        f"export OPHELIAN_ARTIFACT_PREFIX={config.artifact_prefix}\n"
        "export OPHELIAN_ARTIFACT_BACKEND=gcs\n"
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


def _build_compute_client(config: GCPConfig) -> Any:
    try:
        from google.cloud import compute_v1
    except ImportError as exc:
        raise _missing_gcp_extra("compute") from exc
    return compute_v1.InstancesClient()


def _raise_gcp_credential_error(exc: Exception, action: str) -> None:
    msg = str(exc).lower()
    if "permission" in msg or "denied" in msg or "credential" in msg:
        raise GCPCredentialError(
            f"GCP {action} failed with a credentials/permissions error: {exc}. "
            "Make sure ADC are configured (`gcloud auth application-default login`)"
            " and the principal has Compute Admin + Storage Admin on the project."
        ) from exc


__all__ = [
    "GCEDriver",
    "GCPCredentialError",
    "LocalGCPDriver",
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
