"""Drivers that actually run pipeline steps on AWS infrastructure.

The :class:`~ophelian.providers.aws.AWSProvider` is intentionally thin —
it owns the run-id, the artifact store and the resume / checkpoint
plumbing, and delegates "make this step happen on AWS" to a
:class:`CloudDriver`. We ship three of them:

* :class:`EC2Driver` (default): provisions a real on-demand or spot EC2
  worker via boto3, bootstraps Docker + ophelian via cloud-init, runs the
  step inside a container, and tears the worker down. Cleanup runs in a
  ``try/finally`` and tags every resource so a later sweeper can pick up
  orphans if the host process is killed mid-flight.

* :class:`EKSDriver` (opt-in): submits a Kubernetes Job for batch steps
  (Data/Train/Tune/Eval) and a Deployment + Service for ``Deploy``. Uses
  the official ``kubernetes`` Python client.

* :class:`LocalDriver` (used by tests): runs the step in-process against
  the same artifact store the real drivers would use. With moto-mocked
  S3 it gives us realistic end-to-end coverage without provisioning a
  real VM.

All three speak the same small interface so the AWS provider does not
care which one is plugged in.
"""

from __future__ import annotations

import base64
import json
import logging
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from ophelian.core.nodes import (
    Data,
    Deploy,
    Eval,
    StepResult,
    Train,
    Tune,
)
from ophelian.runtime.spot import (
    Checkpoint,
    SpotInterruption,
    SpotInterruptionMonitor,
    save_checkpoint,
)

if TYPE_CHECKING:  # pragma: no cover
    from ophelian.envs.aws import AWSConfig
    from ophelian.stores.base import ArtifactStore

logger = logging.getLogger("ophelian.providers.aws_drivers")


# ---------------------------------------------------------------------------
# Driver protocol & data classes
# ---------------------------------------------------------------------------


@dataclass
class StepRequest:
    """Everything a driver needs to run a single step."""

    run_id: str
    pipeline_name: str
    step_name: str
    kind: str
    node: Data | Train | Tune | Eval | Deploy
    upstream_artifacts: dict[str, dict[str, str]] = field(default_factory=dict)
    upstream_info: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Per-upstream-step ``info`` dicts (e.g. ``{"trainer": {"framework":
    "sklearn", ...}}``). Threaded through by ``AWSProvider`` so deploy
    drivers can read the trainer's framework without a second S3 round
    trip — ``upstream_artifacts`` only carries opaque URIs."""
    workspace: Path | None = None
    resume: bool = False
    resume_from: str | None = None
    """``s3://...`` (or local) URI of the previous attempt's checkpoint
    snapshot. Set by ``AWSProvider`` when retrying an interrupted Train
    step so the worker's ``step_runner`` can pre-load weights and the
    PyTorch adapter can pick up at the last completed epoch."""


@dataclass
class StepOutcome:
    """Result of executing a step on AWS."""

    result: StepResult
    instance_id: str | None = None
    instance_type: str | None = None
    spot: bool = False
    logs: str = ""


class CredentialError(RuntimeError):
    """Raised when AWS credentials are missing or insufficient.

    Surfaced with a hint so users immediately know what permissions to
    grant; see ``docs/aws.md`` for the minimal IAM policy.
    """


def _missing_aws_extra(component: str) -> ImportError:
    return ImportError(
        f"AWS {component} support requires the optional `aws` extra: "
        "`pip install 'ophelian[aws]'`."
    )


@runtime_checkable
class CloudDriver(Protocol):
    """Run a single pipeline step on cloud infrastructure."""

    name: str

    def execute(self, request: StepRequest, store: ArtifactStore) -> StepOutcome:
        """Execute *request*; return its :class:`StepOutcome`."""
        ...

    def teardown(self) -> None:
        """Best-effort cleanup of any leftover resources."""
        ...


# ---------------------------------------------------------------------------
# Local driver — used by tests, demos without AWS, and the LocalDriver fallback
# ---------------------------------------------------------------------------


class LocalDriver:
    """Run steps in-process; persist artifacts through *store*.

    This is what the unit tests use — they wire it together with a moto-
    mocked S3 store so the AWSProvider exercises every code path
    (checkpoint, resume, artifact persistence) without provisioning a
    real VM. It is also handy for users who want the AWS env semantics
    locally (e.g. to validate a pipeline before sending it to a real
    region).

    Parameters
    ----------
    spot_monitor:
        Optional :class:`SpotInterruptionMonitor`. When the monitor's
        ``check()`` returns True before a step finishes, the driver
        raises :class:`SpotInterruption` so the provider exercises the
        resume path. Tests typically inject ``force_trigger=True``.
    fail_on:
        Optional set of step names that should raise an exception. Used
        by tests to verify cleanup-on-failure behaviour.
    """

    name = "local"

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
            raise RuntimeError(f"LocalDriver: forced failure on {request.step_name!r}")

        # Use the in-process Standalone handlers to actually do the work.
        if request.workspace is None:
            workspace = Path(_default_workspace(request.run_id, request.step_name))
        else:
            workspace = Path(request.workspace)
        workspace.mkdir(parents=True, exist_ok=True)
        provider = StandaloneProvider(local=True, workspace=workspace, container=False)
        node = request.node

        # Materialise upstream artifacts from the store onto local disk so
        # the in-process handlers can read them.
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
            raise TypeError(f"LocalDriver does not know how to run {type(node).__name__}")

        # Persist result artifacts back into the store under a deterministic prefix.
        persisted = _persist_artifacts(
            run_id=request.run_id,
            step_name=request.step_name,
            artifacts=result.artifacts,
            store=store,
        )
        result = result.model_copy(update={"artifacts": persisted})

        self._executed.append(request.step_name)

        # Honour spot monitor *after* persisting partial work — that's
        # exactly the moment where checkpointing has the latest state.
        if self._monitor is not None and self._monitor.check():
            raise SpotInterruption(
                f"Spot interruption detected while running step {request.step_name!r}",
                deadline=time.time() + 120.0,
            )

        return StepOutcome(
            result=result,
            instance_id=f"local-{request.step_name}",
            instance_type="local",
            spot=False,
            logs="",
        )

    def teardown(self) -> None:
        return None


# ---------------------------------------------------------------------------
# EC2 driver (default)
# ---------------------------------------------------------------------------


class EC2Driver:
    """Provision an EC2 worker per step and run the step inside a container.

    The driver implements the lifecycle:

    1. ``run_instances`` (on-demand or spot) with cloud-init user-data that
       installs Docker, pulls the runtime image and runs the step.
    2. Wait for the instance to reach ``running`` state (and optionally for
       ``status=ok`` once spot interruption monitoring is enabled).
    3. Poll the artifact store for the ``result.json`` the worker uploads.
    4. ``terminate_instances`` — guaranteed by ``try/finally``.

    Tests inject a fake ``ec2_client`` so the boto3 calls go to moto rather
    than real AWS.
    """

    name = "ec2"

    def __init__(
        self,
        config: AWSConfig,
        *,
        ec2_client: Any | None = None,
        ssm_client: Any | None = None,
        spot_monitor: SpotInterruptionMonitor | None = None,
        sleep: Any = time.sleep,
    ) -> None:
        self._config = config
        self._ec2 = ec2_client if ec2_client is not None else _build_boto3_client("ec2", config)
        self._ssm = ssm_client
        self._monitor = spot_monitor
        self._sleep = sleep
        self._tracked_instances: list[str] = []
        # Instances that should *not* be terminated by `teardown()` —
        # populated for `Deploy` steps so the serving uvicorn survives the
        # pipeline run.
        self._keep_alive_instances: set[str] = set()
        # `validate_instance_type_in_region` is expensive (a network round
        # trip) — run it once per driver instance, on first `execute()`.
        self._region_check_done = False

    @property
    def tracked_instances(self) -> list[str]:
        return list(self._tracked_instances)

    @property
    def keep_alive_instances(self) -> set[str]:
        return set(self._keep_alive_instances)

    # ---- boto3 wrappers ------------------------------------------------

    def _run_instances(self, request: StepRequest) -> str:
        cfg = self._config
        kwargs: dict[str, Any] = {
            "ImageId": cfg.ami or _default_ami_for(cfg, ssm_client=self._ssm),
            "InstanceType": cfg.instance,
            "MinCount": 1,
            "MaxCount": 1,
            "TagSpecifications": [
                {
                    "ResourceType": "instance",
                    "Tags": [
                        {"Key": "Name", "Value": f"ophelian-{request.run_id}-{request.step_name}"},
                        {"Key": "ophelian/run-id", "Value": request.run_id},
                        {"Key": "ophelian/pipeline", "Value": request.pipeline_name},
                        {"Key": "ophelian/step", "Value": request.step_name},
                        *(
                            {"Key": k, "Value": v}
                            for k, v in cfg.instance_tags.items()
                        ),
                    ],
                }
            ],
            "UserData": _user_data_script(cfg, request),
        }
        if cfg.vpc:
            kwargs["SubnetId"] = cfg.vpc
        sg_id = cfg.security_group or self._ensure_deploy_security_group(request)
        if sg_id:
            kwargs["SecurityGroupIds"] = [sg_id]
        if cfg.key_pair:
            kwargs["KeyName"] = cfg.key_pair
        if cfg.iam_role:
            kwargs["IamInstanceProfile"] = {"Name": cfg.iam_role}
        if cfg.spot:
            spot_options: dict[str, Any] = {"SpotInstanceType": "one-time"}
            if cfg.spot_max_price is not None:
                spot_options["MaxPrice"] = f"{cfg.spot_max_price:.4f}"
            kwargs["InstanceMarketOptions"] = {
                "MarketType": "spot",
                "SpotOptions": spot_options,
            }
        try:
            response = self._ec2.run_instances(**kwargs)
        except Exception as exc:  # pragma: no cover - boto3 error class
            _raise_credential_error(exc, "EC2 run_instances")
            raise
        instance_id = str(response["Instances"][0]["InstanceId"])
        self._tracked_instances.append(instance_id)
        return instance_id

    def _wait_for_running(self, instance_id: str) -> None:
        deadline = time.monotonic() + self._config.timeout_seconds
        while time.monotonic() < deadline:
            response = self._ec2.describe_instances(InstanceIds=[instance_id])
            reservations = response.get("Reservations", [])
            if reservations and reservations[0]["Instances"]:
                state = reservations[0]["Instances"][0]["State"]["Name"]
                if state == "running":
                    return
                if state in {"terminated", "stopping", "stopped", "shutting-down"}:
                    raise RuntimeError(
                        f"EC2 instance {instance_id} reached unexpected state {state!r}"
                    )
            self._sleep(self._config.poll_interval_seconds)
        raise TimeoutError(
            f"Timed out waiting for instance {instance_id} to reach 'running' state"
        )

    def _wait_for_result(
        self,
        store: ArtifactStore,
        request: StepRequest,
        instance_id: str | None = None,
    ) -> dict[str, Any]:
        result_key = result_json_key(request.run_id, request.step_name)
        deadline = time.monotonic() + self._config.timeout_seconds
        while time.monotonic() < deadline:
            if self._monitor is not None and self._monitor.check():
                raise SpotInterruption(
                    f"Spot interruption signalled while waiting for {request.step_name!r}",
                )
            # On real spot runs, an injected monitor isn't reliable from
            # the driver host (IMDS lives on the *worker*). Poll EC2
            # describe_instances instead — a running spot worker that
            # transitions to shutting-down/terminated before result.json
            # appears is, by definition, a spot reclamation.
            if (
                instance_id is not None
                and self._config.spot
                and self._is_spot_interrupted(instance_id)
            ):
                raise SpotInterruption(
                    f"EC2 instance {instance_id} reclaimed by AWS spot while running "
                    f"{request.step_name!r}"
                )
            if store.exists(result_key):
                if hasattr(store, "get_bytes"):
                    payload = store.get_bytes(result_key)
                else:
                    payload = Path(store.get(result_key)).read_bytes()
                parsed: dict[str, Any] = json.loads(payload)
                return parsed
            self._sleep(self._config.poll_interval_seconds)
        raise TimeoutError(
            f"Timed out waiting for step {request.step_name!r} result on s3://{result_key}"
        )

    def _is_spot_interrupted(self, instance_id: str) -> bool:
        try:
            resp = self._ec2.describe_instances(InstanceIds=[instance_id])
        except Exception:  # pragma: no cover - defensive
            return False
        for reservation in resp.get("Reservations", []) or []:
            for inst in reservation.get("Instances", []) or []:
                lifecycle = inst.get("InstanceLifecycle")
                state = (inst.get("State") or {}).get("Name")
                if lifecycle == "spot" and state in {
                    "shutting-down",
                    "stopping",
                    "stopped",
                    "terminated",
                }:
                    return True
        return False

    def _ensure_deploy_security_group(self, request: StepRequest) -> str | None:
        """Provision a security group exposing the deploy port (0.0.0.0/0).

        Returns ``None`` for non-deploy steps so that the run keeps using
        the default VPC's default security group (which is fine for
        outbound-only train/eval/data workers).

        For deploys we create — or reuse — ``ophelian-deploy-{port}`` and
        authorise inbound TCP on the deploy port. If SG provisioning
        fails we raise ``CredentialError`` with a clear message rather
        than silently launching an unreachable endpoint.
        """
        if request.kind != "deploy":
            return None
        port = getattr(request.node, "port", None)
        if not isinstance(port, int):
            return None
        name = f"ophelian-deploy-{port}"
        # Try to find an existing SG by name first.
        try:
            existing = self._ec2.describe_security_groups(
                Filters=[{"Name": "group-name", "Values": [name]}]
            )
        except Exception as exc:  # pragma: no cover - defensive
            raise CredentialError(
                f"Could not query EC2 security groups for {name!r}: {exc}. "
                "Either grant ec2:DescribeSecurityGroups or pass an explicit "
                "security_group=... to AWS(...)."
            ) from exc
        groups = existing.get("SecurityGroups") or []
        if groups:
            sg_id = groups[0].get("GroupId")
            if isinstance(sg_id, str):
                return sg_id
        # Otherwise create a new SG and open the deploy port to the world.
        try:
            created = self._ec2.create_security_group(
                GroupName=name,
                Description=f"Ophelian deploy ingress for port {port}",
            )
            sg_id = created.get("GroupId")
            assert isinstance(sg_id, str)
            self._ec2.authorize_security_group_ingress(
                GroupId=sg_id,
                IpPermissions=[
                    {
                        "IpProtocol": "tcp",
                        "FromPort": port,
                        "ToPort": port,
                        "IpRanges": [
                            {"CidrIp": "0.0.0.0/0", "Description": "ophelian deploy"},
                        ],
                    }
                ],
            )
        except Exception as exc:
            raise CredentialError(
                f"Could not create security group {name!r} (port {port} ingress). "
                f"Underlying error: {exc}. Either grant ec2:CreateSecurityGroup + "
                "ec2:AuthorizeSecurityGroupIngress, or pass security_group=<id> "
                "to AWS(...) with the deploy port already open."
            ) from exc
        return sg_id

    def _public_dns(self, instance_id: str) -> str | None:
        try:
            resp = self._ec2.describe_instances(InstanceIds=[instance_id])
        except Exception:  # pragma: no cover - defensive
            return None
        for reservation in resp.get("Reservations", []) or []:
            for inst in reservation.get("Instances", []) or []:
                dns = inst.get("PublicDnsName") or inst.get("PublicIpAddress")
                if dns:
                    return str(dns)
        return None

    def _terminate(self, instance_id: str) -> None:
        try:
            self._ec2.terminate_instances(InstanceIds=[instance_id])
        except Exception as exc:  # pragma: no cover
            logger.warning("terminate_instances(%s) failed: %s", instance_id, exc)

    # ---- public surface ------------------------------------------------

    def execute(self, request: StepRequest, store: ArtifactStore) -> StepOutcome:
        instance_id: str | None = None
        is_deploy = request.kind == "deploy"
        # Run the region-availability check at most once per driver, lazily
        # so unit tests with a MagicMock client don't have to mock it.
        if not self._region_check_done:
            self._region_check_done = True
            self._config.validate_instance_type_in_region(self._ec2)
        try:
            instance_id = self._run_instances(request)
            logger.info(
                "EC2 worker %s launched for step %s (spot=%s)",
                instance_id,
                request.step_name,
                self._config.spot,
            )
            self._wait_for_running(instance_id)
            payload = self._wait_for_result(store, request, instance_id=instance_id)
            info: dict[str, Any] = {
                **(payload.get("info") or {}),
                "instance_id": instance_id,
                "instance_type": self._config.instance,
                "spot": self._config.spot,
                "region": self._config.region,
            }
            artifacts_out: dict[str, str] = dict(payload.get("artifacts", {}) or {})
            # Deploy steps stay alive on the worker — rewrite the
            # endpoint URL with the EC2 public DNS so the caller sees a
            # reachable URL instead of `http://localhost:port`.
            if is_deploy:
                port = getattr(request.node, "port", None)
                dns = self._public_dns(instance_id)
                if dns and port:
                    public_url = f"http://{dns}:{port}"
                    info["predict"] = f"{public_url}/predict"
                    info["health"] = f"{public_url}/health"
                    info["endpoint_url"] = public_url
                    info["public_dns"] = dns
                    artifacts_out["endpoint_url"] = public_url
            result = StepResult(
                name=payload.get("name", request.step_name),
                kind=payload.get("kind", request.kind),
                status=payload.get("status", "success"),
                metrics=payload.get("metrics", {}) or {},
                artifacts=artifacts_out,
                info=info,
                error=payload.get("error"),
            )
            return StepOutcome(
                result=result,
                instance_id=instance_id,
                instance_type=self._config.instance,
                spot=self._config.spot,
            )
        finally:
            # Deploy workers are intentionally long-lived — the user has
            # to tear them down explicitly via `provider.cleanup_deploys()`
            # or through the AWS console. Track them in
            # `_keep_alive_instances` so the regular `teardown()` (called
            # from `AWSProvider.execute`'s finally) skips them.
            if instance_id and is_deploy:
                self._keep_alive_instances.add(instance_id)
            if instance_id and not is_deploy:
                self._terminate(instance_id)

    def teardown(self) -> None:
        import contextlib

        for instance_id in list(self._tracked_instances):
            if instance_id in self._keep_alive_instances:
                # Long-lived deploy worker — preserve it across run.
                continue
            with contextlib.suppress(Exception):  # pragma: no cover
                self._ec2.terminate_instances(InstanceIds=[instance_id])
        # Keep the keep-alive set in `_tracked_instances` so a later
        # `cleanup_deploys()` knows what to terminate; drop the rest.
        self._tracked_instances = [
            i for i in self._tracked_instances if i in self._keep_alive_instances
        ]

    def cleanup_deploys(self) -> None:
        """Explicitly terminate every long-lived deploy worker.

        Use this when you're done serving — typically the user calls
        ``provider.cleanup_deploys()`` from a finally block in their own
        script.
        """
        import contextlib

        for instance_id in list(self._keep_alive_instances):
            with contextlib.suppress(Exception):  # pragma: no cover
                self._ec2.terminate_instances(InstanceIds=[instance_id])
        self._keep_alive_instances.clear()
        self._tracked_instances = [
            i for i in self._tracked_instances if i not in self._keep_alive_instances
        ]


# ---------------------------------------------------------------------------
# EKS driver
# ---------------------------------------------------------------------------


class EKSDriver:
    """Submit pipeline steps as Jobs/Deployments to an existing EKS cluster.

    Requires the ``kubernetes`` Python client and a kubeconfig that points
    at the user's cluster. Tests pass a fake ``api_client`` to capture the
    manifests the driver would create.
    """

    name = "eks"

    def __init__(
        self,
        config: AWSConfig,
        *,
        batch_api: Any | None = None,
        core_api: Any | None = None,
        apps_api: Any | None = None,
        sleep: Any = time.sleep,
    ) -> None:
        if not config.eks_cluster:
            raise ValueError("EKSDriver requires AWSConfig.eks_cluster to be set")
        self._config = config
        self._batch_api = batch_api or _build_kubernetes_client("batch")
        self._core_api = core_api or _build_kubernetes_client("core")
        self._apps_api = apps_api or _build_kubernetes_client("apps")
        self._sleep = sleep
        self._submitted: list[tuple[str, str]] = []  # (kind, name)

    @property
    def submitted(self) -> list[tuple[str, str]]:
        return list(self._submitted)

    def _job_manifest(self, request: StepRequest) -> dict[str, Any]:
        cfg = self._config
        spec_b64 = _encode_step_spec(request)
        # The default ``runtime_image`` is a vanilla ``python:3.12-slim``
        # so users without a pre-baked image can still run things — the
        # tradeoff is a one-time pip install per pod. Production users
        # who care about cold-start latency should pre-bake an image
        # with ``ophelian[<extras>]`` already installed and pass it via
        # ``runtime_image=``; the bootstrap step below is a no-op in
        # that case (pip detects the requirement is already satisfied).
        return {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {
                "name": f"ophelian-{request.run_id[:8]}-{request.step_name}",
                "namespace": cfg.eks_namespace,
                "labels": {
                    "ophelian/run-id": request.run_id,
                    "ophelian/pipeline": request.pipeline_name,
                    "ophelian/step": request.step_name,
                },
            },
            "spec": {
                "backoffLimit": 1,
                "template": {
                    "spec": {
                        "restartPolicy": "Never",
                        "containers": [
                            {
                                "name": "step",
                                "image": cfg.runtime_image,
                                "env": [
                                    {"name": "OPHELIAN_RUN_ID", "value": request.run_id},
                                    {"name": "OPHELIAN_STEP_NAME", "value": request.step_name},
                                    {"name": "OPHELIAN_STEP_SPEC_B64", "value": spec_b64},
                                    {
                                        "name": "OPHELIAN_ARTIFACT_BUCKET",
                                        "value": cfg.artifact_bucket or "",
                                    },
                                    {
                                        "name": "OPHELIAN_ARTIFACT_PREFIX",
                                        "value": cfg.artifact_prefix or "",
                                    },
                                    {
                                        "name": "OPHELIAN_RESUME_FROM",
                                        "value": request.resume_from or "",
                                    },
                                ],
                                "command": ["sh", "-c"],
                                "args": [_eks_bootstrap_step(cfg, spec_b64)],
                            }
                        ],
                    }
                },
            },
        }

    def _deployment_manifest(self, request: StepRequest) -> dict[str, Any]:
        cfg = self._config
        node = request.node
        if not isinstance(node, Deploy):
            raise TypeError("Deployment manifest requires a Deploy node")
        return {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": f"ophelian-{request.step_name}",
                "namespace": cfg.eks_namespace,
            },
            "spec": {
                "replicas": node.replicas,
                "selector": {"matchLabels": {"app": f"ophelian-{request.step_name}"}},
                "template": {
                    "metadata": {"labels": {"app": f"ophelian-{request.step_name}"}},
                    "spec": {
                        "containers": [
                            {
                                "name": "serve",
                                "image": cfg.runtime_image,
                                "ports": [{"containerPort": node.port}],
                                "env": [
                                    {"name": "OPHELIAN_FRAMEWORK", "value": _serve_framework(request)},
                                    {"name": "OPHELIAN_MODEL_URI", "value": _serve_model_uri(request)},
                                    {"name": "AWS_DEFAULT_REGION", "value": cfg.region},
                                ],
                                "command": ["sh", "-c"],
                                "args": [_eks_bootstrap_deploy(cfg, node.port)],
                            }
                        ]
                    },
                },
            },
        }

    def _service_manifest(self, request: StepRequest) -> dict[str, Any]:
        node = request.node
        assert isinstance(node, Deploy)
        return {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": f"ophelian-{request.step_name}",
                "namespace": self._config.eks_namespace,
            },
            "spec": {
                "type": "LoadBalancer",
                "selector": {"app": f"ophelian-{request.step_name}"},
                "ports": [{"port": node.port, "targetPort": node.port}],
            },
        }

    def execute(self, request: StepRequest, store: ArtifactStore) -> StepOutcome:
        if isinstance(request.node, Deploy):
            deployment = self._deployment_manifest(request)
            service = self._service_manifest(request)
            self._apps_api.create_namespaced_deployment(
                namespace=self._config.eks_namespace, body=deployment
            )
            self._submitted.append(("Deployment", deployment["metadata"]["name"]))
            self._core_api.create_namespaced_service(
                namespace=self._config.eks_namespace, body=service
            )
            self._submitted.append(("Service", service["metadata"]["name"]))
            url = f"http://ophelian-{request.step_name}.{self._config.eks_namespace}.svc:{request.node.port}"
            result = StepResult(
                name=request.step_name,
                kind=request.kind,
                status="success",
                artifacts={"endpoint_url": url},
                info={
                    "deployment": deployment["metadata"]["name"],
                    "service": service["metadata"]["name"],
                    "replicas": request.node.replicas,
                    "backend": "eks",
                },
            )
            return StepOutcome(result=result, instance_type="eks", instance_id=None)

        # Batch path — Data/Train/Tune/Eval.
        manifest = self._job_manifest(request)
        self._batch_api.create_namespaced_job(
            namespace=self._config.eks_namespace, body=manifest
        )
        self._submitted.append(("Job", manifest["metadata"]["name"]))

        # Wait for the corresponding result file in the store. Same contract
        # the EC2 driver uses — the in-pod step_runner must upload it.
        result_key = result_json_key(request.run_id, request.step_name)
        deadline = time.monotonic() + self._config.timeout_seconds
        while time.monotonic() < deadline:
            if store.exists(result_key):
                if hasattr(store, "get_bytes"):
                    payload = store.get_bytes(result_key)
                else:
                    payload = Path(store.get(result_key)).read_bytes()
                data = json.loads(payload)
                result = StepResult(
                    name=data.get("name", request.step_name),
                    kind=data.get("kind", request.kind),
                    status=data.get("status", "success"),
                    metrics=data.get("metrics", {}) or {},
                    artifacts=data.get("artifacts", {}) or {},
                    info={
                        **(data.get("info") or {}),
                        "backend": "eks",
                        "job": manifest["metadata"]["name"],
                    },
                    error=data.get("error"),
                )
                return StepOutcome(result=result, instance_type="eks", instance_id=None)
            self._sleep(self._config.poll_interval_seconds)
        raise TimeoutError(f"EKS Job for {request.step_name!r} never reported a result")

    def teardown(self) -> None:
        for kind, name in self._submitted:
            try:
                if kind == "Job":
                    self._batch_api.delete_namespaced_job(
                        name=name, namespace=self._config.eks_namespace
                    )
                elif kind == "Deployment":
                    self._apps_api.delete_namespaced_deployment(
                        name=name, namespace=self._config.eks_namespace
                    )
                elif kind == "Service":
                    self._core_api.delete_namespaced_service(
                        name=name, namespace=self._config.eks_namespace
                    )
            except Exception:  # pragma: no cover
                pass
        self._submitted.clear()


# ---------------------------------------------------------------------------
# Helpers shared by the drivers
# ---------------------------------------------------------------------------


def result_json_key(run_id: str, step_name: str) -> str:
    """S3 key (relative to the store prefix) where the worker writes its result."""
    return f"runs/{run_id}/{step_name}/result.json"


def artifact_key(run_id: str, step_name: str, name: str) -> str:
    """Prefix for an artifact uploaded by *step_name* under *name*."""
    return f"runs/{run_id}/{step_name}/artifacts/{name}"


def train_checkpoint_key(run_id: str, step_name: str) -> str:
    """Prefix where the worker uploads ``/work/{step}/checkpoint/`` on
    SIGTERM (spot reclamation) so the next attempt can resume from it.
    """
    return f"runs/{run_id}/checkpoints/{step_name}"


def _persist_artifacts(
    *,
    run_id: str,
    step_name: str,
    artifacts: Mapping[str, str],
    store: ArtifactStore,
) -> dict[str, str]:
    """Upload local-disk artifact paths into *store*; return URI map.

    The standalone in-process handlers populate ``artifacts`` with local
    paths. The cloud drivers upload them to S3 and the AWS provider keeps
    the URIs (so subsequent steps fetch them from S3, not from disk).
    """
    uploaded: dict[str, str] = {}
    for key, value in artifacts.items():
        if isinstance(value, str) and value.startswith(("s3://", "https://", "http://")):
            uploaded[key] = value
            continue
        path = Path(value)
        if not path.exists():
            uploaded[key] = value
            continue
        store_key = artifact_key(run_id, step_name, key)
        if path.is_dir():
            uploaded[key] = store.put(store_key, path)
        else:
            uploaded[key] = store.put(f"{store_key}/{path.name}", path)
    return uploaded


def _materialise_upstream(
    upstream: Mapping[str, Mapping[str, str]],
    store: ArtifactStore,
    workspace: Path,
) -> dict[str, dict[str, str]]:
    """Download S3-backed upstream artifacts into *workspace* and return local paths."""
    out: dict[str, dict[str, str]] = {}
    for step_name, items in upstream.items():
        local_items: dict[str, str] = {}
        for key, uri in items.items():
            if not isinstance(uri, str):
                local_items[key] = uri
                continue
            if uri.startswith("s3://"):
                from ophelian.stores.s3 import parse_s3_uri

                _bucket, full_key = parse_s3_uri(uri)
                store_key = _strip_store_prefix(store, full_key)
                local_dst = workspace / "_upstream" / step_name / key
                local_items[key] = str(store.get(store_key, local_dst))
            else:
                local_items[key] = uri
        out[step_name] = local_items
    return out


def _strip_store_prefix(store: ArtifactStore, full_key: str) -> str:
    prefix = getattr(store, "prefix", "") or ""
    if prefix and full_key.startswith(prefix + "/"):
        return full_key[len(prefix) + 1 :]
    return full_key


def _default_workspace(run_id: str, step_name: str) -> str:
    import tempfile

    base = Path(tempfile.gettempdir()) / "ophelian" / run_id / step_name
    base.mkdir(parents=True, exist_ok=True)
    return str(base)


def _eks_install_target(config: AWSConfig) -> str:
    """``ophelian[aws,...]`` extras for EKS pods, mirroring the EC2
    user-data contract (always include ``aws`` so the worker can talk
    to S3)."""
    extras_set = set(config.runtime_extras or ())
    extras_set.add("aws")
    return f"ophelian[{','.join(sorted(extras_set))}]"


def _eks_bootstrap_step(config: AWSConfig, spec_b64: str) -> str:
    """Inline ``sh -c`` script for an EKS Job step container.

    The default ``runtime_image`` is a vanilla ``python:3.12-slim`` so
    we have to bring our own ophelian + extras at pod start, decode the
    step spec into ``/work/step.json`` (which ``step_runner`` requires
    as its positional argument), and then dispatch. Any pre-baked
    image with ``ophelian`` already installed makes the pip step a
    no-op.
    """
    install_target = _eks_install_target(config)
    return (
        "set -eu\n"
        "mkdir -p /work\n"
        "pip install --upgrade pip >/dev/null && "
        f"pip install '{install_target}'\n"
        f"echo {spec_b64} | base64 -d > /work/step.json\n"
        "exec python -m ophelian.runtime.step_runner /work/step.json\n"
    )


def _eks_bootstrap_deploy(config: AWSConfig, port: int) -> str:
    """Inline ``sh -c`` script for an EKS Deployment serve container.

    Mirrors the EC2 deploy contract — the upstream Train step's model
    artifact lives in S3 (the ``OPHELIAN_MODEL_URI`` env var, set by
    :meth:`EKSDriver._deployment_manifest`). Before launching uvicorn
    we sync that prefix to ``/work/model`` and point
    ``OPHELIAN_MODEL_PATH`` at the resulting local file (sklearn /
    xgboost: a single ``.pkl``) or directory (huggingface, pytorch).
    """
    install_target = _eks_install_target(config)
    return (
        "set -eu\n"
        "mkdir -p /work/model\n"
        "pip install --upgrade pip >/dev/null && "
        f"pip install '{install_target}'\n"
        'if [ -n "${OPHELIAN_MODEL_URI:-}" ]; then\n'
        '  if echo "$OPHELIAN_MODEL_URI" | grep -q "^s3://"; then\n'
        '    python -m awscli s3 sync "$OPHELIAN_MODEL_URI" /work/model/ '
        "--no-progress 2>/dev/null || "
        'aws s3 sync "$OPHELIAN_MODEL_URI" /work/model/ --no-progress\n'
        "    NFILES=$(find /work/model -type f | wc -l)\n"
        '    if [ "$NFILES" -eq 1 ]; then\n'
        "      OPHELIAN_MODEL_PATH=$(find /work/model -type f | head -n1)\n"
        "    else\n"
        "      OPHELIAN_MODEL_PATH=/work/model\n"
        "    fi\n"
        "    export OPHELIAN_MODEL_PATH\n"
        "  else\n"
        '    export OPHELIAN_MODEL_PATH="$OPHELIAN_MODEL_URI"\n'
        "  fi\n"
        "fi\n"
        "exec python -m uvicorn --factory "
        "ophelian.runtime.fastapi_runtime:app_from_env "
        f"--host 0.0.0.0 --port {port}\n"
    )


def _user_data_script(config: AWSConfig, request: StepRequest) -> str:
    spec_b64 = _encode_step_spec(request)
    # Always install the ``aws`` extra so step_runner's S3 artifact
    # upload/download path has boto3 available even when the user only
    # asked for, say, ``sklearn`` extras.
    extras_set = set(config.runtime_extras or ())
    extras_set.add("aws")
    extras = ",".join(sorted(extras_set))
    install_target = f"ophelian[{extras}]"
    archive_install = ""
    if config.pipeline_archive_uri:
        archive_install = (
            f"\naws s3 cp {config.pipeline_archive_uri} /tmp/pipeline.tar.gz\n"
            "tar xzf /tmp/pipeline.tar.gz -C /opt/ophelian-pipeline\n"
            "pip install /opt/ophelian-pipeline\n"
        )
    extra = config.user_data or ""
    # For Deploy steps the worker stays alive serving uvicorn; for any
    # other kind it terminates after uploading result.json (the EC2
    # driver issues TerminateInstances regardless, but the explicit
    # `shutdown -h` keeps the box from idling on driver crashes).
    is_deploy = request.kind == "deploy"
    port = getattr(request.node, "port", None) if is_deploy else None
    deploy_serve = ""
    if is_deploy and port:
        # ``step_runner`` (in S3 mode) rewrites every artifact path to
        # an ``s3://...`` URI before writing ``result.json`` so other
        # workers can fetch them. Deploy is the one step that needs the
        # model as a *local* path because the FastAPI runtime hands it
        # to the adapter via ``Path(...)``. So: read the URI back out
        # of ``result.json``, sync the (file or directory) prefix to
        # ``/work/model``, and point ``OPHELIAN_MODEL_PATH`` at the
        # single file inside it (sklearn/xgboost: ``model.pkl``) or at
        # the directory itself (huggingface, pytorch checkpoints).
        deploy_serve = (
            "export OPHELIAN_FRAMEWORK=$(python3 -c \"import json;"
            "d=json.load(open('/work/result.json'));"
            "print(d['info']['framework'])\")\n"
            "MODEL_URI=$(python3 -c \"import json;"
            "d=json.load(open('/work/result.json'));"
            "print(d['artifacts']['model'])\")\n"
            "mkdir -p /work/model\n"
            "if echo \"$MODEL_URI\" | grep -q '^s3://'; then\n"
            "  aws s3 sync \"$MODEL_URI\" /work/model/ --no-progress\n"
            "  NFILES=$(find /work/model -type f | wc -l)\n"
            "  if [ \"$NFILES\" -eq 1 ]; then\n"
            "    export OPHELIAN_MODEL_PATH=$(find /work/model -type f | head -n1)\n"
            "  else\n"
            "    export OPHELIAN_MODEL_PATH=/work/model\n"
            "  fi\n"
            "else\n"
            "  export OPHELIAN_MODEL_PATH=\"$MODEL_URI\"\n"
            "fi\n"
            f"exec python3 -m uvicorn --factory --host 0.0.0.0 --port {port} "
            "ophelian.runtime.fastapi_runtime:app_from_env\n"
        )
    else:
        deploy_serve = "shutdown -h +1\n"

    # `||` and `&&` have equal precedence and are left-associative — wrap
    # the package install in a real if/elif so a successful yum doesn't
    # silently trigger an apt-get install on Amazon Linux.
    pkg_install = (
        "if command -v yum >/dev/null 2>&1; then\n"
        "  yum install -y docker python3-pip\n"
        "elif command -v dnf >/dev/null 2>&1; then\n"
        "  dnf install -y docker python3-pip\n"
        "elif command -v apt-get >/dev/null 2>&1; then\n"
        "  apt-get update && apt-get install -y docker.io python3-pip\n"
        "else\n"
        "  echo 'no supported package manager found' >&2; exit 1\n"
        "fi\n"
    )
    return (
        "#!/bin/bash\n"
        "set -euxo pipefail\n"
        f"{pkg_install}"
        "systemctl enable --now docker || service docker start || true\n"
        f"pip3 install --upgrade pip && pip3 install '{install_target}'\n"
        f"{archive_install}"
        f"export OPHELIAN_RUN_ID={request.run_id}\n"
        f"export OPHELIAN_ARTIFACT_BUCKET={config.artifact_bucket or ''}\n"
        f"export OPHELIAN_ARTIFACT_PREFIX={config.artifact_prefix or ''}\n"
        f"export OPHELIAN_RESUME_FROM={request.resume_from or ''}\n"
        f"export OPHELIAN_STEP_NAME={request.step_name}\n"
        f"export OPHELIAN_STEP_SPEC_B64={spec_b64}\n"
        "mkdir -p /work\n"
        f"echo {spec_b64} | base64 -d > /work/step.json\n"
        "python3 -m ophelian.runtime.step_runner /work/step.json\n"
        f"aws s3 cp /work/result.json s3://{config.artifact_bucket}/{config.artifact_prefix}/{result_json_key(request.run_id, request.step_name)}\n"
        f"{extra}\n"
        f"{deploy_serve}"
    )


def _encode_step_spec(request: StepRequest) -> str:
    spec: dict[str, Any] = {
        "kind": request.kind,
        "node": request.node.model_dump(mode="json"),
        "artifacts": {k: dict(v) for k, v in request.upstream_artifacts.items()},
        "run_id": request.run_id,
    }
    if request.resume_from:
        spec["resume_from"] = request.resume_from
    return base64.b64encode(json.dumps(spec).encode("utf-8")).decode("ascii")


def _serve_framework(request: StepRequest) -> str:
    """Resolve the upstream Train step's framework for a Deploy step.

    ``upstream_artifacts`` only carries opaque URIs, so the framework
    string lives in the parallel ``upstream_info`` map populated by
    :class:`AWSProvider`. Falls back to ``sklearn`` for tests that
    bypass the provider and construct a :class:`StepRequest` by hand.
    """
    upstream_step = getattr(request.node, "model", "")
    info = request.upstream_info.get(upstream_step, {})
    framework = info.get("framework")
    if isinstance(framework, str) and framework:
        return framework
    return "sklearn"


def _serve_model_uri(request: StepRequest) -> str:
    """``s3://...`` URI of the upstream Train step's model artifact.

    Returned empty when no upstream model is present (e.g. a deploy
    that wires its model in some other way) so the bootstrap script's
    guard short-circuits cleanly.
    """
    upstream_step = getattr(request.node, "model", "")
    artifacts = request.upstream_artifacts.get(upstream_step, {})
    return artifacts.get("model", "")


def _build_boto3_client(service: str, config: AWSConfig) -> Any:
    try:
        import boto3
    except ImportError as exc:
        raise _missing_aws_extra(service.upper()) from exc
    try:
        return boto3.client(service, region_name=config.region)
    except Exception as exc:  # pragma: no cover - depends on boto chain
        raise CredentialError(
            f"Could not build a boto3 {service} client for region {config.region!r}: {exc}. "
            "Set AWS credentials via env vars, a profile or an IAM role — see docs/aws.md."
        ) from exc


def _build_kubernetes_client(api: str) -> Any:
    try:
        from kubernetes import client
        from kubernetes import config as k8s_config
    except ImportError as exc:
        raise _missing_aws_extra("EKS") from exc
    try:
        k8s_config.load_kube_config()
    except Exception:
        try:
            k8s_config.load_incluster_config()
        except Exception as exc:  # pragma: no cover - environment dependent
            raise CredentialError(
                "Could not load kubeconfig — set KUBECONFIG or run inside a cluster. "
                "See docs/aws.md for EKS setup."
            ) from exc
    if api == "batch":
        return client.BatchV1Api()
    if api == "apps":
        return client.AppsV1Api()
    if api == "core":
        return client.CoreV1Api()
    raise ValueError(f"Unknown kubernetes API surface: {api!r}")


_AL2023_X86_SSM = "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64"
_AL2023_ARM_SSM = "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-arm64"
_ARM_INSTANCE_PREFIXES = (
    "a1.", "t4g.", "m6g.", "m7g.", "m6gd.", "m7gd.",
    "c6g.", "c7g.", "c6gd.", "c7gd.", "c6gn.", "c7gn.",
    "r6g.", "r7g.", "r6gd.", "r7gd.",
    "x2gd.", "im4gn.", "is4gen.", "g5g.",
)


def _default_ami_for(config: AWSConfig, ssm_client: Any | None = None) -> str:
    """Resolve a default AMI for the configured instance type.

    Looks up the latest Amazon Linux 2023 AMI from SSM Parameter Store
    (``/aws/service/ami-amazon-linux-latest/...``), which is the AWS-
    recommended pattern for region-portable AMI resolution. Returns the
    arm64 image for Graviton instance families and the x86_64 image
    otherwise.

    GPU users should still pin ``ami=`` explicitly to a Deep Learning AMI
    — AL2023 ships without NVIDIA drivers, so the worker would have to
    install them via ``user_data`` to use the GPU. We log a warning when
    this happens but don't fail (drivers can be installed by user_data).
    """
    is_arm = any(config.instance.startswith(p) for p in _ARM_INSTANCE_PREFIXES)
    parameter = _AL2023_ARM_SSM if is_arm else _AL2023_X86_SSM
    client = ssm_client if ssm_client is not None else _build_boto3_client("ssm", config)
    try:
        resp = client.get_parameter(Name=parameter)
    except Exception as exc:
        raise ValueError(
            f"Could not resolve default AMI for region={config.region!r} via SSM "
            f"parameter {parameter!r}: {exc}. Pin an AMI explicitly with ami='ami-...'."
        ) from exc
    ami = (resp.get("Parameter") or {}).get("Value")
    if not isinstance(ami, str) or not ami.startswith("ami-"):
        raise ValueError(
            f"SSM returned an unexpected value for {parameter!r}: {ami!r}. "
            "Pin an AMI explicitly with ami='ami-...'."
        )
    if config.is_gpu_instance:
        logger.warning(
            "Resolved AL2023 AMI %s for GPU instance %s — AL2023 has no NVIDIA "
            "drivers preinstalled. Pin a Deep Learning AMI via `ami=` for "
            "out-of-the-box CUDA support.",
            ami,
            config.instance,
        )
    return ami


def _raise_credential_error(exc: Exception, action: str) -> None:
    """Translate boto3 auth exceptions into a friendly :class:`CredentialError`."""
    response = getattr(exc, "response", None)
    code: str | None = None
    if isinstance(response, dict):
        error = response.get("Error", {})
        if isinstance(error, dict):
            raw_code = error.get("Code")
            if isinstance(raw_code, str):
                code = raw_code
    if code in {"AuthFailure", "UnauthorizedOperation", "AccessDenied", "InvalidClientTokenId"}:
        raise CredentialError(
            f"AWS denied the request during {action}. Check that your credentials and IAM "
            "role have ec2:RunInstances, ec2:TerminateInstances, ec2:DescribeInstances and "
            "the matching s3:* permissions. See docs/aws.md for the minimal policy."
        ) from exc


__all__ = [
    "CloudDriver",
    "CredentialError",
    "EC2Driver",
    "EKSDriver",
    "LocalDriver",
    "StepOutcome",
    "StepRequest",
    "artifact_key",
    "result_json_key",
]


def make_checkpoint(
    run_id: str,
    pipeline: str,
    completed_steps: list[str],
    in_flight_step: str | None,
    artifacts: dict[str, dict[str, str]],
    info: dict[str, Any] | None = None,
) -> Checkpoint:
    """Convenience factory used by the AWS provider when persisting state."""
    return Checkpoint(
        run_id=run_id,
        pipeline=pipeline,
        completed_steps=completed_steps,
        in_flight_step=in_flight_step,
        artifacts=artifacts,
        info=info or {},
    )


def persist_run_checkpoint(
    store: ArtifactStore,
    *,
    run_id: str,
    pipeline: str,
    completed: list[str],
    in_flight: str | None,
    artifacts: dict[str, dict[str, str]],
    info: dict[str, Any] | None = None,
) -> str:
    """Save a checkpoint and return its store URI."""
    cp = make_checkpoint(run_id, pipeline, completed, in_flight, artifacts, info)
    return save_checkpoint(store, cp)


def random_run_id(prefix: str = "run") -> str:
    """Generate a short run id used as the artifact prefix."""
    return f"{prefix}-{uuid.uuid4().hex[:10]}"
