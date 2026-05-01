"""Tests for the AWS env, provider and drivers.

The unit tests run with moto-mocked AWS APIs so they exercise the real
boto3 calls without ever hitting AWS. The integration suite (skipped by
default — see ``tests/test_aws_integration.py``) covers the same scenarios
against a real account.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import boto3
import pytest
from moto import mock_aws
from ophelian import AWS, Data, Deploy, Eval, Pipeline, Train
from ophelian.envs.aws import KNOWN_REGIONS, AWSConfig
from ophelian.providers.aws import AWSProvider
from ophelian.providers.aws_drivers import (
    EC2Driver,
    EKSDriver,
    LocalDriver,
    StepRequest,
    _encode_step_spec,
    artifact_key,
    result_json_key,
    train_checkpoint_key,
)
from ophelian.runtime.spot import (
    Checkpoint,
    SpotInterruption,
    SpotInterruptionMonitor,
    checkpoint_key,
    load_checkpoint,
    save_checkpoint,
)
from ophelian.stores.s3 import S3ArtifactStore

_X = [[0.0, 0.1], [0.1, 0.2], [0.9, 1.0], [1.0, 0.9]]
_Y = [0, 0, 1, 1]


def _make_pipeline(name: str = "aws-test") -> Pipeline:
    return Pipeline(
        [
            Data(name="ds", source="memory://toy", format="inline", options={"X": _X, "y": _Y}),
            Train(
                name="trainer",
                framework="sklearn",
                model="sklearn.linear_model.LogisticRegression",
                data="ds",
                hyperparameters={"max_iter": 200},
            ),
            Eval(name="ev", model="trainer", data="ds", metrics=("accuracy",)),
        ],
        name=name,
    )


@pytest.fixture()
def aws_env() -> Iterator[None]:
    with mock_aws():
        yield


@pytest.fixture()
def store(aws_env: None) -> S3ArtifactStore:
    return S3ArtifactStore(bucket="ophelian-aws-test", region="us-east-1")


# ---------------------------------------------------------------------------
# AWSConfig validation
# ---------------------------------------------------------------------------


def test_aws_config_validates_region() -> None:
    with pytest.raises(ValueError, match="region"):
        AWSConfig(region="not-a-real-region")


def test_aws_config_accepts_lookalike_region() -> None:
    cfg = AWSConfig(region="me-south-1", artifact_bucket="b")
    assert cfg.region == "me-south-1"
    assert "us-east-1" in KNOWN_REGIONS


def test_aws_config_validates_instance_type() -> None:
    with pytest.raises(ValueError, match="instance"):
        AWSConfig(region="us-east-1", instance="not-a-type", artifact_bucket="b")


def test_aws_config_rejects_bad_ami() -> None:
    with pytest.raises(ValueError, match="ami"):
        AWSConfig(region="us-east-1", ami="bad-id", artifact_bucket="b")


def test_aws_config_eks_requires_cluster() -> None:
    with pytest.raises(ValueError, match="eks_cluster"):
        AWSConfig(region="us-east-1", aws_backend="eks", artifact_bucket="b")


def test_aws_config_eks_rejects_spot() -> None:
    with pytest.raises(ValueError, match="Spot"):
        AWSConfig(
            region="us-east-1",
            aws_backend="eks",
            eks_cluster="prod",
            spot=True,
            artifact_bucket="b",
        )


def test_aws_config_iam_role_aliases_instance_profile() -> None:
    cfg = AWSConfig(
        region="us-east-1", artifact_bucket="b", instance_profile="my-role"
    )
    assert cfg.iam_role == "my-role"


def test_aws_factory_returns_provider(aws_env: None) -> None:
    env = AWS(
        region="us-east-1",
        instance="t3.medium",
        artifact_bucket="ophelian-aws-test",
        driver=LocalDriver(),
    )
    assert isinstance(env, AWSProvider)
    assert env.config.region == "us-east-1"
    assert env.config.is_cpu_instance


def test_aws_factory_auto_provisions_default_bucket(aws_env: None) -> None:
    """``AWS(region=..., instance=..., spot=True)`` must work standalone
    per the task spec. We derive a deterministic per-account/per-region
    bucket name and create it lazily so the first call succeeds without
    a manual ``artifact_bucket=`` argument.
    """
    env = AWS(region="us-east-1", instance="t3.medium", spot=True)
    assert isinstance(env, AWSProvider)
    assert env.config.artifact_bucket
    assert env.config.artifact_bucket.startswith("ophelian-artifacts-")
    assert env.config.artifact_bucket.endswith("-us-east-1")


# ---------------------------------------------------------------------------
# LocalDriver + AWSProvider end-to-end
# ---------------------------------------------------------------------------


def test_provider_runs_pipeline_with_local_driver(
    aws_env: None, store: S3ArtifactStore
) -> None:
    driver = LocalDriver()
    provider = AWSProvider(
        config=AWSConfig(region="us-east-1", artifact_bucket=store.bucket),
        driver=driver,
        store=store,
    )

    result = _make_pipeline().run(env=provider)

    assert result.succeeded, [s.error for s in result.steps if s.error]
    assert [s.name for s in result.steps] == ["ds", "trainer", "ev"]
    # Eval reported a real accuracy from real predictions.
    assert result.step("ev").metrics["accuracy"] == 1.0
    # Driver actually saw every step.
    assert driver.executed == ["ds", "trainer", "ev"]
    # All artifacts are now S3 URIs.
    assert result.step("trainer").artifacts["model"].startswith("s3://")
    # Run id is exposed and stable.
    assert provider.run_id.startswith("run-")


def test_provider_persists_artifacts_to_s3(
    aws_env: None, store: S3ArtifactStore
) -> None:
    provider = AWSProvider(
        config=AWSConfig(region="us-east-1", artifact_bucket=store.bucket),
        driver=LocalDriver(),
        store=store,
    )
    _make_pipeline().run(env=provider)

    keys = list(store.list(""))
    assert any("trainer/artifacts/model" in k for k in keys), keys
    assert any("ev/artifacts" in k for k in keys), keys


def test_provider_failure_short_circuits_remaining_steps(
    aws_env: None, store: S3ArtifactStore
) -> None:
    driver = LocalDriver(fail_on=("trainer",))
    provider = AWSProvider(
        config=AWSConfig(region="us-east-1", artifact_bucket=store.bucket),
        driver=driver,
        store=store,
    )
    result = _make_pipeline().run(env=provider)

    assert not result.succeeded
    statuses = {step.name: step.status for step in result.steps}
    assert statuses["ds"] == "success"
    assert statuses["trainer"] == "failed"
    assert "ev" not in statuses
    # Driver tracking confirms only ds executed (trainer raised).
    assert driver.executed == ["ds"]


# ---------------------------------------------------------------------------
# Spot interruption + resume
# ---------------------------------------------------------------------------


def test_spot_interruption_writes_checkpoint(
    aws_env: None, store: S3ArtifactStore
) -> None:
    monitor = SpotInterruptionMonitor()
    monitor.force_trigger()
    driver = LocalDriver(spot_monitor=monitor)
    provider = AWSProvider(
        config=AWSConfig(
            region="us-east-1",
            instance="g4dn.xlarge",
            spot=True,
            artifact_bucket=store.bucket,
        ),
        driver=driver,
        store=store,
    )

    result = _make_pipeline("spot-pipeline").run(env=provider)

    # The provider returns the partial result and marks the run resumable.
    assert not result.succeeded
    failed = [s for s in result.steps if s.status == "failed"]
    assert failed and failed[0].info["resumable"] is True
    assert failed[0].info["checkpoint_uri"].startswith("s3://")
    assert provider.last_checkpoint_uri == failed[0].info["checkpoint_uri"]

    # The checkpoint persisted to S3 and is readable back.
    checkpoint = load_checkpoint(store, provider.run_id)
    assert checkpoint is not None
    assert checkpoint.run_id == provider.run_id
    assert checkpoint.in_flight_step in {"ds", "trainer", "ev"}


def test_spot_resume_skips_completed_steps(
    aws_env: None, store: S3ArtifactStore
) -> None:
    # First run: spot dies after the data step completes (monitor triggers
    # *after* ds executes but before trainer finishes).
    class _Monitor:
        def __init__(self) -> None:
            self.calls = 0

        def check(self) -> bool:
            self.calls += 1
            # The LocalDriver checks the monitor *after* persisting each
            # step's artifacts — so the first call corresponds to ds
            # completing and the second triggers an interruption while
            # trainer is in flight.
            return self.calls >= 2

    config = AWSConfig(region="us-east-1", spot=True, artifact_bucket=store.bucket)
    driver1 = LocalDriver(spot_monitor=_Monitor())  # type: ignore[arg-type]
    provider1 = AWSProvider(config=config, driver=driver1, store=store)
    result1 = _make_pipeline("resume-pipeline").run(env=provider1)
    assert not result1.succeeded
    interrupted_run_id = provider1.run_id
    interrupted = next(s for s in result1.steps if s.status == "failed")
    assert interrupted.info["resumable"]
    # ds completed successfully before trainer was interrupted.
    completed_names = [s.name for s in result1.steps if s.status == "success"]
    assert "ds" in completed_names

    # Second run: same env, with resume_run_id pointing at the failed run.
    resume_config = config.with_resume(interrupted_run_id)
    driver2 = LocalDriver()
    provider2 = AWSProvider(config=resume_config, driver=driver2, store=store)
    result2 = _make_pipeline("resume-pipeline").run(env=provider2)

    assert result2.succeeded, [s.error for s in result2.steps if s.error]
    # Steps already completed in run1 should NOT execute again in run2.
    assert "ds" not in driver2.executed
    # Run id is preserved across the resume.
    assert provider2.run_id == interrupted_run_id


def test_checkpoint_round_trip(store: S3ArtifactStore) -> None:
    cp = Checkpoint(
        run_id="abc",
        pipeline="p",
        completed_steps=["a", "b"],
        in_flight_step="c",
        artifacts={"a": {"k": "s3://b/a"}},
    )
    save_checkpoint(store, cp)
    assert store.exists(checkpoint_key("abc"))
    loaded = load_checkpoint(store, "abc")
    assert loaded == cp


# ---------------------------------------------------------------------------
# Data(s3://) loading
# ---------------------------------------------------------------------------


def test_data_loader_can_read_s3_jsonl(aws_env: None) -> None:
    client = boto3.client("s3", region_name="us-east-1")
    client.create_bucket(Bucket="ds-bucket")
    rows = [{"a": 1.0, "b": 2.0, "y": 0}, {"a": 3.0, "b": 4.0, "y": 1}]
    body = "\n".join(json.dumps(r) for r in rows).encode("utf-8")
    client.put_object(Bucket="ds-bucket", Key="data.jsonl", Body=body)

    from ophelian.data import materialize

    node = Data(name="ds", source="s3://ds-bucket/data.jsonl", format="jsonl")
    payload = materialize(node)
    assert payload["y"] == [0, 1]
    assert payload["X"] == [[1.0, 2.0], [3.0, 4.0]]


# ---------------------------------------------------------------------------
# EC2 driver — boto3 calls go through moto.
# ---------------------------------------------------------------------------


def _seed_amazon_linux_ami(client: Any) -> str:
    """moto's EC2 backend ships canned AMIs — pick the first available."""
    images = client.describe_images()["Images"]
    return images[0]["ImageId"]


def test_ec2_driver_provisions_and_terminates_an_instance(
    aws_env: None, store: S3ArtifactStore
) -> None:
    ec2 = boto3.client("ec2", region_name="us-east-1")
    ami = _seed_amazon_linux_ami(ec2)
    config = AWSConfig(
        region="us-east-1",
        instance="t3.medium",
        ami=ami,
        artifact_bucket=store.bucket,
        poll_interval_seconds=0.01,
        timeout_seconds=60,
    )
    driver = EC2Driver(config, ec2_client=ec2, sleep=lambda _s: None)

    request = StepRequest(
        run_id="run-x",
        pipeline_name="p",
        step_name="trainer",
        kind="train",
        node=Train(
            name="trainer",
            framework="sklearn",
            model="sklearn.linear_model.LogisticRegression",
            data="ds",
        ),
    )

    # Pre-publish the result.json that the worker's user-data would upload.
    store.put_bytes(
        result_json_key("run-x", "trainer"),
        json.dumps(
            {
                "name": "trainer",
                "kind": "train",
                "status": "success",
                "metrics": {},
                "artifacts": {"model": "s3://b/model"},
                "info": {},
            }
        ).encode("utf-8"),
    )

    outcome = driver.execute(request, store)

    assert outcome.result.status == "success"
    assert outcome.instance_id and outcome.instance_id.startswith("i-")
    # Instance was terminated as part of the try/finally.
    description = ec2.describe_instances(InstanceIds=[outcome.instance_id])
    state = description["Reservations"][0]["Instances"][0]["State"]["Name"]
    assert state in {"terminated", "shutting-down"}


def test_ec2_driver_marks_spot_instance(
    aws_env: None, store: S3ArtifactStore
) -> None:
    ec2 = boto3.client("ec2", region_name="us-east-1")
    ami = _seed_amazon_linux_ami(ec2)
    config = AWSConfig(
        region="us-east-1",
        instance="g4dn.xlarge",
        spot=True,
        spot_max_price=0.5,
        ami=ami,
        artifact_bucket=store.bucket,
        poll_interval_seconds=0.01,
        timeout_seconds=60,
    )
    driver = EC2Driver(config, ec2_client=ec2, sleep=lambda _s: None)
    request = StepRequest(
        run_id="run-spot",
        pipeline_name="p",
        step_name="trainer",
        kind="train",
        node=Train(
            name="trainer",
            framework="sklearn",
            model="sklearn.linear_model.LogisticRegression",
            data="ds",
        ),
    )
    store.put_bytes(
        result_json_key("run-spot", "trainer"),
        json.dumps(
            {"name": "trainer", "kind": "train", "status": "success"}
        ).encode("utf-8"),
    )
    outcome = driver.execute(request, store)
    assert outcome.spot is True
    assert outcome.instance_type == "g4dn.xlarge"


def test_ec2_driver_terminates_on_failure(
    aws_env: None, store: S3ArtifactStore
) -> None:
    ec2 = boto3.client("ec2", region_name="us-east-1")
    ami = _seed_amazon_linux_ami(ec2)
    config = AWSConfig(
        region="us-east-1",
        instance="t3.medium",
        ami=ami,
        artifact_bucket=store.bucket,
        poll_interval_seconds=0.01,
        timeout_seconds=2,
    )
    driver = EC2Driver(config, ec2_client=ec2, sleep=lambda _s: None)

    request = StepRequest(
        run_id="run-fail",
        pipeline_name="p",
        step_name="trainer",
        kind="train",
        node=Train(
            name="trainer",
            framework="sklearn",
            model="sklearn.linear_model.LogisticRegression",
            data="ds",
        ),
    )
    # No result file is uploaded → driver times out → terminate still runs.
    with pytest.raises(TimeoutError):
        driver.execute(request, store)

    # Find any instances tagged for our run.
    description = ec2.describe_instances(
        Filters=[{"Name": "tag:ophelian/run-id", "Values": ["run-fail"]}]
    )
    instances = [
        i for r in description.get("Reservations", []) for i in r["Instances"]
    ]
    assert instances, "Expected a worker instance to have been provisioned"
    for inst in instances:
        assert inst["State"]["Name"] in {"terminated", "shutting-down"}


def test_ec2_driver_propagates_spot_interruption(
    aws_env: None, store: S3ArtifactStore
) -> None:
    ec2 = boto3.client("ec2", region_name="us-east-1")
    ami = _seed_amazon_linux_ami(ec2)
    monitor = SpotInterruptionMonitor()
    monitor.force_trigger()
    config = AWSConfig(
        region="us-east-1",
        instance="g4dn.xlarge",
        spot=True,
        ami=ami,
        artifact_bucket=store.bucket,
        poll_interval_seconds=0.01,
        timeout_seconds=10,
    )
    driver = EC2Driver(
        config, ec2_client=ec2, spot_monitor=monitor, sleep=lambda _s: None
    )
    request = StepRequest(
        run_id="run-int",
        pipeline_name="p",
        step_name="trainer",
        kind="train",
        node=Train(
            name="trainer",
            framework="sklearn",
            model="sklearn.linear_model.LogisticRegression",
            data="ds",
        ),
    )
    with pytest.raises(SpotInterruption):
        driver.execute(request, store)


# ---------------------------------------------------------------------------
# EKS driver — pure manifest assertions (no real cluster).
# ---------------------------------------------------------------------------


def _make_eks_driver(store: S3ArtifactStore) -> tuple[EKSDriver, MagicMock, MagicMock, MagicMock]:
    config = AWSConfig(
        region="us-east-1",
        instance="m5.large",
        aws_backend="eks",
        eks_cluster="prod-ml",
        eks_namespace="ophelian",
        artifact_bucket=store.bucket,
        poll_interval_seconds=0.01,
        timeout_seconds=2,
    )
    batch = MagicMock()
    apps = MagicMock()
    core = MagicMock()
    driver = EKSDriver(
        config, batch_api=batch, apps_api=apps, core_api=core, sleep=lambda _s: None
    )
    return driver, batch, apps, core


def test_eks_driver_submits_job_for_train(
    aws_env: None, store: S3ArtifactStore
) -> None:
    driver, batch, _apps, _core = _make_eks_driver(store)
    request = StepRequest(
        run_id="run-eks",
        pipeline_name="p",
        step_name="trainer",
        kind="train",
        node=Train(
            name="trainer",
            framework="sklearn",
            model="sklearn.linear_model.LogisticRegression",
            data="ds",
        ),
    )
    store.put_bytes(
        result_json_key("run-eks", "trainer"),
        json.dumps({"name": "trainer", "kind": "train", "status": "success"}).encode(),
    )
    outcome = driver.execute(request, store)

    assert outcome.result.status == "success"
    assert batch.create_namespaced_job.called
    _args, kwargs = batch.create_namespaced_job.call_args
    manifest = kwargs["body"]
    assert manifest["kind"] == "Job"
    assert manifest["metadata"]["namespace"] == "ophelian"
    assert manifest["metadata"]["labels"]["ophelian/run-id"] == "run-eks"
    # Bootstrap script must (a) install ophelian into the default
    # python:3.12-slim image and (b) materialise /work/step.json before
    # invoking step_runner, otherwise the pod would crash immediately.
    container = manifest["spec"]["template"]["spec"]["containers"][0]
    assert container["command"] == ["sh", "-c"]
    bootstrap = container["args"][0]
    assert "pip install" in bootstrap
    assert "ophelian[" in bootstrap
    assert "/work/step.json" in bootstrap
    assert "base64 -d > /work/step.json" in bootstrap
    assert "python -m ophelian.runtime.step_runner /work/step.json" in bootstrap


def test_eks_driver_submits_deployment_for_deploy(
    aws_env: None, store: S3ArtifactStore
) -> None:
    driver, _batch, apps, core = _make_eks_driver(store)
    request = StepRequest(
        run_id="run-eks",
        pipeline_name="p",
        step_name="serve",
        kind="deploy",
        node=Deploy(name="serve", model="trainer", port=8080, replicas=3),
    )
    outcome = driver.execute(request, store)

    assert outcome.result.status == "success"
    assert apps.create_namespaced_deployment.called
    assert core.create_namespaced_service.called
    deployment = apps.create_namespaced_deployment.call_args.kwargs["body"]
    assert deployment["spec"]["replicas"] == 3
    service = core.create_namespaced_service.call_args.kwargs["body"]
    assert service["spec"]["type"] == "LoadBalancer"
    # Endpoint URL is reported.
    assert "ophelian-serve" in outcome.result.artifacts["endpoint_url"]
    # Deploy container must also bootstrap ophelian before exec'ing
    # uvicorn — otherwise the default python:3.12-slim image has no
    # ophelian.runtime.fastapi_runtime module to import.
    container = deployment["spec"]["template"]["spec"]["containers"][0]
    assert container["command"] == ["sh", "-c"]
    bootstrap = container["args"][0]
    assert "pip install" in bootstrap
    assert "ophelian[" in bootstrap
    assert "uvicorn" in bootstrap
    assert "ophelian.runtime.fastapi_runtime:app_from_env" in bootstrap
    assert "--port 8080" in bootstrap


def test_eks_driver_teardown_deletes_resources(
    aws_env: None, store: S3ArtifactStore
) -> None:
    driver, batch, _apps, _core = _make_eks_driver(store)
    store.put_bytes(
        result_json_key("run-eks", "trainer"),
        json.dumps({"name": "trainer", "kind": "train", "status": "success"}).encode(),
    )
    request = StepRequest(
        run_id="run-eks",
        pipeline_name="p",
        step_name="trainer",
        kind="train",
        node=Train(
            name="trainer",
            framework="sklearn",
            model="sklearn.linear_model.LogisticRegression",
            data="ds",
        ),
    )
    driver.execute(request, store)
    driver.teardown()
    assert batch.delete_namespaced_job.called


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_artifact_key_and_result_key_are_stable() -> None:
    assert artifact_key("r1", "trainer", "model") == "runs/r1/trainer/artifacts/model"
    assert result_json_key("r1", "trainer") == "runs/r1/trainer/result.json"


def test_provider_describe(aws_env: None) -> None:
    env = AWS(
        region="us-east-1",
        instance="t3.medium",
        artifact_bucket="ophelian-aws-test",
        driver=LocalDriver(),
    )
    assert "aws[ec2]" in env.describe()
    assert "us-east-1" in env.describe()


def test_credential_error_message_is_actionable(aws_env: None) -> None:
    """A boto AuthFailure is translated into our friendly CredentialError."""
    from botocore.exceptions import ClientError

    ec2 = MagicMock()
    ec2.run_instances.side_effect = ClientError(
        {"Error": {"Code": "AuthFailure", "Message": "denied"}}, "RunInstances"
    )
    config = AWSConfig(
        region="us-east-1",
        instance="t3.medium",
        artifact_bucket="b",
        ami="ami-1234567",
    )
    driver = EC2Driver(config, ec2_client=ec2, sleep=lambda _s: None)
    from ophelian.providers.aws_drivers import CredentialError

    request = StepRequest(
        run_id="r",
        pipeline_name="p",
        step_name="s",
        kind="train",
        node=Train(
            name="s",
            framework="sklearn",
            model="sklearn.linear_model.LogisticRegression",
            data="ds",
        ),
    )
    with pytest.raises(CredentialError, match="ec2:RunInstances"):
        driver.execute(request, S3ArtifactStore(bucket="b", region="us-east-1"))


def test_resume_run_id_env(aws_env: None) -> None:
    """The env exposes a ``with_resume`` ergonomic for chaining."""
    cfg = AWSConfig(region="us-east-1", artifact_bucket="b")
    resumed = cfg.with_resume("run-xyz")
    assert resumed.resume_run_id == "run-xyz"
    assert cfg.resume_run_id is None


def test_aws_integration_marker_is_skipped_by_default() -> None:
    """Sanity — make sure the integration env var actually opts in."""
    assert os.environ.get("OPHELIAN_AWS_INTEGRATION_TESTS", "") in {"", "0"}


def test_ec2_driver_keeps_deploy_instance_alive_and_rewrites_endpoint(
    aws_env: None, store: S3ArtifactStore
) -> None:
    ec2 = boto3.client("ec2", region_name="us-east-1")
    ami = _seed_amazon_linux_ami(ec2)
    config = AWSConfig(
        region="us-east-1",
        instance="t3.medium",
        ami=ami,
        artifact_bucket=store.bucket,
        poll_interval_seconds=0.01,
        timeout_seconds=60,
    )
    driver = EC2Driver(config, ec2_client=ec2, sleep=lambda _s: None)

    request = StepRequest(
        run_id="run-deploy",
        pipeline_name="p",
        step_name="serve",
        kind="deploy",
        node=Deploy(name="serve", model="trainer", port=8080, replicas=1),
    )
    # Worker would emit this after launching uvicorn locally.
    store.put_bytes(
        result_json_key("run-deploy", "serve"),
        json.dumps(
            {
                "name": "serve",
                "kind": "deploy",
                "status": "success",
                "metrics": {},
                "artifacts": {"endpoint_url": "http://localhost:8080"},
                "info": {
                    "predict": "http://localhost:8080/predict",
                    "health": "http://localhost:8080/health",
                    "framework": "sklearn",
                    "model_path": "/work/artifacts/model.joblib",
                },
            }
        ).encode("utf-8"),
    )

    outcome = driver.execute(request, store)

    # Deploy instance must NOT be terminated — it's now serving traffic.
    assert outcome.instance_id is not None
    description = ec2.describe_instances(InstanceIds=[outcome.instance_id])
    state = description["Reservations"][0]["Instances"][0]["State"]["Name"]
    assert state == "running", f"deploy instance was terminated (state={state})"

    # Endpoint URL must be rewritten to the EC2 public DNS, not localhost.
    info = outcome.result.info
    public_dns = description["Reservations"][0]["Instances"][0].get("PublicDnsName")
    assert public_dns, "moto should populate PublicDnsName for default-VPC instances"
    assert info["public_dns"] == public_dns
    assert info["predict"] == f"http://{public_dns}:8080/predict"
    assert info["health"] == f"http://{public_dns}:8080/health"
    assert outcome.result.artifacts["endpoint_url"] == f"http://{public_dns}:8080"
    assert info["framework"] == "sklearn"  # original info preserved

    # Cleanup so the test doesn't leak a "running" instance into other tests.
    ec2.terminate_instances(InstanceIds=[outcome.instance_id])


def test_ec2_driver_detects_spot_reclamation_via_describe_instances(
    aws_env: None, store: S3ArtifactStore
) -> None:
    """No injected monitor — the driver should detect a real spot
    reclamation by polling DescribeInstances and seeing the worker move
    to ``shutting-down`` while marked ``InstanceLifecycle=spot``.
    """
    ami = "ami-12345678"
    config = AWSConfig(
        region="us-east-1",
        instance="g4dn.xlarge",
        spot=True,
        ami=ami,
        artifact_bucket=store.bucket,
        poll_interval_seconds=0.001,
        timeout_seconds=5,
    )
    ec2 = MagicMock()
    ec2.run_instances.return_value = {"Instances": [{"InstanceId": "i-spot01"}]}
    # First describe (during _wait_for_running) → running.
    # Second describe (inside _wait_for_result → _is_spot_interrupted) →
    # spot lifecycle, shutting-down state → reclamation.
    ec2.describe_instances.side_effect = [
        {
            "Reservations": [
                {
                    "Instances": [
                        {
                            "InstanceId": "i-spot01",
                            "State": {"Name": "running"},
                            "InstanceLifecycle": "spot",
                        }
                    ]
                }
            ]
        },
        {
            "Reservations": [
                {
                    "Instances": [
                        {
                            "InstanceId": "i-spot01",
                            "State": {"Name": "shutting-down"},
                            "InstanceLifecycle": "spot",
                        }
                    ]
                }
            ]
        },
    ]
    driver = EC2Driver(config, ec2_client=ec2, sleep=lambda _s: None)
    request = StepRequest(
        run_id="run-reclaim",
        pipeline_name="p",
        step_name="trainer",
        kind="train",
        node=Train(
            name="trainer",
            framework="sklearn",
            model="sklearn.linear_model.LogisticRegression",
            data="ds",
        ),
    )
    with pytest.raises(SpotInterruption, match="reclaimed"):
        driver.execute(request, store)
    assert ec2.terminate_instances.called  # finally-block cleanup ran


def test_step_runner_dispatches_deploy(tmp_path: Path) -> None:
    """``ophelian.runtime.step_runner`` must produce a ``result.json`` for
    deploy steps (regression — it used to raise ``ValueError``).
    """
    from ophelian.runtime import step_runner

    # Train a tiny model so deploy has something to load.
    work = tmp_path / "work"
    work.mkdir()
    train_spec = work / "train.json"
    ds_path = _make_inline_ds(work)
    train_spec.write_text(
        json.dumps(
            {
                "kind": "train",
                "node": {
                    "name": "trainer",
                    "framework": "sklearn",
                    "model": "sklearn.linear_model.LogisticRegression",
                    "data": "ds",
                    "hyperparameters": {"max_iter": 200},
                },
                "artifacts": {"ds": {"dataset": str(ds_path)}},
            }
        )
    )
    rc = step_runner.run(train_spec)
    assert rc == 0
    train_result = json.loads((work / "result.json").read_text())
    model_path = train_result["artifacts"]["model"]

    # Now run a deploy step against that model.
    deploy_dir = tmp_path / "deploy"
    deploy_dir.mkdir()
    deploy_spec = deploy_dir / "step.json"
    deploy_spec.write_text(
        json.dumps(
            {
                "kind": "deploy",
                "node": {
                    "name": "serve",
                    "model": "trainer",
                    "port": 8080,
                    "replicas": 1,
                },
                "artifacts": {"trainer": {"model": model_path}},
            }
        )
    )
    rc = step_runner.run(deploy_spec)
    assert rc == 0
    payload = json.loads((deploy_dir / "result.json").read_text())
    assert payload["status"] == "success"
    assert payload["kind"] == "deploy"
    # The handler must surface a framework + model artifact so the worker
    # user-data can launch uvicorn with the correct env.
    assert payload["info"].get("framework") in {"sklearn", "auto"}
    assert payload["artifacts"].get("model")


def test_default_ami_resolves_via_ssm_for_x86_and_arm(aws_env: None) -> None:
    """``_default_ami_for`` must read from SSM Parameter Store rather
    than returning hard-coded placeholders.
    """
    from ophelian.providers.aws_drivers import _default_ami_for

    captured: list[str] = []

    def fake_get_parameter(*, Name: str) -> dict[str, Any]:
        captured.append(Name)
        return {"Parameter": {"Value": "ami-deadbeef0001"}}

    ssm = MagicMock()
    ssm.get_parameter.side_effect = fake_get_parameter

    cfg_x86 = AWSConfig(region="us-east-1", instance="t3.medium", artifact_bucket="b")
    ami = _default_ami_for(cfg_x86, ssm_client=ssm)
    assert ami == "ami-deadbeef0001"
    assert captured[-1].endswith("al2023-ami-kernel-default-x86_64")

    cfg_arm = AWSConfig(region="us-east-1", instance="t4g.small", artifact_bucket="b")
    _default_ami_for(cfg_arm, ssm_client=ssm)
    assert captured[-1].endswith("al2023-ami-kernel-default-arm64")


def test_default_ami_raises_friendly_error_when_ssm_fails(aws_env: None) -> None:
    """SSM failures must surface as an actionable ValueError that points
    the user at the ``ami=`` override.
    """
    from ophelian.providers.aws_drivers import _default_ami_for

    ssm = MagicMock()
    ssm.get_parameter.side_effect = RuntimeError("AccessDenied")
    cfg = AWSConfig(region="us-east-1", instance="t3.medium", artifact_bucket="b")
    with pytest.raises(ValueError, match=r"Pin an AMI explicitly"):
        _default_ami_for(cfg, ssm_client=ssm)


def test_validate_instance_type_in_region_accepts_known_offerings() -> None:
    cfg = AWSConfig(region="us-east-1", instance="t3.medium", artifact_bucket="b")
    ec2 = MagicMock()
    ec2.describe_instance_type_offerings.return_value = {
        "InstanceTypeOfferings": [
            {"InstanceType": "t3.medium", "Location": "us-east-1"},
        ]
    }
    cfg.validate_instance_type_in_region(ec2)  # no raise
    ec2.describe_instance_type_offerings.assert_called_once()


def test_validate_instance_type_in_region_rejects_unavailable_combo() -> None:
    cfg = AWSConfig(region="ca-central-1", instance="p4d.24xlarge", artifact_bucket="b")
    ec2 = MagicMock()
    ec2.describe_instance_type_offerings.return_value = {"InstanceTypeOfferings": []}
    with pytest.raises(ValueError, match="not offered in region"):
        cfg.validate_instance_type_in_region(ec2)


def test_validate_instance_type_in_region_swallows_iam_errors() -> None:
    """A restrictive IAM policy (no ``ec2:DescribeInstanceTypeOfferings``)
    must NOT block the run — the check is best-effort.
    """
    cfg = AWSConfig(region="us-east-1", instance="t3.medium", artifact_bucket="b")
    ec2 = MagicMock()
    ec2.describe_instance_type_offerings.side_effect = RuntimeError("AccessDenied")
    cfg.validate_instance_type_in_region(ec2)  # no raise


def test_aws_provider_pipeline_keeps_deploy_instance_alive_e2e(
    aws_env: None, store: S3ArtifactStore
) -> None:
    """End-to-end: ``Pipeline.run(env=AWS(...))`` with a Deploy step
    must leave the Deploy worker running after the pipeline returns,
    while terminating the upstream train/eval workers. ``cleanup_deploys``
    then takes the Deploy box down explicitly.
    """
    ec2 = boto3.client("ec2", region_name="us-east-1")
    ami = _seed_amazon_linux_ami(ec2)
    config = AWSConfig(
        region="us-east-1",
        instance="t3.medium",
        ami=ami,
        artifact_bucket=store.bucket,
        poll_interval_seconds=0.01,
        timeout_seconds=60,
    )
    driver = EC2Driver(config, ec2_client=ec2, sleep=lambda _s: None)
    provider = AWSProvider(config=config, driver=driver, store=store)
    run_id = provider.run_id

    # Mimic the worker by pre-publishing the result.json each step would
    # upload from /work — the EC2 driver only polls S3 for these.
    def publish(step: str, kind: str, info: dict[str, Any] | None = None,
                artifacts: dict[str, str] | None = None) -> None:
        store.put_bytes(
            result_json_key(run_id, step),
            json.dumps(
                {
                    "name": step,
                    "kind": kind,
                    "status": "success",
                    "metrics": {},
                    "artifacts": artifacts or {},
                    "info": info or {},
                }
            ).encode("utf-8"),
        )

    publish("ds", "data", artifacts={"dataset": "s3://bucket/ds"})
    publish("trainer", "train", artifacts={"model": "s3://bucket/model"})
    publish("ev", "eval", artifacts={"report": "s3://bucket/report"})
    publish(
        "serve",
        "deploy",
        artifacts={"endpoint_url": "http://localhost:8080"},
        info={
            "predict": "http://localhost:8080/predict",
            "health": "http://localhost:8080/health",
            "framework": "sklearn",
        },
    )

    pipeline = Pipeline(
        [
            Data(name="ds", source="memory://", format="inline",
                 options={"X": _X, "y": _Y}),
            Train(name="trainer", framework="sklearn",
                  model="sklearn.linear_model.LogisticRegression", data="ds"),
            Eval(name="ev", model="trainer", data="ds", metrics=("accuracy",)),
            Deploy(name="serve", model="trainer", port=8080, replicas=1),
        ],
        name="aws-deploy-e2e",
    )

    result = pipeline.run(env=provider)
    assert result.succeeded, [s.error for s in result.steps if s.error]

    serve_step = result.step("serve")
    serve_instance = serve_step.info["instance_id"]
    public_dns = serve_step.info["public_dns"]
    assert serve_instance.startswith("i-")
    assert public_dns
    # The endpoint URL was rewritten to the EC2 public DNS, not localhost.
    assert serve_step.info["predict"] == f"http://{public_dns}:8080/predict"
    assert serve_step.artifacts["endpoint_url"] == f"http://{public_dns}:8080"

    # Critical: the deploy worker must still be running after the
    # pipeline returns (regression — was being killed by teardown()).
    desc = ec2.describe_instances(InstanceIds=[serve_instance])
    state = desc["Reservations"][0]["Instances"][0]["State"]["Name"]
    assert state == "running", f"deploy worker was terminated post-run (state={state})"

    # Upstream train/eval/data instances must be terminated by the
    # driver's per-step finally block.
    other_ids = {s.info["instance_id"] for s in result.steps if s.name != "serve"}
    for inst_id in other_ids:
        d = ec2.describe_instances(InstanceIds=[inst_id])
        st = d["Reservations"][0]["Instances"][0]["State"]["Name"]
        assert st in {"terminated", "shutting-down"}

    # cleanup_deploys() takes down the long-lived deploy worker.
    provider.cleanup_deploys()
    desc = ec2.describe_instances(InstanceIds=[serve_instance])
    state = desc["Reservations"][0]["Instances"][0]["State"]["Name"]
    assert state in {"terminated", "shutting-down"}


def _make_inline_ds(work: Path) -> Path:
    """Materialise a tiny inline dataset by running a Data step."""
    from ophelian.runtime import step_runner

    ds_dir = work / "ds"
    ds_dir.mkdir()
    spec = ds_dir / "step.json"
    spec.write_text(
        json.dumps(
            {
                "kind": "data",
                "node": {
                    "name": "ds",
                    "source": "memory://toy",
                    "format": "inline",
                    "options": {"X": _X, "y": _Y},
                },
                "artifacts": {},
            }
        )
    )
    assert step_runner.run(spec) == 0
    payload = json.loads((ds_dir / "result.json").read_text())
    return Path(payload["artifacts"]["dataset"])


def test_local_driver_workspace_isolated(
    aws_env: None, store: S3ArtifactStore, tmp_path: Path
) -> None:
    driver = LocalDriver()
    request = StepRequest(
        run_id="run-iso",
        pipeline_name="p",
        step_name="ds",
        kind="data",
        node=Data(name="ds", source="memory://", format="inline", options={"X": _X, "y": _Y}),
        workspace=tmp_path / "ws",
    )
    outcome = driver.execute(request, store)
    assert outcome.result.status == "success"
    assert (tmp_path / "ws").exists()


# ---------------------------------------------------------------------------
# step_runner artifact persistence (the missing EC2 worker contract)
# ---------------------------------------------------------------------------


def _run_step(
    work: Path,
    *,
    kind: str,
    node: dict[str, Any],
    artifacts: dict[str, dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Invoke ``step_runner.run`` and return the parsed ``result.json``."""
    from ophelian.runtime import step_runner

    work.mkdir(parents=True, exist_ok=True)
    spec = work / "step.json"
    spec.write_text(json.dumps({"kind": kind, "node": node, "artifacts": artifacts or {}}))
    rc = step_runner.run(spec)
    payload = json.loads((work / "result.json").read_text())
    assert rc == 0, payload
    return payload


def test_step_runner_uploads_artifacts_to_s3_when_bucket_set(
    aws_env: None, store: S3ArtifactStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When ``OPHELIAN_ARTIFACT_BUCKET`` + ``OPHELIAN_RUN_ID`` are set
    (as the EC2 user-data script does), produced artifacts must be
    uploaded to S3 and ``result.artifacts`` must contain ``s3://...``
    URIs — otherwise the next EC2 instance has no way to read them.
    """
    monkeypatch.setenv("OPHELIAN_ARTIFACT_BUCKET", store.bucket)
    monkeypatch.setenv("OPHELIAN_RUN_ID", "run-upload")

    payload = _run_step(
        tmp_path / "ds",
        kind="data",
        node={"name": "ds", "source": "memory://toy", "format": "inline",
              "options": {"X": _X, "y": _Y}},
    )
    dataset_uri = payload["artifacts"]["dataset"]
    assert dataset_uri.startswith(f"s3://{store.bucket}/runs/run-upload/ds/artifacts/dataset"), (
        dataset_uri
    )

    s3 = boto3.client("s3", region_name="us-east-1")
    listing = s3.list_objects_v2(
        Bucket=store.bucket, Prefix="runs/run-upload/ds/artifacts/dataset"
    )
    keys = [obj["Key"] for obj in listing.get("Contents", [])]
    assert keys, "step_runner should have uploaded the dataset directory to S3"


def test_step_runner_materialises_upstream_s3_artifacts(
    aws_env: None, store: S3ArtifactStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A downstream step running on a fresh EC2 box only sees ``s3://...``
    pointers from upstream — step_runner must download them transparently
    so the in-process Train handler keeps reading plain local paths.

    This is the regression the round-3 review caught: without this the
    Train step would receive ``/work/...`` paths that don't exist on the
    new instance and immediately fail.
    """
    monkeypatch.setenv("OPHELIAN_ARTIFACT_BUCKET", store.bucket)
    monkeypatch.setenv("OPHELIAN_RUN_ID", "run-mat")

    ds_payload = _run_step(
        tmp_path / "ds",
        kind="data",
        node={"name": "ds", "source": "memory://toy", "format": "inline",
              "options": {"X": _X, "y": _Y}},
    )
    upstream_dataset = ds_payload["artifacts"]["dataset"]
    assert upstream_dataset.startswith("s3://"), upstream_dataset

    train_payload = _run_step(
        tmp_path / "trainer",
        kind="train",
        node={
            "name": "trainer",
            "framework": "sklearn",
            "model": "sklearn.linear_model.LogisticRegression",
            "data": "ds",
            "hyperparameters": {"max_iter": 200},
        },
        artifacts={"ds": {"dataset": upstream_dataset}},
    )
    assert train_payload["status"] == "success", train_payload.get("error")
    model_uri = train_payload["artifacts"]["model"]
    assert model_uri.startswith(f"s3://{store.bucket}/runs/run-mat/trainer/artifacts/model"), (
        model_uri
    )
    # Confirm the downloaded upstream actually landed in /work/_upstream
    # — proves materialisation, not just that the upload worked.
    assert (tmp_path / "trainer" / "_upstream" / "ds" / "dataset").exists()


def test_step_runner_no_op_when_bucket_env_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """In container mode (no OPHELIAN_ARTIFACT_BUCKET) step_runner must
    keep emitting plain local paths so the existing Standalone container
    contract is preserved.
    """
    monkeypatch.delenv("OPHELIAN_ARTIFACT_BUCKET", raising=False)
    monkeypatch.delenv("OPHELIAN_RUN_ID", raising=False)
    payload = _run_step(
        tmp_path / "ds",
        kind="data",
        node={"name": "ds", "source": "memory://toy", "format": "inline",
              "options": {"X": _X, "y": _Y}},
    )
    assert not payload["artifacts"]["dataset"].startswith("s3://")


# ---------------------------------------------------------------------------
# Deploy security group auto-provisioning (round-3 review fix)
# ---------------------------------------------------------------------------


def test_ec2_driver_auto_creates_deploy_security_group(
    aws_env: None, store: S3ArtifactStore
) -> None:
    """For Deploy steps with no user-supplied ``security_group``, the
    driver must create one with ingress on the deploy port — otherwise
    the public endpoint is unreachable from the internet.
    """
    ec2 = boto3.client("ec2", region_name="us-east-1")
    ami = _seed_amazon_linux_ami(ec2)
    config = AWSConfig(
        region="us-east-1",
        instance="t3.medium",
        ami=ami,
        artifact_bucket=store.bucket,
        poll_interval_seconds=0.01,
        timeout_seconds=10,
    )
    driver = EC2Driver(config, ec2_client=ec2, sleep=lambda _s: None)

    # Pre-publish the deploy result so EC2Driver's poller succeeds.
    store.put_bytes(
        result_json_key("run-sg", "serve"),
        json.dumps(
            {
                "name": "serve",
                "kind": "deploy",
                "status": "success",
                "metrics": {},
                "artifacts": {"endpoint_url": "http://localhost:9999"},
                "info": {"framework": "sklearn"},
            }
        ).encode("utf-8"),
    )
    request = StepRequest(
        run_id="run-sg",
        pipeline_name="p",
        step_name="serve",
        kind="deploy",
        node=Deploy(name="serve", model="trainer", port=9999, replicas=1),
    )
    outcome = driver.execute(request, store)
    assert outcome.result.status == "success"

    sg = ec2.describe_security_groups(
        Filters=[{"Name": "group-name", "Values": ["ophelian-deploy-9999"]}]
    )["SecurityGroups"]
    assert sg, "deploy SG should have been auto-created"
    perms = sg[0]["IpPermissions"]
    assert any(
        p.get("FromPort") == 9999 and p.get("ToPort") == 9999
        and any(r.get("CidrIp") == "0.0.0.0/0" for r in p.get("IpRanges", []))
        for p in perms
    ), perms
    # Cleanup so other tests don't fight over the same SG name.
    driver.cleanup_deploys()


def test_ec2_driver_skips_sg_provisioning_for_non_deploy_steps(
    aws_env: None, store: S3ArtifactStore
) -> None:
    """Train/eval/data workers don't need ingress — verify we don't
    waste API calls (or quota) creating SGs for them.
    """
    ec2 = boto3.client("ec2", region_name="us-east-1")
    ami = _seed_amazon_linux_ami(ec2)
    config = AWSConfig(
        region="us-east-1",
        instance="t3.medium",
        ami=ami,
        artifact_bucket=store.bucket,
        poll_interval_seconds=0.01,
        timeout_seconds=10,
    )
    driver = EC2Driver(config, ec2_client=ec2, sleep=lambda _s: None)
    store.put_bytes(
        result_json_key("run-nosg", "ds"),
        json.dumps(
            {"name": "ds", "kind": "data", "status": "success", "metrics": {},
             "artifacts": {"dataset": "s3://x/ds"}, "info": {}}
        ).encode("utf-8"),
    )
    request = StepRequest(
        run_id="run-nosg",
        pipeline_name="p",
        step_name="ds",
        kind="data",
        node=Data(name="ds", source="memory://", format="inline",
                  options={"X": _X, "y": _Y}),
    )
    driver.execute(request, store)
    sgs = ec2.describe_security_groups(
        Filters=[{"Name": "group-name", "Values": ["ophelian-deploy-*"]}]
    )["SecurityGroups"]
    assert sgs == []


# ---------------------------------------------------------------------------
# Train resume protocol (round-4 review fix)
# ---------------------------------------------------------------------------


def test_train_checkpoint_key_is_stable() -> None:
    assert (
        train_checkpoint_key("run-xyz", "trainer")
        == "runs/run-xyz/checkpoints/trainer"
    )


def test_provider_train_checkpoint_uri_includes_prefix(aws_env: None) -> None:
    """The URI passed back to the worker on retry must respect the
    user-pinned ``artifact_prefix`` so we don't write outside it."""
    store = S3ArtifactStore(
        bucket="ophelian-aws-test", prefix="team-a", region="us-east-1"
    )
    provider = AWSProvider(
        config=AWSConfig(
            region="us-east-1",
            artifact_bucket="ophelian-aws-test",
            artifact_prefix="team-a",
        ),
        driver=LocalDriver(),
        store=store,
    )
    uri = provider._train_checkpoint_uri("run-1", "trainer")
    assert uri == "s3://ophelian-aws-test/team-a/runs/run-1/checkpoints/trainer"


def test_provider_train_checkpoint_uri_none_for_local_store(tmp_path: Path) -> None:
    """In-process executor uses local paths; no S3 round trip needed."""
    from ophelian.stores.local import LocalArtifactStore

    provider = AWSProvider(
        config=AWSConfig(region="us-east-1", artifact_bucket="x"),
        driver=LocalDriver(),
        store=LocalArtifactStore(str(tmp_path)),
    )
    assert provider._train_checkpoint_uri("run-1", "trainer") is None


def test_step_request_resume_from_threads_through_user_data() -> None:
    """User-data must export ``OPHELIAN_RESUME_FROM`` so the worker's
    ``step_runner`` can pull the checkpoint snapshot down before the
    Train handler runs."""
    request = StepRequest(
        run_id="run-resume",
        pipeline_name="p",
        step_name="trainer",
        kind="train",
        node=Train(
            name="trainer",
            framework="sklearn",
            model="sklearn.linear_model.LogisticRegression",
            data="ds",
        ),
        resume_from="s3://b/team/runs/run-resume/checkpoints/trainer",
    )
    spec = _encode_step_spec(request)
    import base64

    decoded = json.loads(base64.b64decode(spec))
    assert decoded["resume_from"] == "s3://b/team/runs/run-resume/checkpoints/trainer"


def test_step_runner_uploads_checkpoint_on_sigterm(
    aws_env: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SIGTERM (sent ~2 minutes before spot reclamation) must snapshot
    ``/work/checkpoint`` to S3 so the next attempt can resume."""
    from ophelian.runtime import step_runner

    bucket = "ophelian-resume-test"
    boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=bucket)
    work = tmp_path / "work"
    (work / "checkpoint").mkdir(parents=True)
    (work / "checkpoint" / "epoch-3.pt").write_bytes(b"weights")

    uri = step_runner._upload_checkpoint_to_s3(
        work / "checkpoint",
        bucket=bucket,
        prefix="team",
        run_id="run-9",
        step_name="trainer",
    )
    assert uri is not None
    assert uri.rstrip("/") == f"s3://{bucket}/team/runs/run-9/checkpoints/trainer"
    objs = boto3.client("s3", region_name="us-east-1").list_objects_v2(
        Bucket=bucket, Prefix="team/runs/run-9/checkpoints/trainer/"
    )
    assert any(
        o["Key"].endswith("epoch-3.pt") for o in objs.get("Contents", [])
    )


def test_step_runner_resumes_train_from_local_uri(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``OPHELIAN_RESUME_FROM`` (or the encoded ``resume_from`` in the
    spec) must be wired into ``_handle_train`` as the ``resume_from``
    keyword so the adapter sees the previous attempt's snapshot."""
    from ophelian.providers import standalone as _standalone
    from ophelian.runtime import step_runner

    work = tmp_path / "work"
    work.mkdir()
    resume_dir = tmp_path / "snapshot"
    resume_dir.mkdir()
    (resume_dir / "model.pt").write_bytes(b"prev")

    spec = {
        "kind": "train",
        "node": {
            "name": "trainer",
            "framework": "sklearn",
            "model": "sklearn.linear_model.LogisticRegression",
            "data": "ds",
            "hyperparameters": {"max_iter": 50},
        },
        "artifacts": {},
        "run_id": "run-r",
        "resume_from": str(resume_dir),
    }
    spec_path = work / "step.json"
    spec_path.write_text(json.dumps(spec))

    captured: dict[str, Any] = {}
    real_handle_train = _standalone.StandaloneProvider._handle_train

    def _spy(self, node, step_dir, artifacts, *, resume_from=None):  # type: ignore[no-untyped-def]
        captured["resume_from"] = resume_from
        # No upstream artifact in this minimal spec — emit a dummy so
        # _handle_train can run without falling over upstream lookups.
        from ophelian.core.nodes import StepResult

        return StepResult(name=node.name, kind="train", status="success",
                           metrics={}, artifacts={}, info={})

    monkeypatch.setattr(
        _standalone.StandaloneProvider, "_handle_train", _spy
    )
    try:
        rc = step_runner.run(spec_path)
    finally:
        monkeypatch.setattr(
            _standalone.StandaloneProvider, "_handle_train", real_handle_train
        )
    assert rc == 0
    assert captured["resume_from"] is not None
    assert Path(captured["resume_from"]).name == "snapshot"


def test_ec2_user_data_for_deploy_downloads_model_from_s3() -> None:
    """Round-6 regression: ``step_runner`` rewrites ``artifacts.model``
    to an ``s3://...`` URI before writing ``result.json`` (see
    ``_persist_artifacts_to_s3``). The deploy worker's user-data must
    therefore (a) sync that prefix to a local path and (b) point
    ``OPHELIAN_MODEL_PATH`` at the local file/dir — handing
    ``app_from_env`` a raw S3 URI would crash inside ``Path(...)``.
    """
    from ophelian.providers.aws_drivers import _user_data_script

    config = AWSConfig(
        region="us-east-1",
        instance="t3.medium",
        artifact_bucket="ophelian-deploy-bucket",
        artifact_prefix="team-a",
    )
    request = StepRequest(
        run_id="run-deploy",
        pipeline_name="p",
        step_name="serve",
        kind="deploy",
        node=Deploy(name="serve", model="trainer", port=8080, replicas=1),
    )
    script = _user_data_script(config, request)
    assert "aws s3 sync" in script
    assert "/work/model" in script
    assert "OPHELIAN_MODEL_PATH=" in script
    assert "OPHELIAN_FRAMEWORK=" in script
    assert "uvicorn --factory --host 0.0.0.0 --port 8080" in script
    assert "ophelian.runtime.fastapi_runtime:app_from_env" in script


def test_eks_deploy_manifest_threads_framework_and_model_uri_from_upstream(
    aws_env: None, store: S3ArtifactStore,
) -> None:
    """Round-6 regression: ``_serve_framework`` previously inspected
    ``upstream_artifacts`` (which only carries opaque URIs) so it
    *always* defaulted to ``sklearn`` regardless of the trainer's
    framework. Drives both fixes — framework from ``upstream_info``
    and model URI from ``upstream_artifacts`` plumbed via env."""
    driver, _batch, apps, _core = _make_eks_driver(store)
    model_uri = "s3://ophelian-eks/team/runs/run-eks/trainer/artifacts/model"
    request = StepRequest(
        run_id="run-eks",
        pipeline_name="p",
        step_name="serve",
        kind="deploy",
        node=Deploy(name="serve", model="trainer", port=8080, replicas=1),
        upstream_artifacts={"trainer": {"model": model_uri}},
        upstream_info={"trainer": {"framework": "huggingface"}},
    )
    driver.execute(request, store)
    deployment = apps.create_namespaced_deployment.call_args.kwargs["body"]
    env = {
        e["name"]: e["value"]
        for e in deployment["spec"]["template"]["spec"]["containers"][0]["env"]
    }
    assert env["OPHELIAN_FRAMEWORK"] == "huggingface"
    assert env["OPHELIAN_MODEL_URI"] == model_uri
    assert env["AWS_DEFAULT_REGION"] == "us-east-1"
    bootstrap = deployment["spec"]["template"]["spec"]["containers"][0]["args"][0]
    assert "aws s3 sync" in bootstrap
    assert "/work/model" in bootstrap
    assert "OPHELIAN_MODEL_PATH" in bootstrap


def test_aws_provider_auto_bucket_resolved_before_driver_built(
    aws_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Round-6 regression: when the user constructs ``AWS(region=...,
    instance=..., spot=True)`` without an explicit ``artifact_bucket``,
    ``_build_default_store`` derives one from ``sts:GetCallerIdentity``
    and writes it back into ``self._config``. The driver was being
    built *before* the store, so it captured a ``None`` bucket and
    every spawned EC2 instance got an empty
    ``OPHELIAN_ARTIFACT_BUCKET`` — result polling then failed.
    """
    config = AWSConfig(
        region="us-east-1",
        instance="t3.medium",
        ami="ami-12345678",
        spot=True,
    )
    provider = AWSProvider(config=config)
    assert provider._config.artifact_bucket
    # Driver must observe the same resolved bucket — otherwise the
    # user-data script's ``OPHELIAN_ARTIFACT_BUCKET=`` would be empty.
    assert provider._driver._config.artifact_bucket == provider._config.artifact_bucket
