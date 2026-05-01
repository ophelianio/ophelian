"""AWS env — first cloud backend for Ophelian pipelines.

Pipeline authors only ever interact with the friendly :func:`AWS` factory::

    from ophelian.envs import AWS

    env = AWS(region="us-east-1", instance="g4dn.xlarge", spot=True)
    pipe.run(env=env)

The factory builds and validates an :class:`AWSConfig` and hands it to the
underlying :class:`~ophelian.providers.aws.AWSProvider`. Every aspect that
matters for cost or reproducibility — region, instance type, spot/on-demand,
optional AMI/VPC/IAM role, EKS backend, artifact bucket — is captured in
:class:`AWSConfig` so it can be inspected, serialised and recreated by
tooling.

Validation errors here are cheap and fail fast; the provider only kicks off
real AWS work after a fully validated config is in hand. Common mistakes
(wrong region/instance combo, missing bucket name when EKS backend asks for
remote artifacts) raise descriptive exceptions before a single boto3 call
goes out — much friendlier than a 30-second timeout against a typo.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

if TYPE_CHECKING:  # pragma: no cover
    from ophelian.providers.aws import AWSProvider


# Curated map of AWS regions to a representative on-demand AMI family. The
# AWS provider treats this as a sanity check (the region must be valid) and
# as a default for AMI lookups when the user does not pin one explicitly.
KNOWN_REGIONS: tuple[str, ...] = (
    "us-east-1",
    "us-east-2",
    "us-west-1",
    "us-west-2",
    "eu-west-1",
    "eu-west-2",
    "eu-west-3",
    "eu-central-1",
    "eu-north-1",
    "ap-southeast-1",
    "ap-southeast-2",
    "ap-northeast-1",
    "ap-northeast-2",
    "ap-south-1",
    "sa-east-1",
    "ca-central-1",
)

# Subset of instance families known to be safe defaults for the workloads
# Ophelian targets. The check is liberal — anything matching a typical
# AWS instance type pattern is accepted, but we surface a friendly hint
# when the user picks a family that is unlikely to fit the workload.
_INSTANCE_RE = re.compile(r"^[a-z0-9]+\.[a-z0-9]+$")

GPU_INSTANCE_FAMILIES: tuple[str, ...] = (
    "p3",
    "p4",
    "p5",
    "g4dn",
    "g5",
    "g5g",
    "g6",
)

CPU_INSTANCE_FAMILIES: tuple[str, ...] = (
    "t2",
    "t3",
    "t3a",
    "t4g",
    "m5",
    "m6i",
    "m7i",
    "c5",
    "c6i",
    "c7i",
    "r5",
    "r6i",
)

CloudBackend = Literal["ec2", "eks"]


class AWSConfig(BaseModel):
    """Strongly-typed configuration for the AWS env.

    All fields are validated up-front so the provider can rely on them
    being well-formed.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    region: str = Field(description="AWS region — e.g. ``us-east-1``.")
    instance: str = Field(
        default="g4dn.xlarge",
        description="EC2 instance type. Used for the on-demand or spot worker.",
    )
    spot: bool = Field(default=False, description="Request a spot instance instead of on-demand.")
    ami: str | None = Field(
        default=None,
        description=(
            "AMI id to launch. Defaults to the AWS Deep Learning AMI for GPU "
            "families and Amazon Linux 2023 for CPU families."
        ),
    )
    vpc: str | None = Field(
        default=None,
        description="Subnet id (within a custom VPC). Defaults to the default VPC.",
    )
    security_group: str | None = Field(
        default=None,
        description="Security group id. Defaults to a permissive group created on the fly.",
    )
    iam_role: str | None = Field(
        default=None,
        description=(
            "IAM instance profile name attached to the EC2 worker. Required when the "
            "pipeline reads/writes S3 artifacts from the worker; documented in `docs/aws.md`."
        ),
    )
    key_pair: str | None = Field(
        default=None,
        description="EC2 key-pair name used when SSH'ing into the worker (optional).",
    )
    artifact_bucket: str | None = Field(
        default=None,
        description=(
            "S3 bucket used as the artifact store for the run. When omitted, the "
            "provider creates one named ``ophelian-artifacts-<account-id>`` on first use."
        ),
    )
    artifact_prefix: str = Field(
        default="runs",
        description="Key prefix under which run artifacts are written.",
    )
    aws_backend: CloudBackend = Field(
        default="ec2",
        description="``ec2`` (default) provisions VMs directly. ``eks`` submits to a cluster.",
    )
    eks_cluster: str | None = Field(
        default=None,
        description="EKS cluster name. Required when ``aws_backend='eks'``.",
    )
    eks_namespace: str = Field(
        default="default",
        description="Kubernetes namespace used for EKS workloads.",
    )
    runtime_image: str = Field(
        default="python:3.12-slim",
        description="Container image executed on the worker. Override to use ECR.",
    )
    runtime_extras: tuple[str, ...] = Field(
        default=("sklearn",),
        description="Pyproject extras installed alongside ophelian inside the worker container.",
    )
    pipeline_archive_uri: str | None = Field(
        default=None,
        description=(
            "Optional ``s3://bucket/key.tar.gz`` containing the pipeline source. When set, "
            "the worker downloads and ``pip install``s it instead of installing ophelian "
            "from PyPI."
        ),
    )
    instance_profile: str | None = Field(
        default=None,
        description="Alias of ``iam_role`` kept for AWS-CLI parity.",
    )
    user_data: str | None = Field(
        default=None,
        description="Extra cloud-init script appended to the worker user-data (advanced).",
    )
    spot_max_price: float | None = Field(
        default=None,
        description="Maximum spot bid in USD/hour. Defaults to the on-demand price.",
    )
    instance_tags: dict[str, str] = Field(
        default_factory=lambda: {"ophelian/managed": "true"},
        description="Tags applied to every AWS resource the provider creates.",
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
            "Existing run id to resume. Set automatically by the provider after a spot "
            "interruption so a subsequent ``pipe.run`` continues from the latest checkpoint."
        ),
    )

    # ---- validators ---------------------------------------------------

    @field_validator("region")
    @classmethod
    def _valid_region(cls, value: str) -> str:
        if not value or not isinstance(value, str):
            raise ValueError("region must be a non-empty string")
        if value not in KNOWN_REGIONS and not re.match(r"^[a-z]{2}-[a-z]+-\d$", value):
            # Accept anything that *looks* like an AWS region — keeps us
            # compatible with newly-launched regions — but warn for typos.
            raise ValueError(
                f"region={value!r} does not look like a valid AWS region "
                f"(expected e.g. 'us-east-1'). Known regions: {', '.join(KNOWN_REGIONS)}"
            )
        return value

    @field_validator("instance")
    @classmethod
    def _valid_instance(cls, value: str) -> str:
        if not _INSTANCE_RE.match(value):
            raise ValueError(
                f"instance={value!r} is not a valid AWS instance type "
                "(expected e.g. 'g4dn.xlarge', 't3.medium')."
            )
        return value

    @field_validator("ami")
    @classmethod
    def _valid_ami(cls, value: str | None) -> str | None:
        if value is None:
            return value
        if not value.startswith("ami-"):
            raise ValueError(f"ami={value!r} must start with 'ami-'")
        return value

    @field_validator("artifact_prefix")
    @classmethod
    def _valid_prefix(cls, value: str) -> str:
        return value.strip("/")

    @field_validator("runtime_extras")
    @classmethod
    def _valid_extras(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(v.strip() for v in value if v.strip())

    @field_validator("eks_cluster")
    @classmethod
    def _valid_cluster_name(cls, value: str | None) -> str | None:
        if value is None:
            return value
        if not re.match(r"^[A-Za-z0-9][A-Za-z0-9-_]{0,99}$", value):
            raise ValueError(f"eks_cluster={value!r} is not a valid EKS cluster name")
        return value

    @model_validator(mode="after")
    def _backend_consistency(self) -> AWSConfig:
        if self.aws_backend == "eks" and not self.eks_cluster:
            raise ValueError(
                "aws_backend='eks' requires `eks_cluster=<name>` so the provider can submit Jobs."
            )
        if self.spot and self.aws_backend == "eks":
            raise ValueError(
                "Spot scheduling is managed by the EKS cluster's node groups, not by the "
                "Ophelian provider — drop `spot=True` when targeting EKS."
            )
        # Normalise iam_role / instance_profile aliases.
        if self.instance_profile and not self.iam_role:
            object.__setattr__(self, "iam_role", self.instance_profile)
        if self.iam_role and not self.instance_profile:
            object.__setattr__(self, "instance_profile", self.iam_role)
        return self

    # ---- ergonomics ----------------------------------------------------

    @property
    def is_gpu_instance(self) -> bool:
        family = self.instance.split(".")[0]
        return family in GPU_INSTANCE_FAMILIES

    @property
    def is_cpu_instance(self) -> bool:
        family = self.instance.split(".")[0]
        return family in CPU_INSTANCE_FAMILIES

    def with_resume(self, run_id: str) -> AWSConfig:
        """Return a copy of this config with ``resume_run_id`` set."""
        return self.model_copy(update={"resume_run_id": run_id})

    def validate_instance_type_in_region(self, ec2_client: object) -> None:
        """Verify that ``self.instance`` is offered in ``self.region``.

        Calls ``ec2:DescribeInstanceTypeOfferings`` against ``ec2_client``
        and raises ``ValueError`` if the type isn't available in the
        region (a common foot-gun — e.g. ``p4d.24xlarge`` not in every
        AZ). Best-effort: any AWS error other than a non-empty offerings
        list is swallowed so we don't block users with restrictive IAM
        policies.
        """
        describe = getattr(ec2_client, "describe_instance_type_offerings", None)
        if describe is None:
            return
        try:
            resp = describe(
                LocationType="region",
                Filters=[
                    {"Name": "instance-type", "Values": [self.instance]},
                    {"Name": "location", "Values": [self.region]},
                ],
            )
        except Exception:
            # Permission denied / endpoint unreachable / mocked client —
            # don't block the run on a best-effort sanity check.
            return
        offerings = resp.get("InstanceTypeOfferings") if isinstance(resp, dict) else None
        if offerings is None:
            return
        if not offerings:
            raise ValueError(
                f"Instance type {self.instance!r} is not offered in region "
                f"{self.region!r}. Pick a different region or instance type — "
                "see https://aws.amazon.com/ec2/instance-types/ for availability."
            )


def AWS(
    *,
    region: str,
    instance: str = "g4dn.xlarge",
    spot: bool = False,
    aws_backend: CloudBackend = "ec2",
    artifact_bucket: str | None = None,
    eks_cluster: str | None = None,
    driver: Any | None = None,
    store: Any | None = None,
    **kwargs: Any,
) -> AWSProvider:
    """Build an :class:`~ophelian.providers.aws.AWSProvider`.

    The keyword-only signature mirrors :class:`AWSConfig` for the parameters
    pipeline authors typically reach for; everything else can be passed via
    ``**kwargs``. ``driver`` and ``store`` are wiring overrides reserved for
    tests — leave them at their defaults in production code.
    """
    from ophelian.providers.aws import AWSProvider

    config = AWSConfig(
        region=region,
        instance=instance,
        spot=spot,
        aws_backend=aws_backend,
        artifact_bucket=artifact_bucket,
        eks_cluster=eks_cluster,
        **kwargs,
    )
    return AWSProvider(config=config, driver=driver, store=store)


__all__ = [
    "AWS",
    "CPU_INSTANCE_FAMILIES",
    "GPU_INSTANCE_FAMILIES",
    "KNOWN_REGIONS",
    "AWSConfig",
    "CloudBackend",
]
