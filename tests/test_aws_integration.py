"""Opt-in integration tests against a *real* AWS account.

These tests are skipped unless ``OPHELIAN_AWS_INTEGRATION_TESTS=1`` is set
in the environment. They cost real money — by default they target the
``t3.medium`` family for the EC2 worker (~$0.04/hour on us-east-1) and a
single S3 bucket (~$0.02 for the round trip). Override the instance/region
with ``OPHELIAN_AWS_INTEGRATION_INSTANCE`` and ``OPHELIAN_AWS_REGION``.

Estimated cost per full run (one workload): **< $0.10 USD**.

The tests assume the executing identity has:

* ``ec2:RunInstances``, ``ec2:TerminateInstances``, ``ec2:DescribeInstances``
* ``s3:CreateBucket``, ``s3:PutObject``, ``s3:GetObject``, ``s3:ListBucket``,
  ``s3:DeleteObject``
* an IAM instance profile that grants the worker S3 read/write on the
  artifact bucket (``OPHELIAN_AWS_INSTANCE_PROFILE``).

See ``docs/aws.md`` for the full IAM policy.
"""

from __future__ import annotations

import os
import uuid

import pytest
from ophelian import AWS, Data, Eval, Pipeline, Train

ENABLED = os.environ.get("OPHELIAN_AWS_INTEGRATION_TESTS") == "1"

pytestmark = [
    pytest.mark.aws_integration,
    pytest.mark.skipif(not ENABLED, reason="OPHELIAN_AWS_INTEGRATION_TESTS not set to 1"),
]


def _bucket() -> str:
    return os.environ.get(
        "OPHELIAN_AWS_INTEGRATION_BUCKET",
        f"ophelian-it-{uuid.uuid4().hex[:8]}",
    )


def _region() -> str:
    return os.environ.get("OPHELIAN_AWS_REGION", "us-east-1")


def _instance() -> str:
    return os.environ.get("OPHELIAN_AWS_INTEGRATION_INSTANCE", "t3.medium")


def test_real_aws_runs_minimal_pipeline() -> None:
    """Smoke: spin up a t3.medium, train sklearn, write artifacts to S3."""
    env = AWS(
        region=_region(),
        instance=_instance(),
        spot=False,
        artifact_bucket=_bucket(),
        iam_role=os.environ.get("OPHELIAN_AWS_INSTANCE_PROFILE"),
        timeout_seconds=15 * 60,
    )

    pipeline = Pipeline(
        [
            Data(
                name="ds",
                source="memory://toy",
                format="inline",
                options={"X": [[0.0], [1.0]], "y": [0, 1]},
            ),
            Train(
                name="trainer",
                framework="sklearn",
                model="sklearn.linear_model.LogisticRegression",
                data="ds",
            ),
            Eval(name="ev", model="trainer", data="ds", metrics=("accuracy",)),
        ],
        name="aws-it",
    )

    result = pipeline.run(env=env)
    assert result.succeeded, [s.error for s in result.steps if s.error]
    assert result.step("ev").metrics["accuracy"] >= 0.5
