# AWS provider

Ophelian's `AWS` env runs the same declarative pipelines you wrote against
`Standalone(local=True)` on real EC2 (or EKS) workers, with artifacts
persisted to S3 and automatic resume on spot interruption.

```python
from ophelian import AWS, Pipeline, Data, Train, Eval

env = AWS(
    region="us-east-1",
    instance="g4dn.xlarge",
    spot=True,
    artifact_bucket="my-ophelian-bucket",
)

result = Pipeline([...]).run(env=env)
```

This document covers credentials, IAM, costs, the spot/resume protocol
and the optional EKS backend.

## Install

```bash
pip install 'ophelian[aws]'        # boto3 + paramiko
pip install 'ophelian[aws,eks]'    # also pull in the kubernetes client
```

## Credentials

`AWS()` does **not** take API keys directly. boto3 resolves credentials
from its standard chain, in order:

1. Environment variables: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`,
   optionally `AWS_SESSION_TOKEN`.
2. Shared credentials file (`~/.aws/credentials`) and the active profile
   (`AWS_PROFILE`).
3. The IAM role attached to the workstation / CI runner (instance
   profile, ECS task role, EKS service account, etc.).

Pick whichever is most convenient for your environment. The env var
approach is the most CI-friendly:

```bash
export AWS_ACCESS_KEY_ID=AKIA...
export AWS_SECRET_ACCESS_KEY=...
export AWS_DEFAULT_REGION=us-east-1
```

You can pin the profile or region per-call with
`AWS(profile="research", region="eu-west-1", ...)`.

## IAM — minimum policy

The identity that *invokes* `pipeline.run(env=AWS(...))` needs to launch
EC2, manage S3 artifacts, and pass a worker role to the instance:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "EC2Manage",
      "Effect": "Allow",
      "Action": [
        "ec2:RunInstances",
        "ec2:TerminateInstances",
        "ec2:DescribeInstances",
        "ec2:DescribeInstanceStatus",
        "ec2:CreateTags"
      ],
      "Resource": "*"
    },
    {
      "Sid": "PassWorkerRole",
      "Effect": "Allow",
      "Action": "iam:PassRole",
      "Resource": "arn:aws:iam::<account>:role/ophelian-worker"
    },
    {
      "Sid": "ArtifactBucket",
      "Effect": "Allow",
      "Action": [
        "s3:CreateBucket",
        "s3:ListBucket"
      ],
      "Resource": "arn:aws:s3:::my-ophelian-bucket"
    },
    {
      "Sid": "ArtifactObjects",
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"],
      "Resource": "arn:aws:s3:::my-ophelian-bucket/*"
    }
  ]
}
```

The `ophelian-worker` role attached to the EC2 worker only needs S3
read/write on the artifact bucket — it does **not** need EC2 permissions:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket"],
      "Resource": [
        "arn:aws:s3:::my-ophelian-bucket",
        "arn:aws:s3:::my-ophelian-bucket/*"
      ]
    }
  ]
}
```

Pass the worker role's *instance profile* name to `AWS()`:

```python
env = AWS(
    region="us-east-1",
    instance="g5.xlarge",
    artifact_bucket="my-ophelian-bucket",
    iam_role="ophelian-worker",        # name of the instance profile
)
```

## What the provider creates (and tears down)

For every `pipeline.run()`:

* **One EC2 instance per step** (default driver). The worker is launched
  with the configured AMI / instance type, runs the step's `step_runner`
  via user-data, persists `result.json` to S3, then terminates itself.
* **Tagged resources.** Every EC2 instance gets the tags from
  `instance_tags` (default `{"ophelian/managed": "true"}`) plus
  `ophelian/run-id` and `ophelian/step`, so you can audit costs in Cost
  Explorer.
* **`runs/{run_id}/{step}/result.json`** in the artifact bucket — the
  serialized `StepResult` for every completed step.
* **`runs/{run_id}/{step}/artifacts/...`** for any artifacts the step
  emits (model checkpoints, eval reports, deploy descriptors).
* **`checkpoints/{run_id}/checkpoint.json`** when a spot interruption
  fires, so the next run can pick up where this one left off.

The provider always terminates the EC2 instance in a `finally` block,
even on Python exceptions, so a crashed driver shouldn't leak hardware.

## Spot + resume

```python
env = AWS(region="us-east-1", instance="g5.xlarge", spot=True, artifact_bucket="...")
result = pipeline.run(env=env)

if not result.succeeded:
    failed = next(s for s in result.steps if s.status == "failed")
    if failed.info.get("resumable"):
        # Re-run with the same run_id — completed steps are skipped.
        env_resume = env.with_resume(run_id=failed.info["run_id"])
        result = pipeline.run(env=env_resume)
```

What happens under the hood:

1. The driver polls the EC2 instance metadata service for spot
   interruption notices (`/latest/meta-data/spot/instance-action`).
2. On interruption, the in-flight step raises `SpotInterruption`. The
   provider writes a checkpoint to `s3://.../checkpoints/{run_id}/`
   recording which steps already finished and what their results were.
3. The current step is reported as `status="failed"` with
   `info={"resumable": True, "run_id": "...", "interrupted_step": "..."}`.
4. Resuming with `with_resume(run_id=...)` (or
   `AWS(..., resume_run_id=...)`) reuses the same run id, restores the
   completed step results from the checkpoint, and only re-runs the
   interrupted step plus everything downstream.

Spot is opt-in; on-demand is the default. EKS + spot is not supported
yet — the AWS env will refuse the combination at construction time.

## Optional EKS backend

```python
env = AWS(
    region="us-west-2",
    instance="g4dn.xlarge",
    artifact_bucket="my-ophelian-bucket",
    backend="eks",
    eks_cluster="research-prod",
    eks_namespace="ophelian",
)
```

EKS mode replaces per-step EC2 instances with per-step Kubernetes Jobs.
The same artifact contract applies (results and artifacts in S3), so you
can mix-and-match EKS + EC2 across runs without changing the pipeline
code. You'll need to install the cluster-side controller (out of scope
for this doc; see the platform team's runbook).

## Cost guidance

Order-of-magnitude pricing for the three reference workloads in
`examples/aws/` (us-east-1, on-demand, end of 2026):

| Workload | Instance | $/hour | Typical run | $ per run |
| --- | --- | --- | --- | --- |
| XGBoost tabular | `m5.large` | $0.10 | 5–10 min | < $0.05 |
| ResNet image classification | `g4dn.xlarge` | $0.53 | ~2 hours | ~$1.10 |
| HuggingFace LoRA fine-tune | `g5.2xlarge` | $1.21 | ~6 hours | ~$7.30 |

Spot pricing is typically 60–80% cheaper. S3 storage is negligible for
typical artifact sizes (< $0.05/month per run).

## Troubleshooting

* **`pydantic.ValidationError: artifact_bucket required`** — the AWS env
  always needs an S3 bucket; create one with `aws s3 mb`.
* **`AccessDenied` on `s3:PutObject`** — the *worker* IAM role is
  missing `s3:PutObject` on the bucket. Re-check the worker policy
  above; the calling identity's permissions are not enough on their
  own.
* **Workers stuck in `pending`** — usually a missing IAM instance
  profile or an AMI/region mismatch. Override with
  `AWS(..., ami="ami-...")` for a known-good Deep Learning AMI.
* **Spot kept interrupting** — switch to a less-contested instance
  family or fall back to on-demand with `spot=False`.
