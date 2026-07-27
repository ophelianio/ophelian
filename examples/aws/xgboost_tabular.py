"""XGBoost on a tabular dataset, run on AWS EC2 with S3-backed artifacts.

Run with::

    OPHELIAN_AWS_REGION=us-east-1 \
        OPHELIAN_AWS_BUCKET=my-ophelian-bucket \
        python examples/aws/xgboost_tabular.py

Requires:

* the optional extras: ``pip install 'ophelian[xgboost,aws]'``
* AWS credentials reachable through the standard boto3 chain (env vars,
  shared credentials file, or an IAM role attached to the workstation)
* an IAM instance profile that the EC2 worker can assume to read/write the
  artifact bucket (its name goes in ``OPHELIAN_AWS_INSTANCE_PROFILE``)

Switch to ``spot=True`` once you've validated the run; if the spot worker
is reclaimed, re-run the same command with
``OPHELIAN_AWS_RESUME_RUN_ID=run-...`` to pick up from the last
checkpoint.
"""

from __future__ import annotations

import os

from ophelian import AWS, Data, Deploy, Eval, Pipeline, Train

# Read config at import time with a placeholder default so the module
# imports (and `ophelian dry-run` can compile the plan) without any AWS
# env vars set. `main()` refuses to run for real until a real bucket is
# provided.
_PLACEHOLDER_BUCKET = "your-ophelian-bucket"
REGION = os.environ.get("OPHELIAN_AWS_REGION", "us-east-1")
BUCKET = os.environ.get("OPHELIAN_AWS_BUCKET", _PLACEHOLDER_BUCKET)
DATA_URI = os.environ.get(
    "OPHELIAN_AWS_DATASET",
    f"s3://{BUCKET}/datasets/tabular/train.jsonl",
)


pipe = Pipeline(
    [
        Data(name="ds", source=DATA_URI, format="jsonl", options={"target": "y"}),
        Train(
            name="booster",
            framework="xgboost",
            model="xgboost.XGBClassifier",
            data="ds",
            hyperparameters={"max_depth": 6, "n_estimators": 100, "learning_rate": 0.1},
        ),
        Eval(name="ev", model="booster", data="ds", metrics=("accuracy",)),
        Deploy(name="serve", model="booster", port=8080, replicas=1),
    ],
    name="xgboost-tabular-aws",
)


def main() -> None:
    if BUCKET == _PLACEHOLDER_BUCKET:
        raise SystemExit(
            "Set OPHELIAN_AWS_BUCKET to your S3 artifact bucket before running "
            "this example against AWS (see the module docstring)."
        )
    env = AWS(
        region=REGION,
        instance=os.environ.get("OPHELIAN_AWS_INSTANCE", "m5.large"),
        spot=os.environ.get("OPHELIAN_AWS_SPOT", "0") == "1",
        artifact_bucket=BUCKET,
        iam_role=os.environ.get("OPHELIAN_AWS_INSTANCE_PROFILE"),
        runtime_extras=("xgboost",),
        resume_run_id=os.environ.get("OPHELIAN_AWS_RESUME_RUN_ID") or None,
    )
    result = pipe.run(env=env)
    if result.succeeded:
        info = result.step("serve").info
        print("Endpoint:", info.get("endpoint_url") or info)
    else:
        for step in result.steps:
            print(f"{step.name}: {step.status} {step.error or ''}")


if __name__ == "__main__":
    main()
