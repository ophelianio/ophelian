"""ResNet image classification on AWS GPU instances.

Run with::

    OPHELIAN_AWS_REGION=us-east-1 \
        OPHELIAN_AWS_BUCKET=my-ophelian-bucket \
        python examples/aws/pytorch_resnet.py

Defaults to a ``g4dn.xlarge`` (single NVIDIA T4) — change
``OPHELIAN_AWS_INSTANCE`` for ``g5.xlarge`` (A10G) or ``p4d.24xlarge`` for
multi-GPU. The pipeline expects a JSONL manifest at
``OPHELIAN_AWS_DATASET`` whose rows look like::

    {"image": "s3://my-bucket/imgs/000001.jpg", "label": 17}

(A plain ``image-folder`` loader is on the v0.6 roadmap; for now the
pytorch adapter resolves the image URIs lazily during training.)

Spot is highly recommended for vision workloads — the resume mechanism
makes interruptions cheap. Re-run with
``OPHELIAN_AWS_RESUME_RUN_ID=...`` if the worker dies mid-training.
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
DATASET = os.environ.get(
    "OPHELIAN_AWS_DATASET",
    f"s3://{BUCKET}/datasets/imagenet/manifest.jsonl",
)


pipe = Pipeline(
    [
        Data(
            name="ds",
            source=DATASET,
            format="jsonl",
            split="train",
            options={"target": "label"},
        ),
        Train(
            name="trainer",
            framework="pytorch",
            model="torchvision.models.resnet18",
            data="ds",
            hyperparameters={"num_classes": 1000, "lr": 1e-3, "momentum": 0.9},
            epochs=20,
            batch_size=128,
        ),
        Eval(name="ev", model="trainer", data="ds", metrics=("accuracy",)),
        Deploy(name="serve", model="trainer", port=8080, replicas=2),
    ],
    name="pytorch-resnet-aws",
)


def main() -> None:
    if BUCKET == _PLACEHOLDER_BUCKET:
        raise SystemExit(
            "Set OPHELIAN_AWS_BUCKET to your S3 artifact bucket before running "
            "this example against AWS (see the module docstring)."
        )
    env = AWS(
        region=REGION,
        instance=os.environ.get("OPHELIAN_AWS_INSTANCE", "g4dn.xlarge"),
        spot=os.environ.get("OPHELIAN_AWS_SPOT", "1") == "1",
        artifact_bucket=BUCKET,
        iam_role=os.environ.get("OPHELIAN_AWS_INSTANCE_PROFILE"),
        runtime_extras=("pytorch",),
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
