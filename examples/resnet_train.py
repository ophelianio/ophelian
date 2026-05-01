"""Train a ResNet-50 image classifier on the cheapest available GPU.

Demonstrates a full pipeline against a parquet image dataset, with
checkpointing and spot/preemptible interruption recovery handled
transparently by the chosen provider. By default targets a T4 (the
cheapest current-gen GPU across all three clouds) and falls back to
on-demand if no spot quote is available.

::

    OPHELIAN_DATASET=s3://my-bucket/imagenette/train.parquet \
        python examples/resnet_train.py
"""

from __future__ import annotations

import os

from ophelian import Auto, Data, Eval, Pipeline, Train

DATASET = os.environ.get("OPHELIAN_DATASET", "s3://ophelian-demo/imagenette/train.parquet")
EVAL_DATASET = os.environ.get("OPHELIAN_EVAL_DATASET", "s3://ophelian-demo/imagenette/val.parquet")


pipe = Pipeline(
    [
        Data(name="train_ds", source=DATASET, format="parquet", split="train"),
        Data(name="val_ds", source=EVAL_DATASET, format="parquet", split="val"),
        Train(
            name="resnet",
            framework="pytorch",
            model="resnet50",
            data="train_ds",
            hyperparameters={
                "lr": 0.05,
                "momentum": 0.9,
                "weight_decay": 1e-4,
                "image_size": 224,
            },
            epochs=int(os.environ.get("OPHELIAN_EPOCHS", "10")),
            batch_size=int(os.environ.get("OPHELIAN_BATCH", "128")),
        ),
        Eval(
            name="ev",
            model="resnet",
            data="val_ds",
            metrics=("accuracy", "top5_accuracy"),
        ),
    ],
    name="resnet-train",
)


def main() -> None:
    env = Auto(
        cheapest_gpu=os.environ.get("OPHELIAN_GPU", "T4"),
        spot=os.environ.get("OPHELIAN_SPOT", "1") == "1",
        dry_run=os.environ.get("OPHELIAN_DRY_RUN", "0") == "1",
        require_credentials=os.environ.get("OPHELIAN_REQUIRE_CREDS", "1") == "1",
    )
    result = pipe.run(env=env)
    if result.succeeded:
        metrics = result.step("ev").metrics
        print("Eval metrics:", metrics)
    else:
        for step in result.steps:
            print(f"{step.name}: {step.status} {step.error or ''}")


if __name__ == "__main__":
    main()
