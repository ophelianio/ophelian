"""Train XGBoost on tabular data — the no-GPU happy-path demo.

XGBoost is great for the demo because it (a) has zero GPU dependency,
(b) trains in seconds on a small CSV, and (c) showcases that Ophelian
isn't just for deep learning. We still go through Auto so the user
sees the routing log, but with ``cheapest_gpu="T4"`` and a tiny
instance — the actual training runs on CPU and finishes well under
the spot-interruption window.

::

    OPHELIAN_DATASET=s3://my-bucket/customers.csv python examples/xgboost_tabular.py
"""

from __future__ import annotations

import os

from ophelian import Auto, Data, Deploy, Eval, Pipeline, Train

DATASET = os.environ.get("OPHELIAN_DATASET", "s3://ophelian-demo/tabular/train.parquet")
EVAL_DATASET = os.environ.get("OPHELIAN_EVAL_DATASET", "s3://ophelian-demo/tabular/eval.parquet")
TARGET = os.environ.get("OPHELIAN_TARGET", "label")


pipe = Pipeline(
    [
        Data(name="train_ds", source=DATASET, format="parquet", split="train"),
        Data(name="eval_ds", source=EVAL_DATASET, format="parquet", split="eval"),
        Train(
            name="xgb",
            framework="xgboost",
            model="xgb-classifier",
            data="train_ds",
            hyperparameters={
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "max_depth": 8,
                "eta": 0.1,
                "subsample": 0.9,
                "target": TARGET,
            },
            epochs=int(os.environ.get("OPHELIAN_ROUNDS", "200")),
        ),
        Eval(name="ev", model="xgb", data="eval_ds", metrics=("auc", "accuracy")),
        Deploy(name="serve", model="xgb", port=8080, replicas=1),
    ],
    name="xgboost-tabular",
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
        info = result.step("serve").info
        print("Endpoint:", info.get("endpoint_url") or info)
        print("Eval:", result.step("ev").metrics)
    else:
        for step in result.steps:
            print(f"{step.name}: {step.status} {step.error or ''}")


if __name__ == "__main__":
    main()
