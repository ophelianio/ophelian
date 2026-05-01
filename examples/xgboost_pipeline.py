"""XGBoost quickstart — train a tiny tabular classifier locally.

Uses a small inline dataset so it runs without any network or extra data
files. Install the optional adapter with ``pip install -e '.[xgboost]'``.
"""

from __future__ import annotations

from ophelian import Data, Pipeline, Standalone, Train

pipe = Pipeline(
    [
        Data(
            name="ds",
            source="inline://",
            format="inline",
            options={
                "X": [
                    [0.0, 0.0],
                    [0.1, 0.0],
                    [0.0, 0.1],
                    [0.1, 0.1],
                    [1.0, 1.0],
                    [0.9, 1.0],
                    [1.0, 0.9],
                    [0.9, 0.9],
                ],
                "y": [0, 0, 0, 0, 1, 1, 1, 1],
            },
        ),
        Train(
            name="trainer",
            framework="xgboost",
            model="XGBClassifier",
            data="ds",
            hyperparameters={"max_depth": 3, "n_estimators": 30, "verbosity": 0},
        ),
    ],
    name="xgboost-quickstart",
)


if __name__ == "__main__":
    result = pipe.run(env=Standalone(local=True))
    if result.succeeded:
        print("trained model:", result.step("trainer").info.get("artifact"))
    else:
        for step in result.steps:
            print(f"{step.name}: {step.status} {step.error or ''}")
