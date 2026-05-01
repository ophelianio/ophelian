"""PyTorch quickstart — train a tiny linear regression locally.

This example trains `torch.nn.Linear` on a small inline dataset so it runs
without any network or extra data files. Install the optional adapter with
``pip install -e '.[pytorch]'``.
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
                "X": [[1.0], [2.0], [3.0], [4.0]],
                "y": [2.0, 4.0, 6.0, 8.0],
            },
        ),
        Train(
            name="trainer",
            framework="pytorch",
            model="torch.nn.Linear",
            data="ds",
            hyperparameters={"in_features": 1, "out_features": 1, "lr": 0.05},
            epochs=200,
        ),
    ],
    name="pytorch-quickstart",
)


if __name__ == "__main__":
    result = pipe.run(env=Standalone(local=True))
    if result.succeeded:
        print("trained model:", result.step("trainer").info.get("artifact"))
    else:
        for step in result.steps:
            print(f"{step.name}: {step.status} {step.error or ''}")
