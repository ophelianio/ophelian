"""Minimal sklearn quickstart — train + deploy a logistic regression locally."""

from __future__ import annotations

from ophelian import Data, Deploy, Pipeline, Standalone, Train

pipe = Pipeline(
    [
        Data(name="ds", source="synthetic://iris", format="synthetic", options={"name": "iris"}),
        Train(
            name="trainer",
            framework="sklearn",
            model="sklearn.linear_model.LogisticRegression",
            data="ds",
            hyperparameters={"max_iter": 200},
        ),
        Deploy(name="serve", model="trainer", port=8080),
    ],
    name="sklearn-quickstart",
)


if __name__ == "__main__":
    result = pipe.run(env=Standalone(local=True))
    if result.succeeded:
        info = result.step("serve").info
        print(f"Predict endpoint: {info['predict']}")
        print(f"Health endpoint:  {info['health']}")
    else:
        for step in result.steps:
            print(f"{step.name}: {step.status} {step.error or ''}")
