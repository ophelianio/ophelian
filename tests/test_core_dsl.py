"""Tests for the declarative DSL."""

from __future__ import annotations

import pytest
from ophelian import Data, Deploy, Eval, Pipeline, Train, Tune
from pydantic import ValidationError


def test_train_auto_wires_data_dependency() -> None:
    train = Train(name="trainer", framework="sklearn", model="LogisticRegression", data="ds")
    assert "ds" in train.depends_on


def test_pipeline_rejects_unknown_dependency() -> None:
    train = Train(name="trainer", framework="sklearn", model="LogisticRegression", data="missing")
    with pytest.raises(ValidationError):
        Pipeline([train])


def test_pipeline_rejects_duplicate_node_names() -> None:
    a = Data(name="ds", source="s3://bucket/x.parquet")
    b = Data(name="ds", source="s3://bucket/y.parquet")
    with pytest.raises(ValidationError):
        Pipeline([a, b])


def test_pipeline_requires_at_least_one_step() -> None:
    with pytest.raises(ValidationError):
        Pipeline([])


def test_tune_wires_train_dependency() -> None:
    tune = Tune(name="hpo", train="trainer", search_space={"lr": [0.1, 0.01]})
    assert tune.depends_on == ("trainer",)


def test_eval_wires_model_and_data_dependencies() -> None:
    evaluator = Eval(name="ev", model="trainer", data="ds", metrics=("acc",))
    assert set(evaluator.depends_on) == {"trainer", "ds"}


def test_deploy_wires_model_dependency() -> None:
    deploy = Deploy(name="serve", model="trainer", port=8080)
    assert deploy.depends_on == ("trainer",)


def test_eval_rejects_empty_metrics() -> None:
    with pytest.raises(ValidationError):
        Eval(name="ev", model="m", data="d", metrics=())


def test_nodes_are_immutable() -> None:
    data = Data(name="ds", source="file://x.csv", format="csv")
    with pytest.raises(ValidationError):
        data.source = "file://other.csv"  # type: ignore[misc]


def test_pipeline_keeps_steps_as_tuple() -> None:
    data = Data(name="ds", source="file://x.csv", format="csv")
    train = Train(name="t", framework="sklearn", model="LogisticRegression", data="ds")
    pipe = Pipeline([data, train])
    assert isinstance(pipe.steps, tuple)
    assert pipe.steps[0].name == "ds"
