"""Tests for the graph compiler."""

from __future__ import annotations

import pytest
from ophelian import Data, Deploy, Eval, Pipeline, Train
from ophelian.core import GraphCompiler


def _build_pipeline() -> Pipeline:
    return Pipeline(
        [
            Data(name="ds", source="file://train.parquet"),
            Train(
                name="trainer",
                framework="sklearn",
                model="LogisticRegression",
                data="ds",
            ),
            Eval(name="ev", model="trainer", data="ds", metrics=("accuracy",)),
            Deploy(name="serve", model="trainer", port=8080),
        ],
        name="pipeline",
    )


def test_compile_orders_topologically() -> None:
    plan = GraphCompiler().compile(_build_pipeline())
    names = [step.name for step in plan.steps]
    assert names.index("ds") < names.index("trainer")
    assert names.index("trainer") < names.index("ev")
    assert names.index("trainer") < names.index("serve")


def test_compile_assigns_sequential_orders() -> None:
    plan = GraphCompiler().compile(_build_pipeline())
    assert [s.order for s in plan.steps] == list(range(len(plan.steps)))


def test_compile_detects_cycle() -> None:
    a = Data(name="a", source="file://x")
    # Manually fabricate a cycle: depends_on b which depends on a.
    b = Data(name="b", source="file://y", depends_on=("a",))
    object.__setattr__(a, "depends_on", ("b",))
    pipe = Pipeline.model_construct(name="bad", steps=(a, b), description=None)
    with pytest.raises(ValueError, match="Cycle"):
        GraphCompiler().compile(pipe)


def test_dry_run_does_not_raise(capsys: pytest.CaptureFixture[str]) -> None:
    pipe = _build_pipeline()
    pipe.dry_run()
    captured = capsys.readouterr()
    assert "execution plan" in captured.out.lower()
    assert "trainer" in captured.out
