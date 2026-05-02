"""DAG invariant tests for the pipeline compiler.

These tests pin **properties** the compiler must always satisfy
regardless of input, not specific output shapes. A failure here means
some refactor of `GraphCompiler` broke a guarantee the rest of the
framework (and external users) silently relies on:

* compilation is **deterministic** — the same Pipeline object compiles
  to the same plan every time, so retries and cached executions match;
* compilation is **side-effect free** — the user's Pipeline instance is
  never mutated, so a user can call `dry_run()` and `run()` back to
  back without surprise behavior;
* the resulting plan is a **valid topological order** — every step's
  dependencies appear before it;
* cycles are detected loudly with the offending nodes named;
* duplicate node names and dangling references fail at construction,
  not later when the pipeline executes against a real cloud.
"""

from __future__ import annotations

import copy
import json

import pytest
from ophelian import Data, Eval, Pipeline, Train
from ophelian.core.compiler import ExecutionPlan, GraphCompiler
from pydantic import ValidationError


def _toy_pipeline() -> Pipeline:
    """A canonical 3-node DAG: data -> train -> eval."""
    return Pipeline(
        name="toy",
        steps=[
            Data(name="ds", source="inline://", format="inline"),
            Train(
                name="trainer",
                framework="sklearn",
                model="LogisticRegression",
                data="ds",
            ),
            Eval(name="evaluator", model="trainer", data="ds"),
        ],
    )


def _diamond_pipeline() -> Pipeline:
    """A 5-node diamond: ds -> {a, b} -> joiner -> evaluator."""
    return Pipeline(
        name="diamond",
        steps=[
            Data(name="ds", source="inline://", format="inline"),
            Train(name="a", framework="sklearn", model="LogisticRegression", data="ds"),
            Train(name="b", framework="sklearn", model="LogisticRegression", data="ds"),
            Eval(name="evaluator_a", model="a", data="ds"),
            Eval(name="evaluator_b", model="b", data="ds"),
        ],
    )


def _plan_signature(plan: ExecutionPlan) -> str:
    """JSON-stable representation of a plan for equality comparison."""
    return json.dumps(
        [
            {"order": s.order, "name": s.name, "kind": s.kind, "deps": list(s.depends_on)}
            for s in plan.steps
        ],
        sort_keys=True,
    )


def test_compile_is_deterministic_across_repeated_runs() -> None:
    """Compiling the same pipeline N times must produce structurally
    identical plans. Catches regressions where iteration order leaks
    into the output (e.g. swapping `dict` for an unordered structure)."""
    pipeline = _diamond_pipeline()
    compiler = GraphCompiler()
    signatures = {_plan_signature(compiler.compile(pipeline)) for _ in range(10)}
    assert len(signatures) == 1, (
        f"Compilation produced different plans across runs — non-determinism detected: {signatures}"
    )


def test_compile_does_not_mutate_pipeline_or_nodes() -> None:
    """The user must be able to reuse a Pipeline object across calls
    (e.g. `pipe.dry_run()` then `pipe.run()`). If compile mutates the
    nodes or the steps tuple, the second call sees garbage."""
    pipeline = _toy_pipeline()
    before = pipeline.model_dump()
    before_steps_id = id(pipeline.steps)
    GraphCompiler().compile(pipeline)
    after = pipeline.model_dump()
    assert before == after, "GraphCompiler.compile() mutated the Pipeline contents"
    assert id(pipeline.steps) == before_steps_id, (
        "GraphCompiler.compile() rebound Pipeline.steps to a new tuple"
    )


def test_compiled_order_respects_every_dependency() -> None:
    """For every PlannedStep, all of its declared dependencies must
    appear earlier in the plan. This is the one guarantee everything
    else (resume, parallel scheduling, artifact passing) is built on."""
    plan = GraphCompiler().compile(_diamond_pipeline())
    seen: set[str] = set()
    for step in plan.steps:
        for dep in step.depends_on:
            assert dep in seen, (
                f"Step {step.name!r} (order={step.order}) depends on {dep!r}, "
                f"which has not appeared yet. Order so far: {sorted(seen)}"
            )
        seen.add(step.name)


def test_cycle_detection_names_offending_nodes() -> None:
    """Cycle detection must mention the node names involved so the
    user can find the bug. Avoid generic 'cycle detected' messages."""
    pipeline = Pipeline(
        name="cyclic",
        steps=[
            Train(
                name="a",
                framework="sklearn",
                model="LogisticRegression",
                data="b",
                depends_on=("b",),
            ),
            Train(
                name="b",
                framework="sklearn",
                model="LogisticRegression",
                data="a",
                depends_on=("a",),
            ),
        ],
    )
    with pytest.raises(ValueError) as exc:
        GraphCompiler().compile(pipeline)
    msg = str(exc.value).lower()
    assert "cycle" in msg, f"Cycle error must mention 'cycle'; got: {exc.value!r}"
    assert "a" in msg and "b" in msg, (
        f"Cycle error must name the offending nodes; got: {exc.value!r}"
    )


def test_duplicate_node_names_rejected_at_construction() -> None:
    """Defense in depth: duplicates must blow up at Pipeline build
    time, not silently overwrite each other in the `nodes_by_name` dict
    inside the compiler."""
    with pytest.raises(ValidationError) as exc:
        Pipeline(
            name="dup",
            steps=[
                Data(name="x", source="inline://", format="inline"),
                Data(name="x", source="inline://", format="inline"),
            ],
        )
    assert "duplicate" in str(exc.value).lower()


def test_unknown_dependency_rejected_at_construction() -> None:
    """A node referencing a missing upstream must fail loudly at
    Pipeline construction, not produce a half-baked plan."""
    with pytest.raises(ValidationError) as exc:
        Pipeline(
            name="dangling",
            steps=[
                Train(
                    name="trainer",
                    framework="sklearn",
                    model="LogisticRegression",
                    data="ghost",
                    depends_on=("ghost",),
                ),
            ],
        )
    assert "ghost" in str(exc.value)


def test_empty_pipeline_rejected() -> None:
    """A zero-step Pipeline is meaningless — must not silently compile
    to an empty plan that 'succeeds' by doing nothing."""
    with pytest.raises(ValidationError):
        Pipeline(name="empty", steps=[])


def test_plan_contains_every_step_exactly_once() -> None:
    """The plan must cover every node, no more, no less. Catches a
    bug where indegree bookkeeping accidentally drops a node."""
    pipeline = _diamond_pipeline()
    plan = GraphCompiler().compile(pipeline)
    plan_names = [s.name for s in plan.steps]
    pipeline_names = [n.name for n in pipeline.steps]
    assert sorted(plan_names) == sorted(pipeline_names)
    assert len(plan_names) == len(set(plan_names)), "Plan repeats a step"


def test_orders_are_contiguous_starting_at_zero() -> None:
    """`PlannedStep.order` must be a contiguous 0..N-1 range — the
    runtime uses it to index into per-step state."""
    plan = GraphCompiler().compile(_diamond_pipeline())
    orders = [s.order for s in plan.steps]
    assert orders == list(range(len(orders))), f"Non-contiguous orders: {orders}"


def test_pipeline_model_is_immutable() -> None:
    """Pydantic `frozen=True` is what makes `_compile_does_not_mutate`
    a meaningful guarantee. Pin it so a future refactor does not
    silently drop frozen and break callers that rely on hashability."""
    pipeline = _toy_pipeline()
    snapshot = copy.deepcopy(pipeline.model_dump())
    with pytest.raises(ValidationError):
        pipeline.name = "renamed"
    assert pipeline.model_dump() == snapshot
