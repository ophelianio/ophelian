"""End-to-end pipeline integration tests.

These exercise a **real** Standalone run from `Pipeline.run(env=...)` to
on-disk artifacts, with synthetic in-memory data. They are
deliberately small and deterministic so they belong in the default
suite — but they cover the full machinery: compile → schedule →
per-step dispatch → adapter → artifact persistence → PipelineResult
aggregation.

If any of these fail, the framework is unusable end-to-end regardless
of how many unit tests pass.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from ophelian import Data, Eval, Pipeline, Train
from ophelian.providers.standalone import StandaloneProvider

# A tiny strictly linearly-separable dataset — class label = (x0 > 0).
# LogisticRegression converges in a handful of iterations and reaches
# perfect accuracy on the training set.
_X = [
    [-2.0, -1.0],
    [-1.5, 0.0],
    [-1.0, 1.0],
    [-0.5, -0.5],
    [0.5, 0.5],
    [1.0, -1.0],
    [1.5, 0.0],
    [2.0, 1.0],
]
_Y = [0, 0, 0, 0, 1, 1, 1, 1]


@pytest.fixture
def provider(tmp_path: Path) -> StandaloneProvider:
    """Standalone provider pinned to a clean per-test workspace and
    container mode forced off so we never accidentally try to spin up
    Docker in CI."""
    return StandaloneProvider(workspace=tmp_path / "ws", container=False)


def _toy_pipeline() -> Pipeline:
    return Pipeline(
        name="e2e",
        steps=[
            Data(
                name="ds",
                source="memory://toy",
                format="inline",
                options={"X": _X, "y": _Y},
            ),
            Train(
                name="trainer",
                framework="sklearn",
                model="sklearn.linear_model.LogisticRegression",
                data="ds",
                hyperparameters={"max_iter": 200},
            ),
            Eval(name="ev", model="trainer", data="ds", metrics=("accuracy",)),
        ],
    )


def test_full_pipeline_runs_and_returns_successful_result(
    provider: StandaloneProvider,
) -> None:
    """The whole 3-step pipeline must succeed end-to-end and return a
    PipelineResult whose .succeeded is True with one StepResult per
    declared node, in declaration order."""
    pipe = _toy_pipeline()
    result = pipe.run(env=provider)

    assert result.succeeded, (
        f"Pipeline reported failure: {[(s.name, s.status, s.error) for s in result.steps]}"
    )
    assert result.pipeline == "e2e"
    assert [s.name for s in result.steps] == ["ds", "trainer", "ev"]
    assert all(s.status == "success" for s in result.steps)
    # Every step must report a duration so observability has something
    # to render.
    assert all(s.duration_seconds is not None for s in result.steps)


def test_data_step_persists_dataset_json(provider: StandaloneProvider) -> None:
    """Inline Data must materialize `dataset.json` on disk so the
    downstream Train step can read it across process boundaries."""
    result = _toy_pipeline().run(env=provider)
    ds = result.step("ds")
    dataset_path = Path(ds.artifacts["dataset"])
    assert dataset_path.exists(), f"missing dataset artifact: {ds.artifacts}"
    payload = json.loads(dataset_path.read_text())
    assert payload["X"] == _X
    assert payload["y"] == _Y


def test_train_step_persists_model_artifact_and_descriptor(
    provider: StandaloneProvider,
) -> None:
    """Train must write both the framework-specific model file
    (joblib/pkl) AND an `ophelian.json` descriptor so the runtime can
    rehydrate it later (resume, deploy)."""
    result = _toy_pipeline().run(env=provider)
    trainer = result.step("trainer")
    model_dir = Path(trainer.artifacts["model"])
    assert model_dir.is_dir(), f"model dir missing: {model_dir}"
    assert (model_dir / "ophelian.json").exists(), (
        "Missing ophelian.json descriptor — resume/deploy will break."
    )
    artifact = Path(trainer.artifacts["artifact"])
    assert artifact.exists(), f"missing model file: {artifact}"
    assert artifact.suffix in {".joblib", ".pkl"}, (
        f"unexpected model artifact extension: {artifact.suffix}"
    )


def test_eval_step_emits_accuracy_metric(provider: StandaloneProvider) -> None:
    """Eval must surface declared metrics. The toy dataset is
    linearly separable, so accuracy must be 1.0; any other value
    means the data flow from Data→Train→Eval is broken."""
    result = _toy_pipeline().run(env=provider)
    ev = result.step("ev")
    assert "accuracy" in ev.metrics, f"Eval did not report 'accuracy'; metrics={ev.metrics!r}"
    assert ev.metrics["accuracy"] == pytest.approx(1.0), (
        f"accuracy != 1.0 on linearly-separable toy data: {ev.metrics['accuracy']}"
    )


def test_dry_run_does_not_create_workspace_artifacts(
    provider: StandaloneProvider, tmp_path: Path
) -> None:
    """`dry_run=True` must NOT execute any step — no files written,
    no model trained. Catches a regression where dry_run accidentally
    falls through to the real executor."""
    workspace = provider._workspace
    pipe = _toy_pipeline()
    result = pipe.run(env=provider, dry_run=True)
    # `Pipeline.run(dry_run=True)` returns an empty PipelineResult.
    # Crucially, no steps were executed.
    assert result.steps == [], f"dry_run produced step results: {result.steps}"
    # Workspace must contain no per-step subdirs.
    leftover = [p.name for p in workspace.iterdir()] if workspace.exists() else []
    assert not any(name in leftover for name in ("ds", "trainer", "ev")), (
        f"dry_run wrote artifacts: {leftover}"
    )


def test_failed_step_short_circuits_remaining_steps(
    provider: StandaloneProvider,
) -> None:
    """If a step fails, downstream steps must NOT run (their status
    must be ``skipped`` or absent), and `result.succeeded` must be
    False so callers can branch on it without parsing strings."""
    pipe = Pipeline(
        name="will-fail",
        steps=[
            Data(
                name="ds",
                source="memory://toy",
                format="inline",
                options={"X": _X, "y": _Y},
            ),
            Train(
                name="bad_trainer",
                framework="sklearn",
                # Nonsense model path — adapter must raise on load.
                model="totally.not.a.real.Estimator",
                data="ds",
            ),
            Eval(name="ev", model="bad_trainer", data="ds"),
        ],
    )
    result = pipe.run(env=provider)
    assert not result.succeeded
    bad = result.step("bad_trainer")
    assert bad.status == "failed"
    assert bad.error, "failed step must carry an error message for the user"
    # Eval must NOT have run successfully against a missing model.
    ev_present = any(s.name == "ev" and s.status == "success" for s in result.steps)
    assert not ev_present, "downstream Eval ran after Train failed"
