"""Tests for the typer-based CLI."""

from __future__ import annotations

import re
from pathlib import Path

from ophelian._version import __version__
from ophelian.cli import app
from typer.testing import CliRunner

runner = CliRunner()

# Rich's auto-highlighting wraps numbers (and other tokens) in ANSI escape
# sequences when the CLI runs under a CI terminal that advertises color
# support. That breaks naive substring asserts like ``"1.0.0" in stdout``
# because the version becomes ``\x1b[...m1.0\x1b[0m.\x1b[...m0\x1b[0m``.
# We strip ANSI here to keep assertions robust without disabling Rich's
# user-facing highlighting in the CLI itself.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _plain(text: str) -> str:
    return _ANSI_RE.sub("", text)


def _write_pipeline(tmp_path: Path) -> Path:
    pipeline_file = tmp_path / "my_pipe.py"
    pipeline_file.write_text(
        """
from ophelian import Data, Deploy, Pipeline, Train

pipe = Pipeline([
    Data(
        name="ds",
        source="memory://toy",
        format="inline",
        options={"X": [[0.0, 0.1], [0.1, 0.2], [0.9, 1.0], [1.0, 0.9]], "y": [0, 0, 1, 1]},
    ),
    Train(
        name="trainer",
        framework="sklearn",
        model="sklearn.linear_model.LogisticRegression",
        data="ds",
    ),
    Deploy(name="serve", model="trainer", port=9100),
], name="cli-test")
"""
    )
    return pipeline_file


def test_version_command_prints_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in _plain(result.stdout)


def test_dry_run_prints_plan(tmp_path: Path) -> None:
    pipeline_file = _write_pipeline(tmp_path)
    result = runner.invoke(app, ["dry-run", str(pipeline_file)])
    assert result.exit_code == 0, result.stdout
    assert "trainer" in result.stdout


def test_run_executes_pipeline(tmp_path: Path) -> None:
    pipeline_file = _write_pipeline(tmp_path)
    result = runner.invoke(app, ["run", str(pipeline_file)])
    assert result.exit_code == 0, result.stdout
    # All three steps should appear in the rendered status table.
    for step in ("ds", "trainer", "serve"):
        assert step in result.stdout
    assert "success" in result.stdout


def test_dry_run_with_attribute(tmp_path: Path) -> None:
    pipeline_file = _write_pipeline(tmp_path)
    result = runner.invoke(app, ["dry-run", str(pipeline_file), "--attribute", "pipe"])
    assert result.exit_code == 0, result.stdout
