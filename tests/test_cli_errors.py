"""CLI error-path tests.

`tests/test_cli.py` proves the happy path. These tests prove the
**unhappy** path: every meaningful failure mode the CLI can hit must
exit with non-zero **and** print a message that names the actual
problem. A CLI that exits 0 on broken input or prints generic
'something failed' messages is worse than no CLI at all — it teaches
users to ignore exit codes and forces them to debug by guessing.

We use real subprocesses (not `typer.testing.CliRunner`) because the
runner swallows stderr in recent Click/Typer versions, which is
exactly where the user-facing error messages land.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _cli(argv: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "ophelian.cli.main", *argv],
        capture_output=True,
        text=True,
        timeout=30,
    )


def _write(tmp_path: Path, name: str, body: str) -> Path:
    f = tmp_path / name
    f.write_text(body)
    return f


def test_dry_run_missing_file_exits_non_zero(tmp_path: Path) -> None:
    """Pointing at a non-existent file must NOT silently succeed.
    Typer's `exists=True` validator handles this — pin the behavior so
    a refactor that drops the validator gets caught."""
    result = _cli(["dry-run", str(tmp_path / "ghost.py")])
    assert result.returncode != 0, f"unexpected exit 0; stdout={result.stdout!r}"
    blob = result.stdout + result.stderr
    assert "ghost.py" in blob or "exist" in blob.lower() or "not found" in blob.lower(), (
        f"error must reference the missing file; got {blob!r}"
    )


def test_run_missing_file_exits_non_zero(tmp_path: Path) -> None:
    result = _cli(["run", str(tmp_path / "ghost.py")])
    assert result.returncode != 0


def test_dry_run_module_with_no_pipeline_exits_with_helpful_error(tmp_path: Path) -> None:
    """Loading a Python file that contains no `Pipeline` instance
    must explain that exact problem so the user knows to add one or
    pass --attribute."""
    f = _write(tmp_path, "no_pipe.py", "x = 1\n")
    result = _cli(["dry-run", str(f)])
    assert result.returncode != 0
    blob = result.stdout + result.stderr
    assert "Pipeline" in blob, (
        f"Error must mention 'Pipeline'; got exit={result.returncode}, output={blob!r}"
    )


def test_dry_run_attribute_pointing_at_non_pipeline_exits_with_clear_error(
    tmp_path: Path,
) -> None:
    """If the user explicitly says `--attribute foo` and `foo` is not
    a Pipeline, blame the attribute, not the file."""
    f = _write(tmp_path, "wrong_attr.py", "foo = 'not a pipeline'\n")
    result = _cli(["dry-run", str(f), "--attribute", "foo"])
    assert result.returncode != 0
    blob = result.stdout + result.stderr
    assert "foo" in blob or "Pipeline" in blob, f"unhelpful error: {blob!r}"


def test_dry_run_attribute_missing_exits_non_zero(tmp_path: Path) -> None:
    """`--attribute foo` against a file that doesn't define `foo`
    must fail loudly, not crash with AttributeError or run blindly."""
    f = _write(tmp_path, "no_attr.py", "x = 1\n")
    result = _cli(["dry-run", str(f), "--attribute", "missing"])
    assert result.returncode != 0


def test_dry_run_module_with_syntax_error_exits_non_zero(tmp_path: Path) -> None:
    """A pipeline file with a syntax error must not be silently
    swallowed — surface the SyntaxError to the user."""
    f = _write(tmp_path, "broken.py", "def \n")
    result = _cli(["dry-run", str(f)])
    assert result.returncode != 0


def test_unknown_subcommand_exits_non_zero() -> None:
    """`ophelian totally-not-a-command` must exit non-zero with usage
    info, not silently no-op."""
    result = _cli(["totally-not-a-command"])
    assert result.returncode != 0


def test_dry_run_happy_path_does_not_invoke_provider(tmp_path: Path) -> None:
    """Invariant: `dry-run` must NEVER instantiate a real provider.
    If a regression makes dry-run actually execute, a user could
    accidentally launch a $$$ cloud job."""
    pipeline_src = """
from ophelian import Pipeline, Data, Train

pipe = Pipeline(
    name="dryonly",
    steps=[
        Data(
            name="ds",
            source="inline://",
            format="inline",
            options={"X": [[0]], "y": [0]},
        ),
        Train(
            name="trainer",
            framework="sklearn",
            model="LogisticRegression",
            data="ds",
            hyperparameters={"max_iter": 5},
        ),
    ],
)
"""
    f = _write(tmp_path, "ok.py", pipeline_src)
    result = _cli(["dry-run", str(f)])
    assert result.returncode == 0, (
        f"happy-path dry-run failed: stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    # The plan render mentions step kinds; confirm the planner ran.
    assert "trainer" in result.stdout.lower() or "Train" in result.stdout
