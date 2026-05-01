"""Regression: every viral demo must run end-to-end in dry-run mode.

The Auto router contract is "no env, no infra, no creds → still works
as a no-op pipeline you can paste into a slide". Each script is
imported as a module and `main()` is invoked under
``OPHELIAN_DRY_RUN=1``. We assert the run prints something and does
not raise.
"""

from __future__ import annotations

import importlib
import io
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest

DEMOS = ["llama_finetune", "resnet_train", "xgboost_tabular"]


@pytest.fixture(autouse=True)
def _clear_demo_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force a default-env smoke test by clearing every OPHELIAN_* override."""
    for key in list(__import__("os").environ):
        if key.startswith("OPHELIAN_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("OPHELIAN_DRY_RUN", "1")
    monkeypatch.setenv("OPHELIAN_REQUIRE_CREDS", "0")


@pytest.mark.parametrize("module_name", DEMOS)
def test_demo_dry_run(module_name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    examples_dir = Path(__file__).resolve().parent.parent / "examples"
    monkeypatch.syspath_prepend(str(examples_dir))
    sys.modules.pop(module_name, None)
    module = importlib.import_module(module_name)

    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        module.main()

    combined = out.getvalue() + err.getvalue()
    assert combined.strip(), f"{module_name} produced no output"
    assert "Cheapest" in combined or "auto-router" in combined or "auto-dry-run" in combined, (
        f"{module_name} did not show a routing decision: {combined!r}"
    )
