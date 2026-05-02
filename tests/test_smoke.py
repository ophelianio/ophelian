"""Smoke tests — does the package even import and stand up?

These are the cheapest, loudest tests in the suite. They run in a
fresh Python subprocess so import-time side effects (network calls,
disk writes, slow optional deps) get caught instead of hidden by the
already-warm test process.

A failure here means the package is broken at the most basic level —
no point running anything else.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest


def _python(code: str) -> subprocess.CompletedProcess[str]:
    """Run *code* in a fresh Python subprocess and return the result."""
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_top_level_package_imports_in_fresh_process() -> None:
    """`import ophelian` must not raise, must not hit the network, and
    must not leave junk on stderr."""
    result = _python("import ophelian; print(ophelian.__version__)")
    assert result.returncode == 0, f"stderr was: {result.stderr}"
    assert result.stdout.strip(), "ophelian.__version__ must be printable"
    # Import-time stderr noise (warnings, deprecation messages) is a
    # smell — pin it down so we notice if we add a chatty dependency.
    assert "Error" not in result.stderr
    assert "Traceback" not in result.stderr


def test_every_subpackage_imports_in_fresh_process() -> None:
    """Each subpackage must import standalone — catches circular
    imports that the warm test process would mask."""
    subpackages = [
        "ophelian.cli",
        "ophelian.core",
        "ophelian.data",
        "ophelian.envs",
        "ophelian.models",
        "ophelian.observability",
        "ophelian.pricing",
        "ophelian.pricing.live",
        "ophelian.providers",
        "ophelian.runtime",
        "ophelian.stores",
    ]
    code = "\n".join(f"import {pkg}" for pkg in subpackages)
    result = _python(code)
    assert result.returncode == 0, f"stderr was: {result.stderr}"


def test_top_level_reexports_are_callable_or_class() -> None:
    """Every name in ``ophelian.__all__`` must resolve to something
    that is at minimum callable or a string (for ``__version__``)."""
    import ophelian

    for name in ophelian.__all__:
        assert hasattr(ophelian, name), f"{name} missing from ophelian"
        value = getattr(ophelian, name)
        if name == "__version__":
            assert isinstance(value, str) and value, "__version__ must be a non-empty string"
            continue
        assert callable(value), f"{name} should be a class/callable, got {type(value)!r}"


@pytest.mark.parametrize(
    "argv", [["version"], ["--help"], ["dry-run", "--help"], ["run", "--help"]]
)
def test_cli_basic_invocations_exit_zero(argv: list[str]) -> None:
    """``python -m ophelian.cli.main <argv>`` must exit 0 for the
    non-destructive informational commands. Failure here means the CLI
    entry point is broken before a user even has a pipeline file."""
    result = subprocess.run(
        [sys.executable, "-m", "ophelian.cli.main", *argv],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, (
        f"argv={argv!r} exit={result.returncode} stderr={result.stderr!r}"
    )


def test_version_string_is_pep440() -> None:
    """``__version__`` must look like a PEP 440 version. Catches
    accidents where the build pipeline strips or mangles the tag."""
    import re

    import ophelian

    # Loose PEP 440: digits/dots, optional pre/post/dev, optional local segment.
    assert re.match(
        r"^\d+(\.\d+)*((a|b|rc|\.post|\.dev)\d+)?(\+[a-zA-Z0-9.]+)?$",
        ophelian.__version__,
    ), f"Not a PEP 440 version: {ophelian.__version__!r}"
