#!/usr/bin/env python3
"""Audit helper for the top-level ``NOTICE`` file.

(This script is intentionally an *auditor*, not a generator: the
upstream copyright lines in ``NOTICE`` are human-curated and cannot
be reliably derived from ``uv export`` / ``pip show`` output, so we
keep ``NOTICE`` authoritative and use this script to fail on drift.)

The NOTICE file at the repository root enumerates the third-party
Python distributions Ophelian declares in ``pyproject.toml`` (core
dependencies + the optional ``[aws]`` / ``[gcp]`` / ``[azure]`` /
``[pytorch]`` / ``[huggingface]`` / ``[sklearn]`` / ``[xgboost]`` /
``[otel]`` extras) so that downstream redistributors and enterprise
consumers have a single auditable place to find upstream
attributions.

This script does not *edit* ``NOTICE`` in place. Instead, it
cross-references the set of declared dependencies in
``pyproject.toml`` against the project names mentioned in the
existing ``NOTICE`` file and reports:

* dependencies declared in ``pyproject.toml`` that are missing an
  entry in ``NOTICE`` (likely a packaging audit follow-up — add
  the attribution before the next release), and
* names mentioned in ``NOTICE`` that no longer correspond to a
  declared dependency (likely a stale entry — verify and remove).

Intentional design notes:

* The script is read-only by default and exits non-zero when it
  finds drift, so it can be wired into a release / CI check
  without surprising side effects.
* Transitive dependencies are deliberately *not* expanded here.
  Use ``uv export --format requirements-txt`` or an SBOM tool
  (e.g. ``cyclonedx-py``) when the full transitive surface
  matters.

Usage::

    python scripts/audit_notice.py
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"
NOTICE = REPO_ROOT / "NOTICE"

# Extras whose declared dependencies must be attributed in NOTICE.
# ``dev`` is intentionally excluded — those are not redistributed
# to PyPI consumers. ``all`` is excluded because it is a union of
# the others. ``docs`` and ``pricing`` are tooling-only / empty.
_TRACKED_EXTRAS = {
    "pytorch",
    "huggingface",
    "sklearn",
    "xgboost",
    "aws",
    "eks",
    "gcp",
    "gke",
    "azure",
    "aks",
    "otel",
}

# Distribution names we deliberately do not list in NOTICE even if
# they appear under a tracked extra (none today, but kept for
# future deny-listing).
_IGNORED: set[str] = set()


def _normalize(name: str) -> str:
    """PEP 503 normalisation: lower-case, runs of [-_.] → ``-``."""
    return re.sub(r"[-_.]+", "-", name).strip().lower()


def _dep_name(spec: str) -> str:
    """Extract the bare distribution name from a PEP 508 spec string."""
    # Strip env markers, extras, version specifiers.
    head = re.split(r"[\[<>=!~;\s]", spec, maxsplit=1)[0]
    return _normalize(head)


def _declared_dependencies() -> set[str]:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    project = data.get("project", {})
    deps: set[str] = {_dep_name(d) for d in project.get("dependencies", [])}
    extras = project.get("optional-dependencies", {})
    for extra_name, extra_deps in extras.items():
        if extra_name not in _TRACKED_EXTRAS:
            continue
        deps.update(_dep_name(d) for d in extra_deps)
    return {d for d in deps if d and d not in _IGNORED}


def _notice_mentions() -> set[str]:
    """Return the set of normalised distribution names referenced in NOTICE.

    The NOTICE file uses lines of the form::

        <project>  --  <license>  --  <copyright>

    so we pick the first whitespace-delimited token of every line that
    matches that shape.
    """
    mentions: set[str] = set()
    pattern = re.compile(r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)\s+--\s+")
    for raw in NOTICE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        match = pattern.match(line)
        if match:
            mentions.add(_normalize(match.group("name")))
    return mentions


def main() -> int:
    declared = _declared_dependencies()
    mentioned = _notice_mentions()

    missing = sorted(declared - mentioned)
    stale = sorted(mentioned - declared)

    if not missing and not stale:
        print(
            f"NOTICE is in sync with pyproject.toml: "
            f"{len(declared)} declared dependencies, all attributed."
        )
        return 0

    if missing:
        print("Declared in pyproject.toml but missing from NOTICE:")
        for name in missing:
            print(f"  - {name}")
    if stale:
        print("Mentioned in NOTICE but no longer declared in pyproject.toml:")
        for name in stale:
            print(f"  - {name}")
    print(
        "\nUpdate NOTICE so it lists every redistributed third-party "
        "component, then re-run this script."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
