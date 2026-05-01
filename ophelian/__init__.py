"""Ophelian — declarative ML framework.

Write your pipeline once with the high-level DSL and run it anywhere — locally
inside Docker today, on the cloud tomorrow. The top-level package only re-exports
the small surface that pipeline authors need; the rest of the codebase is
organised into focused subpackages.
"""

from __future__ import annotations

from ophelian._version import __version__
from ophelian.core import (
    Data,
    Deploy,
    Eval,
    Pipeline,
    Train,
    Tune,
)
from ophelian.envs import AWS, Standalone

__all__ = [
    "AWS",
    "Data",
    "Deploy",
    "Eval",
    "Pipeline",
    "Standalone",
    "Train",
    "Tune",
    "__version__",
]
