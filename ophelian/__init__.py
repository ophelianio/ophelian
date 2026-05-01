"""Ophelian — declarative ML framework.

Write your pipeline once with the high-level DSL and run it anywhere
— locally inside Docker, on AWS, GCP, Azure, or via the cost-aware
:func:`Auto` router that picks the cheapest cloud for a given GPU
class. The top-level package only re-exports the small surface
pipeline authors need; the rest of the codebase is organised into
focused subpackages (``core``, ``envs``, ``providers``, ``stores``,
``pricing``, ``observability``, ``runtime``).
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
from ophelian.envs import AWS, GCP, Auto, Azure, Standalone

__all__ = [
    "AWS",
    "GCP",
    "Auto",
    "Azure",
    "Data",
    "Deploy",
    "Eval",
    "Pipeline",
    "Standalone",
    "Train",
    "Tune",
    "__version__",
]
