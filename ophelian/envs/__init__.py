"""Friendly env factories that wrap providers in pipeline-author syntax.

Pipelines are written declaratively, so the env constructor is the user-facing
entry point — not the provider class. `Standalone(local=True)` returns the
local Docker provider; future cloud envs will live alongside it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ophelian.providers.standalone import StandaloneProvider

if TYPE_CHECKING:
    from ophelian.providers.base import Provider


def Standalone(*, local: bool = True, **kwargs: Any) -> Provider:
    """Return a `StandaloneProvider`.

    `local=True` is the only mode supported in v0.1; future remote-standalone
    deployments will reuse the same constructor.
    """
    return StandaloneProvider(local=local, **kwargs)


__all__ = ["Standalone"]
