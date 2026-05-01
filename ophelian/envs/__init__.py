"""Friendly env factories that wrap providers in pipeline-author syntax.

Pipelines are written declaratively, so the env constructor is the
user-facing entry point — not the provider class. Every cloud is
accessed through a single-symbol factory that mirrors the same shape:

* :func:`Standalone` — local Docker / in-process executor.
* :func:`AWS` — AWS (EC2 + EKS).
* :func:`GCP` — Google Cloud (GCE).
* :func:`Azure` — Microsoft Azure (VM).
* :func:`Auto` — cost-aware router that picks the cheapest provider
  for a requested GPU class.

Third-party clouds can plug in via the ``ophelian.envs`` entry-point
group declared in ``pyproject.toml``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ophelian.envs.auto import Auto, AutoRouterError
from ophelian.envs.aws import AWS, AWSConfig
from ophelian.envs.azure import Azure, AzureConfig
from ophelian.envs.gcp import GCP, GCPConfig
from ophelian.providers.standalone import StandaloneProvider

if TYPE_CHECKING:
    from ophelian.providers.base import Provider


def Standalone(*, local: bool = True, **kwargs: Any) -> Provider:
    """Return a :class:`~ophelian.providers.standalone.StandaloneProvider`.

    ``local=True`` is the only mode supported in v1.0; future remote-
    standalone deployments will reuse the same constructor.
    """
    return StandaloneProvider(local=local, **kwargs)


def discover_plugin_envs() -> dict[str, Any]:
    """Discover env factories registered via the ``ophelian.envs`` entry-point.

    Third-party packages can ship a custom env (e.g. ``OnPrem``) by
    declaring an entry point::

        [project.entry-points."ophelian.envs"]
        OnPrem = "my_pkg.envs:OnPrem"

    The return value is ``{name: callable}``. Failures import-side are
    logged and swallowed so a broken plugin does not poison the host
    process.
    """
    import importlib.metadata as md
    import logging

    log = logging.getLogger("ophelian.envs")
    plugins: dict[str, Any] = {}
    try:
        eps = md.entry_points(group="ophelian.envs")
    except TypeError:  # pragma: no cover - python <3.10
        eps = md.entry_points().get("ophelian.envs", [])  # type: ignore[arg-type]
    for ep in eps:
        try:
            plugins[ep.name] = ep.load()
        except Exception as exc:  # pragma: no cover
            log.warning("Failed to load env plugin %s: %s", ep.name, exc)
    return plugins


__all__ = [
    "AWS",
    "GCP",
    "AWSConfig",
    "Auto",
    "AutoRouterError",
    "Azure",
    "AzureConfig",
    "GCPConfig",
    "Standalone",
    "discover_plugin_envs",
]
