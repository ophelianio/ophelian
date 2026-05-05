"""Auto env — pick the cheapest cloud + region for a GPU class.

Usage::

    from ophelian import Auto, Pipeline

    env = Auto(cheapest_gpu="A100", regions=["us-east-1", "us-central1"])
    pipeline.run(env=env)

The Auto env consults :func:`ophelian.pricing.lookup_cheapest_with_meta`
(legacy :func:`ophelian.pricing.lookup_cheapest` is the back-compat
wrapper that drops the provenance map), picks
the winner and constructs the matching :func:`AWS` / :func:`GCP` /
:func:`Azure` provider with credentials inferred from the local
environment. ``dry_run=True`` returns a wrapper that prints what would
have run and then no-ops every step — perfect for blogposts and CI.

Per the v1.0 release contract documented in ``docs/envs/auto.md``, the
router stays *deliberately conservative*: it never picks a provider
the user has not enabled (via ``providers=[...]``), and it never
silently swaps clouds mid-run. If the cheapest provider has missing
credentials we raise :class:`AutoRouterError` with an actionable
message instead of falling back to a more expensive cloud the user did
not consent to.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import TYPE_CHECKING, Any

from ophelian.pricing import (
    PriceQuote,
    RouterDecision,
    explain,
    lookup_cheapest_with_meta,
    static_quotes,
)
from ophelian.providers.base import Provider

if TYPE_CHECKING:  # pragma: no cover
    from ophelian.providers.base import Provider

logger = logging.getLogger("ophelian.envs.auto")


class AutoRouterError(RuntimeError):
    """Raised when the Auto router cannot satisfy a request."""


SUPPORTED_PROVIDERS = ("aws", "gcp", "azure")


def _detect_credentials() -> dict[str, bool]:
    """Best-effort detection of which clouds the local env can reach."""
    aws = bool(
        os.environ.get("AWS_ACCESS_KEY_ID")
        or os.environ.get("AWS_PROFILE")
        or os.path.exists(os.path.expanduser("~/.aws/credentials"))
    )
    gcp = bool(
        os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        or os.path.exists(
            os.path.expanduser("~/.config/gcloud/application_default_credentials.json")
        )
    )
    azure = bool(
        os.environ.get("AZURE_CLIENT_ID")
        or os.environ.get("AZURE_SUBSCRIPTION_ID")
        or os.path.exists(os.path.expanduser("~/.azure/azureProfile.json"))
    )
    return {"aws": aws, "gcp": gcp, "azure": azure}


def _build_provider_for_quote(
    quote: PriceQuote,
    *,
    spot: bool,
    project: str | None,
    subscription_id: str | None,
    resource_group: str | None,
    artifact_account: str | None,
    extra_kwargs: dict[str, Any] | None = None,
) -> Provider:
    extras = extra_kwargs or {}
    if quote.provider == "aws":
        from ophelian.envs.aws import AWS

        return AWS(
            region=quote.region,
            instance=_aws_instance_for(quote),
            spot=spot,
            **extras,
        )
    if quote.provider == "gcp":
        from ophelian.envs.gcp import GCP

        if not project:
            raise AutoRouterError(
                "GCP routing needs a project — pass `Auto(..., project='my-project')`"
                " or export `GOOGLE_CLOUD_PROJECT`."
            )
        machine_type, gpu_type = _gcp_instance_for(quote)
        return GCP(
            project=project,
            region=quote.region,
            machine_type=machine_type,
            gpu_type=gpu_type,
            gpu_count=quote.gpu_count,
            spot=spot,
            **extras,
        )
    if quote.provider == "azure":
        from ophelian.envs.azure import Azure

        if not (subscription_id and resource_group):
            raise AutoRouterError(
                "Azure routing needs a subscription_id + resource_group — pass them"
                " explicitly to Auto(...) or export AZURE_SUBSCRIPTION_ID/AZURE_RESOURCE_GROUP."
            )
        return Azure(
            subscription_id=subscription_id,
            resource_group=resource_group,
            location=quote.region,
            vm_size=quote.instance,
            spot=spot,
            artifact_account=artifact_account or os.environ.get("AZURE_STORAGE_ACCOUNT"),
            **extras,
        )
    raise AutoRouterError(f"Auto router does not yet support provider {quote.provider!r}")


def _aws_instance_for(quote: PriceQuote) -> str:
    """The static table records the EC2 instance type verbatim."""
    return quote.instance


def _gcp_instance_for(quote: PriceQuote) -> tuple[str, str | None]:
    """Map a pricing-table GCP instance to ``(machine_type, gpu_type)``.

    Two notations are supported:

    * Explicit suffix — ``"n1-standard-4+t4"`` → ``("n1-standard-4",
      "nvidia-tesla-t4")``. Used for the older N1 family where any
      accelerator can be attached à la carte.
    * Implicit, baked-in accelerator — ``"a2-highgpu-1g"`` (A100),
      ``"a3-highgpu-8g"`` (H100), ``"g2-standard-4"`` (L4). For these
      modern families the GPU is part of the SKU itself, so we infer
      ``gpu_type`` from the machine prefix. ``GCPConfig`` validation
      requires ``gpu_type`` whenever ``gpu_count > 0``, and the router
      always passes ``gpu_count`` from the quote — so returning ``None``
      here would raise on real construction.
    """
    name = quote.instance
    if "+" in name:
        machine, gpu_short = name.split("+", 1)
        gpu_type = {
            "t4": "nvidia-tesla-t4",
            "v100": "nvidia-tesla-v100",
            "p100": "nvidia-tesla-p100",
            "a100": "nvidia-tesla-a100",
            "h100": "nvidia-h100-80gb",
            "l4": "nvidia-l4",
        }.get(gpu_short.lower(), f"nvidia-tesla-{gpu_short.lower()}")
        return machine, gpu_type

    # Implicit-accelerator families: GPU is part of the SKU.
    family = name.split("-", 1)[0].lower()
    implicit_gpu = {
        "a2": "nvidia-tesla-a100",
        "a3": "nvidia-h100-80gb",
        "g2": "nvidia-l4",
    }.get(family)
    return name, implicit_gpu


# ---------------------------------------------------------------------------
# Dry-run no-op provider
# ---------------------------------------------------------------------------


class _DryRunProvider(Provider):
    """Provider returned by ``Auto(..., dry_run=True)``.

    Skips every step and returns a synthetic :class:`PipelineResult` so
    users can validate routing without spinning up infra. The chosen
    provider/region are surfaced in ``info`` and printed to stderr.
    """

    name = "auto-dry-run"

    def __init__(self, decision: RouterDecision) -> None:
        self._decision = decision

    @property
    def decision(self) -> RouterDecision:
        return self._decision

    def describe(self) -> str:
        q = self._decision.quote
        return f"auto-dry-run -> {q.provider}/{q.region} {q.instance} ({q.hourly_usd:.3f} USD/h)"

    def execute(self, pipeline: Any, plan: Any) -> Any:
        from ophelian.core.nodes import PipelineResult, StepResult

        sys.stderr.write(explain(self._decision) + "\n")
        results = [
            StepResult(
                name=step.name,
                kind=step.kind,
                status="skipped",
                info={"reason": "auto-router dry-run", "would_run_on": self.describe()},
            )
            for step in plan.steps
        ]
        return PipelineResult(pipeline=pipeline.name, steps=results)

    def cleanup(self) -> None:  # pragma: no cover - nothing to clean up
        return None


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def Auto(
    *,
    cheapest_gpu: str,
    regions: list[str] | None = None,
    providers: list[str] | None = None,
    spot: bool = True,
    dry_run: bool = False,
    allow_live: bool = True,
    require_live: list[str] | None = None,
    project: str | None = None,
    subscription_id: str | None = None,
    resource_group: str | None = None,
    artifact_account: str | None = None,
    require_credentials: bool = True,
    **provider_kwargs: Any,
) -> Provider:
    """Pick the cheapest cloud/region for *cheapest_gpu* and build a provider.

    Parameters
    ----------
    cheapest_gpu:
        GPU family — ``"T4"``, ``"L4"``, ``"V100"``, ``"A10G"``, ``"A100"``,
        ``"H100"``.
    regions:
        Whitelisted regions across all providers (e.g.
        ``["us-east-1", "us-central1", "eastus"]``). When omitted, every
        region in the static table is eligible.
    providers:
        Whitelisted providers (defaults to whichever clouds have local
        credentials when ``require_credentials=True``).
    spot:
        Prefer spot/preemptible quotes (default ``True``). Falls back
        to on-demand if no spot quote matches.
    dry_run:
        Return a :class:`_DryRunProvider` that prints the chosen
        provider/region and skips every step. Great for blog posts and
        CI smoke tests.
    allow_live:
        Pass through to :func:`pricing.lookup_cheapest`. Defaults to
        ``True`` so the router queries AWS spot pricing live (with a
        24 h on-disk cache) and only falls back to the static table
        when no live quote is available. Set to ``False`` to force
        offline routing.
    require_live:
        Strict-mode allow-list. When given, every provider in this
        list must have provenance ``"live"`` in the resulting
        :attr:`RouterDecision.data_quality` map — anything else
        (``"cached@<N>h"``, ``"static"``, ``"unavailable"``,
        ``"disabled"``) raises :class:`AutoRouterError` with the
        actual provenance map in the message. Use when you would
        rather fail loudly than route on stale or static prices.
        Defaults to ``None`` (permissive: any provenance is fine).
    project / subscription_id / resource_group / artifact_account:
        Cloud-specific identifiers needed when the router picks GCP /
        Azure. Falls back to env vars.
    require_credentials:
        When ``True`` (default), only consider providers whose local
        credentials look configured. Set to ``False`` to let the router
        consider any provider in ``providers``.
    **provider_kwargs:
        Forwarded to the underlying ``AWS()`` / ``GCP()`` / ``Azure()``
        factory. Used for advanced overrides (e.g. ``runtime_extras``).
    """
    detected = _detect_credentials()
    if providers is None:
        if require_credentials:
            providers = [name for name, ok in detected.items() if ok]
            if not providers:
                providers = list(SUPPORTED_PROVIDERS)
        else:
            providers = list(SUPPORTED_PROVIDERS)

    candidates = static_quotes(cheapest_gpu, providers=providers, regions=regions)
    if spot:
        spot_only = [q for q in candidates if q.spot]
        if spot_only:
            candidates = spot_only
    quote, data_quality = lookup_cheapest_with_meta(
        cheapest_gpu,
        providers=providers,
        regions=regions,
        spot=spot if any(q.spot for q in candidates) else None,
        allow_live=allow_live,
    )
    if quote is None:
        raise AutoRouterError(
            f"No quotes available for {cheapest_gpu!r} across providers={providers!r}"
            f", regions={regions!r}. Pricing table last reviewed:"
            f" {os.environ.get('OPHELIAN_PRICING_REVIEW', 'see ophelian.pricing.STATIC_PRICES_LAST_REVIEW')}."
        )
    decision = RouterDecision(quote=quote, considered=candidates, data_quality=data_quality)

    if require_live:
        not_live = {
            p: data_quality.get(p, "disabled")
            for p in require_live
            if data_quality.get(p) != "live"
        }
        if not_live:
            details = ", ".join(f"{p}={v!r}" for p, v in sorted(not_live.items()))
            raise AutoRouterError(
                f"require_live={require_live!r} but provenance was {details}."
                f" Full data_quality map: {data_quality!r}."
                " Either widen require_live, set allow_live=True, or accept"
                " the static fallback by removing this kwarg."
            )

    # Provenance suffix renders providers in stable alphabetical order
    # so log scrapers and dashboards parse a deterministic format.
    data_suffix = " ".join(f"{p}={data_quality[p]}" for p in sorted(data_quality))
    logger.info(
        "Auto router selected %s/%s %s @ %.3f USD/h (%s) | data: %s | considered: %d quotes",
        quote.provider,
        quote.region,
        quote.instance,
        quote.hourly_usd,
        "spot" if quote.spot else "on-demand",
        data_suffix,
        len(candidates),
    )

    if dry_run:
        return _DryRunProvider(decision)

    if require_credentials and not detected.get(quote.provider, False):
        raise AutoRouterError(
            f"Auto router picked {quote.provider!r} but no local credentials"
            f" were detected. Either configure them or pass providers=[...]"
            " restricting Auto to clouds you have access to."
        )

    if quote.provider == "gcp" and project is None:
        project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if quote.provider == "azure":
        subscription_id = subscription_id or os.environ.get("AZURE_SUBSCRIPTION_ID")
        resource_group = resource_group or os.environ.get("AZURE_RESOURCE_GROUP")

    provider = _build_provider_for_quote(
        quote,
        spot=spot,
        project=project,
        subscription_id=subscription_id,
        resource_group=resource_group,
        artifact_account=artifact_account,
        extra_kwargs=provider_kwargs or None,
    )
    # Tag the provider with the quote so the run summary can compute
    # cost = quote * duration without re-querying pricing.
    try:
        provider._router_quote_hourly_usd = quote.hourly_usd  # type: ignore[attr-defined]
        provider._router_quote_label = (  # type: ignore[attr-defined]
            f"{quote.provider}/{quote.region} {quote.instance}"
        )
    except Exception:  # pragma: no cover - exotic Provider subclasses
        pass
    return provider


__all__ = ["SUPPORTED_PROVIDERS", "Auto", "AutoRouterError"]
