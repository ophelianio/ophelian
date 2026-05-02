"""Quote and router-decision dataclasses for the pricing subsystem.

These are the value types every other module in :mod:`ophelian.pricing`
returns or consumes:

- :class:`PriceQuote` — one ``(provider, region, instance)`` price entry
  produced either by the static fallback table or by a live fetch.
- :class:`RouterDecision` — what :class:`Auto` ended up choosing,
  together with the runners-up considered.
- :func:`explain` — pretty-print a :class:`RouterDecision` for
  ``ophelian … --explain`` output.

These types have no I/O and no third-party dependencies on purpose so
they can be imported anywhere (including by tests) without dragging in
boto3, requests, or the cache module.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PriceQuote:
    """A single (provider, region, instance) on-demand or spot quote.

    ``hourly_usd`` is the per-hour billing rate; ``spot`` indicates
    whether the price is the spot/preemptible discount or the
    on-demand rate. ``gpu_family`` is the user-facing label
    (``"A100"``, ``"H100"``, ...) — *not* the SKU-specific name (e.g.
    ``"nvidia-tesla-a100"`` on GCP) which lives in ``instance``.
    """

    provider: str
    region: str
    instance: str
    gpu_family: str
    gpu_count: int
    hourly_usd: float
    spot: bool = False
    source: str = "static"
    notes: str = ""

    def annual_cost(self, hours: float = 24 * 30) -> float:
        return self.hourly_usd * hours


@dataclass
class RouterDecision:
    """The chosen :class:`PriceQuote` plus context for ``--explain`` output."""

    quote: PriceQuote
    considered: list[PriceQuote] = field(default_factory=list)

    @property
    def savings_vs_most_expensive(self) -> float:
        if not self.considered:
            return 0.0
        worst = max(self.considered, key=lambda q: q.hourly_usd).hourly_usd
        return max(0.0, worst - self.quote.hourly_usd)


def explain(decision: RouterDecision) -> str:
    """Render a short, copy-pasteable explanation of a router decision."""
    q = decision.quote
    lines = [
        f"Cheapest {q.gpu_family}: {q.provider}/{q.region} on {q.instance}",
        f"  {q.hourly_usd:.3f} USD/hour ({'spot' if q.spot else 'on-demand'}, source={q.source})",
    ]
    if decision.considered:
        runners_up = [
            qq for qq in sorted(decision.considered, key=lambda q: q.hourly_usd)[:5] if qq is not q
        ]
        if runners_up:
            lines.append("  Runners-up:")
            for qq in runners_up:
                lines.append(
                    f"    - {qq.provider}/{qq.region} {qq.instance:>30}  {qq.hourly_usd:.3f} USD/h"
                )
    return "\n".join(lines)


__all__ = ["PriceQuote", "RouterDecision", "explain"]
