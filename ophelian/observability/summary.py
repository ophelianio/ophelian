"""Rich-table run summary printed at the end of every provider execute().

The table is the user-facing pay-off of a successful (or failed) run —
one glanceable view of every step, status, key metrics, and the run id
to copy/paste into a resume call. We deliberately keep the dependency
optional: if ``rich`` cannot be imported (e.g. when the user pinned an
older version) we degrade to a plain-text table so the contract still
holds.
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from ophelian.core.nodes import StepResult

logger = logging.getLogger("ophelian.observability.summary")


def emit_run_summary(
    *,
    provider: str,
    run_id: str,
    pipeline: str,
    steps: Iterable[StepResult],
    description: str | None = None,
    stream: Any | None = None,
    hourly_usd: float | None = None,
) -> None:
    """Render a one-shot summary table for a finished pipeline run.

    Honours ``OPHELIAN_NO_SUMMARY=1`` (skip entirely) and
    ``OPHELIAN_LOG_FORMAT=json`` (emit a structured ``run.summary``
    record on the ``ophelian.observability.summary`` logger instead of
    a human-readable table — useful in CI / log-aggregation pipelines).
    """
    if os.environ.get("OPHELIAN_NO_SUMMARY") in {"1", "true", "True"}:
        return

    rows = list(steps)

    if os.environ.get("OPHELIAN_LOG_FORMAT", "").lower() == "json":
        logger.info(
            "run.summary",
            extra={
                "provider": provider,
                "run_id": run_id,
                "pipeline": pipeline,
                "description": description,
                "hourly_usd": hourly_usd,
                "steps": [_step_payload(s, hourly_usd) for s in rows],
            },
        )
        return

    out = stream if stream is not None else sys.stderr
    try:
        _render_rich(provider, run_id, pipeline, description, rows, out, hourly_usd)
    except Exception:  # pragma: no cover - rich missing or terminal hostile
        _render_plain(provider, run_id, pipeline, description, rows, out, hourly_usd)


def _step_payload(
    step: StepResult, hourly_usd: float | None = None
) -> dict[str, Any]:
    duration = step.duration_seconds
    cost = _estimate_cost_usd(step, duration, hourly_usd)
    return {
        "name": step.name,
        "kind": step.kind,
        "status": step.status,
        "metrics": dict(step.metrics or {}),
        "artifacts": dict(step.artifacts or {}),
        "error": step.error,
        "duration_seconds": duration,
        "cost_estimate_usd": cost,
        "gpu_utilization": None,
    }


def _estimate_cost_usd(
    step: StepResult,
    duration: float | None,
    fallback_hourly: float | None = None,
) -> float | None:
    """Best-effort cost = (per-step or run-level hourly_usd) * duration.

    Returning ``None`` is honest when no quote is available — we never
    fabricate a number.
    """
    if duration is None:
        return None
    info = step.info or {}
    hourly = (
        info.get("router_quote_hourly_usd")
        or info.get("hourly_usd")
        or fallback_hourly
    )
    try:
        hourly_f = float(hourly) if hourly is not None else None
    except (TypeError, ValueError):
        return None
    if hourly_f is None:
        return None
    return round(hourly_f * (duration / 3600.0), 6)


def _render_rich(
    provider: str,
    run_id: str,
    pipeline: str,
    description: str | None,
    steps: list[StepResult],
    out: Any,
    hourly_usd: float | None = None,
) -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console(file=out, highlight=False, force_terminal=False)
    title = f"[bold]Ophelian run summary[/] — {pipeline}  ([dim]{run_id}[/])"
    if description:
        title += f"\n[dim]{description}[/]"
    table = Table(title=title, show_lines=False, header_style="bold")
    table.add_column("Step")
    table.add_column("Kind")
    table.add_column("Status")
    table.add_column("Duration")
    table.add_column("Cost (USD)")
    table.add_column("Metrics")
    table.add_column("Artifacts")
    total_cost = 0.0
    have_cost = False
    total_duration = 0.0
    for step in steps:
        status = step.status
        style = (
            "green"
            if status == "success"
            else "yellow"
            if status == "skipped"
            else "red"
        )
        duration = step.duration_seconds
        cost = _estimate_cost_usd(step, duration, hourly_usd)
        if duration is not None:
            total_duration += duration
        if cost is not None:
            total_cost += cost
            have_cost = True
        table.add_row(
            step.name,
            step.kind,
            f"[{style}]{status}[/]",
            _format_duration(duration),
            _format_cost(cost),
            _format_metrics(step.metrics or {}),
            _format_artifacts(step.artifacts or {}),
        )
    console.print(table)
    footer_bits = [f"total {_format_duration(total_duration)}"]
    if have_cost:
        footer_bits.append(f"≈ ${total_cost:.4f} (router quote * duration)")
    footer_bits.append(
        "GPU util: not collected — install an in-VM nvidia-smi sidecar to populate."
    )
    console.print("[dim]" + " · ".join(footer_bits) + "[/]")
    failed = [s for s in steps if s.status == "failed"]
    if failed:
        console.print(
            f"[red]✘ {len(failed)} step(s) failed.[/] Resume with "
            f"[bold]{provider.upper()}(... resume_run_id={run_id!r})[/]."
        )


def _render_plain(
    provider: str,
    run_id: str,
    pipeline: str,
    description: str | None,
    steps: list[StepResult],
    out: Any,
    hourly_usd: float | None = None,
) -> None:
    out.write(f"\nOphelian run summary — {pipeline}  ({run_id})\n")
    if description:
        out.write(f"  {description}\n")
    out.write(
        f"  {'STEP':<24} {'KIND':<8} {'STATUS':<10} {'DURATION':<10} "
        f"{'COST(USD)':<12} METRICS\n"
    )
    total_cost = 0.0
    have_cost = False
    total_duration = 0.0
    for step in steps:
        duration = step.duration_seconds
        cost = _estimate_cost_usd(step, duration, hourly_usd)
        if duration is not None:
            total_duration += duration
        if cost is not None:
            total_cost += cost
            have_cost = True
        out.write(
            f"  {step.name:<24} {step.kind:<8} {step.status:<10} "
            f"{_format_duration(duration):<10} {_format_cost(cost):<12} "
            f"{_format_metrics(step.metrics or {})}\n"
        )
    out.write(f"  total {_format_duration(total_duration)}")
    if have_cost:
        out.write(f"  ≈ ${total_cost:.4f} (router quote * duration)")
    out.write(
        "  GPU util: not collected — install an in-VM nvidia-smi sidecar.\n"
    )
    failed = [s for s in steps if s.status == "failed"]
    if failed:
        out.write(
            f"  {len(failed)} step(s) failed. Resume with "
            f"{provider.upper()}(... resume_run_id={run_id!r}).\n"
        )
    out.write("\n")


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m{secs:04.1f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h{int(minutes):02d}m"


def _format_cost(cost: float | None) -> str:
    if cost is None:
        return "-"
    if cost < 0.01:
        return f"${cost:.4f}"
    return f"${cost:.3f}"


def _format_metrics(metrics: dict[str, Any]) -> str:
    if not metrics:
        return "-"
    parts: list[str] = []
    for key, value in list(metrics.items())[:4]:
        if isinstance(value, float):
            parts.append(f"{key}={value:.4g}")
        else:
            parts.append(f"{key}={value}")
    if len(metrics) > 4:
        parts.append(f"+{len(metrics) - 4} more")
    return ", ".join(parts)


def _format_artifacts(artifacts: dict[str, Any]) -> str:
    if not artifacts:
        return "-"
    keys = list(artifacts.keys())
    head = ", ".join(keys[:3])
    if len(keys) > 3:
        head += f" +{len(keys) - 3}"
    return head


__all__ = ["emit_run_summary"]
