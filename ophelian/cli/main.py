"""Typer-based CLI for the Ophelian framework."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console

from ophelian._version import __version__
from ophelian.core import Pipeline

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Ophelian — declarative ML framework. Run pipelines anywhere.",
)
console = Console()


def _load_pipeline(path: Path, attribute: str | None = None) -> Pipeline:
    """Import `path` as a Python module and locate a `Pipeline` instance."""
    if not path.exists():
        raise typer.BadParameter(f"Pipeline file not found: {path}")
    spec = importlib.util.spec_from_file_location("ophelian_pipeline_module", path)
    if spec is None or spec.loader is None:
        raise typer.BadParameter(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    if attribute is not None:
        candidate: Any = getattr(module, attribute, None)
        if not isinstance(candidate, Pipeline):
            raise typer.BadParameter(f"Attribute {attribute!r} in {path} is not a Pipeline")
        return candidate

    for value in vars(module).values():
        if isinstance(value, Pipeline):
            return value
    raise typer.BadParameter(
        f"No `Pipeline` instance found in {path}. Define one at module level or pass --attribute."
    )


@app.command()
def version() -> None:
    """Print the installed Ophelian version."""
    console.print(f"[bold]ophelian[/bold] {__version__}")


@app.command(name="dry-run")
def dry_run(
    pipeline_file: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    attribute: Annotated[str | None, typer.Option("--attribute", "-a")] = None,
) -> None:
    """Compile and print the execution plan without running anything."""
    pipeline = _load_pipeline(pipeline_file, attribute)
    pipeline.dry_run()


@app.command()
def run(
    pipeline_file: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    attribute: Annotated[str | None, typer.Option("--attribute", "-a")] = None,
    serve: Annotated[
        bool,
        typer.Option(
            "--serve/--no-serve",
            help="Start uvicorn for any Deploy step instead of just building the app.",
        ),
    ] = False,
) -> None:
    """Execute a pipeline locally with `Standalone(local=True)`.

    Loads the first `Pipeline` defined in `pipeline_file` (or the one named via
    `--attribute`), runs it on the standalone provider, and prints a per-step
    status table. With `--serve`, every Deploy step also starts a uvicorn server.
    """
    from ophelian.envs import Standalone

    pipeline = _load_pipeline(pipeline_file, attribute)
    env = Standalone(local=True, serve_deploys=serve)
    mode = getattr(env, "mode", "inprocess")
    console.print(f"[bold]Running pipeline:[/bold] {pipeline.name} (mode={mode})")
    result = pipeline.run(env=env)
    _render_result(result)
    if not result.succeeded:
        raise typer.Exit(code=1)


@app.command()
def costs(
    since: Annotated[
        str | None,
        typer.Option(
            "--since",
            help="Only include rows on/after this ISO timestamp (e.g. 2026-01-01).",
        ),
    ] = None,
    until: Annotated[
        str | None,
        typer.Option(
            "--until",
            help="Only include rows strictly before this ISO timestamp.",
        ),
    ] = None,
    by: Annotated[
        str | None,
        typer.Option(
            "--by",
            help=(
                "Group rows by a built-in field (provider, env_class, "
                "region, gpu_type, instance, pipeline, status) or any "
                "key inside the row's `context` dict."
            ),
        ),
    ] = None,
    output_format: Annotated[
        str,
        typer.Option(
            "--format",
            "-f",
            help="Output format: table (default), markdown, json, csv.",
        ),
    ] = "table",
    path: Annotated[
        Path | None,
        typer.Option(
            "--path",
            help="Override the ledger path (defaults to OPHELIAN_LEDGER_PATH or ~/.ophelian/ledger.jsonl).",
        ),
    ] = None,
) -> None:
    """Showback report over the local cost ledger.

    The ledger is appended to on every terminal pipeline run. Use
    ``--by team`` (or any other context key your pipelines stash via
    ``Pipeline(context={...})``) to aggregate spend by tenant /
    project / cost center.
    """
    from ophelian.observability.ledger import ledger_path, read_rows

    rows = read_rows(path)
    rows = _filter_rows(rows, since=since, until=until)
    if by:
        groups = _group_rows(rows, by)
        _emit_groups(groups, by=by, output_format=output_format)
    else:
        _emit_rows(rows, output_format=output_format)
    if not rows and output_format == "table":
        console.print(f"[dim]Ledger empty: {path or ledger_path()}[/]")


def _parse_when(value: str | None) -> float | None:
    if value is None:
        return None
    import datetime as _dt

    # Accept either a Unix timestamp or an ISO8601 date/time.
    try:
        return float(value)
    except ValueError:
        pass
    try:
        # ``fromisoformat`` accepts "2026-01-01" and "2026-01-01T12:34:56".
        dt = _dt.datetime.fromisoformat(value)
    except ValueError as exc:
        raise typer.BadParameter(
            f"Could not parse {value!r} as ISO timestamp or epoch seconds"
        ) from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_dt.UTC)
    return dt.timestamp()


def _filter_rows(
    rows: list[dict[str, Any]],
    *,
    since: str | None,
    until: str | None,
) -> list[dict[str, Any]]:
    lo = _parse_when(since)
    hi = _parse_when(until)
    out: list[dict[str, Any]] = []
    for row in rows:
        ts = row.get("timestamp")
        if not isinstance(ts, (int, float)):
            continue
        if lo is not None and ts < lo:
            continue
        if hi is not None and ts >= hi:
            continue
        out.append(row)
    return out


_BUILTIN_GROUP_KEYS = {
    "provider",
    "env_class",
    "region",
    "gpu_type",
    "instance",
    "pipeline",
    "status",
    "run_id",
}


def _group_rows(rows: list[dict[str, Any]], by: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = row.get(by) if by in _BUILTIN_GROUP_KEYS else (row.get("context") or {}).get(by)
        key_str = str(key) if key is not None else "(unset)"
        bucket = groups.setdefault(
            key_str,
            {"runs": 0, "hours": 0.0, "actual_usd": 0.0, "estimated_usd": 0.0},
        )
        bucket["runs"] += 1
        bucket["hours"] += float(row.get("hours") or 0.0)
        bucket["actual_usd"] += float(row.get("actual_usd") or 0.0)
        bucket["estimated_usd"] += float(row.get("estimated_usd") or 0.0)
    return groups


def _emit_rows(rows: list[dict[str, Any]], *, output_format: str) -> None:
    if output_format == "json":
        import json as _json

        typer.echo(_json.dumps(rows, indent=2, default=str))
        return
    if output_format == "csv":
        import csv as _csv
        import io as _io

        if not rows:
            typer.echo("")
            return
        # Stable column order: built-ins first, then any extra keys.
        builtins = [
            "timestamp",
            "run_id",
            "pipeline",
            "env_class",
            "provider",
            "region",
            "gpu_type",
            "instance",
            "hours",
            "hourly_usd",
            "estimated_usd",
            "actual_usd",
            "status",
            "schema_version",
            "context",
        ]
        buf = _io.StringIO()
        writer = _csv.DictWriter(buf, fieldnames=builtins, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _csv_value(row.get(k)) for k in builtins})
        typer.echo(buf.getvalue().rstrip("\n"))
        return
    if output_format == "markdown":
        headers = ["timestamp", "pipeline", "provider", "region", "status", "hours", "actual_usd"]
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
        ]
        for row in rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        _format_ts(row.get("timestamp")),
                        str(row.get("pipeline", "")),
                        str(row.get("provider", "")),
                        str(row.get("region", "")),
                        str(row.get("status", "")),
                        f"{float(row.get('hours') or 0.0):.4f}",
                        _format_usd(row.get("actual_usd")),
                    ]
                )
                + " |"
            )
        typer.echo("\n".join(lines))
        return
    # default: rich table
    from rich.table import Table

    table = Table(title="Ophelian cost ledger")
    for col in (
        "Timestamp",
        "Pipeline",
        "Provider",
        "Region",
        "GPU",
        "Status",
        "Hours",
        "Actual (USD)",
    ):
        table.add_column(col)
    for row in rows:
        table.add_row(
            _format_ts(row.get("timestamp")),
            str(row.get("pipeline", "")),
            str(row.get("provider", "")),
            str(row.get("region", "")),
            str(row.get("gpu_type") or "-"),
            str(row.get("status", "")),
            f"{float(row.get('hours') or 0.0):.4f}",
            _format_usd(row.get("actual_usd")),
        )
    console.print(table)


def _emit_groups(
    groups: dict[str, dict[str, Any]],
    *,
    by: str,
    output_format: str,
) -> None:
    flat = [
        {by: key, **values}
        for key, values in sorted(groups.items(), key=lambda kv: -kv[1]["actual_usd"])
    ]
    if output_format == "json":
        import json as _json

        typer.echo(_json.dumps(flat, indent=2, default=str))
        return
    if output_format == "csv":
        import csv as _csv
        import io as _io

        cols = [by, "runs", "hours", "estimated_usd", "actual_usd"]
        buf = _io.StringIO()
        writer = _csv.DictWriter(buf, fieldnames=cols)
        writer.writeheader()
        for row in flat:
            writer.writerow({k: row.get(k) for k in cols})
        typer.echo(buf.getvalue().rstrip("\n"))
        return
    if output_format == "markdown":
        headers = [by, "runs", "hours", "actual_usd"]
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
        ]
        for row in flat:
            lines.append(
                f"| {row[by]} | {row['runs']} | {row['hours']:.4f} | {_format_usd(row['actual_usd'])} |"
            )
        typer.echo("\n".join(lines))
        return
    from rich.table import Table

    table = Table(title=f"Ophelian costs by {by}")
    for col in (by.title(), "Runs", "Hours", "Actual (USD)"):
        table.add_column(col)
    for row in flat:
        table.add_row(
            str(row[by]),
            str(row["runs"]),
            f"{row['hours']:.4f}",
            _format_usd(row["actual_usd"]),
        )
    console.print(table)


def _format_ts(value: Any) -> str:
    import datetime as _dt

    if not isinstance(value, (int, float)):
        return str(value or "")
    return _dt.datetime.fromtimestamp(value, tz=_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _format_usd(value: Any) -> str:
    if value is None:
        return "-"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return "-"
    return f"${f:.4f}" if f < 1 else f"${f:.2f}"


def _csv_value(value: Any) -> Any:
    import json as _json

    if isinstance(value, (dict, list)):
        return _json.dumps(value, default=str)
    return value


def _render_result(result: Any) -> None:
    from rich.table import Table

    table = Table(title=f"Pipeline result — {result.pipeline}")
    table.add_column("Step")
    table.add_column("Kind")
    table.add_column("Status")
    table.add_column("Details")
    for step in result.steps:
        details = step.error or ", ".join(f"{k}={v}" for k, v in step.metrics.items()) or "ok"
        table.add_row(step.name, step.kind, step.status, str(details))
    console.print(table)


def main() -> None:  # pragma: no cover — wrapper used by the console_script
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
