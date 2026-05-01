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
