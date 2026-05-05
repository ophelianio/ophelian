"""Tests for the cost ledger and ``ophelian costs`` CLI (Task #28)."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from ophelian import Data, Pipeline, Standalone
from ophelian.cli import app
from ophelian.observability.ledger import (
    SCHEMA_VERSION,
    LedgerRow,
    append_row,
    emit_run_row,
    read_rows,
)
from typer.testing import CliRunner

runner = CliRunner()


@pytest.fixture()
def ledger_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    target = tmp_path / "ledger.jsonl"
    monkeypatch.setenv("OPHELIAN_LEDGER_PATH", str(target))
    monkeypatch.delenv("OPHELIAN_LEDGER_DISABLED", raising=False)
    yield target


def _toy_pipeline(*, context: dict[str, Any] | None = None, name: str = "ledger-test") -> Pipeline:
    kwargs: dict[str, Any] = {"name": name}
    if context is not None:
        kwargs["context"] = context
    return Pipeline(
        [
            Data(
                name="ds",
                source="memory://toy",
                format="inline",
                options={"X": [[1.0, 2.0]], "y": [0, 1]},
            )
        ],
        **kwargs,
    )


# ----------------------------------------------------------------------
# Schema + writer
# ----------------------------------------------------------------------


def test_append_and_read_roundtrip(ledger_file: Path) -> None:
    row = LedgerRow(
        schema_version=SCHEMA_VERSION,
        run_id="r1",
        pipeline="p",
        timestamp=1700000000.0,
        env_class="standalone",
        provider="standalone",
        region="local",
        gpu_type=None,
        instance=None,
        hours=0.5,
        hourly_usd=2.0,
        estimated_usd=1.0,
        actual_usd=1.0,
        status="success",
        context={"team": "ml"},
    )
    append_row(row)
    rows = read_rows()
    assert len(rows) == 1
    assert rows[0]["run_id"] == "r1"
    assert rows[0]["schema_version"] == SCHEMA_VERSION
    assert rows[0]["context"] == {"team": "ml"}


def test_emit_run_row_disabled(ledger_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPHELIAN_LEDGER_DISABLED", "1")
    emit_run_row(
        pipeline_name="p",
        run_id="r1",
        env_class="standalone",
        provider="standalone",
        region="local",
        gpu_type=None,
        instance=None,
        hours=0.1,
        hourly_usd=1.0,
        status="success",
        context=None,
    )
    assert read_rows() == []


# ----------------------------------------------------------------------
# Pipeline integration: success + failure paths
# ----------------------------------------------------------------------


def test_success_run_appends_row(ledger_file: Path) -> None:
    _toy_pipeline().run(env=Standalone(local=True, container=False))
    rows = read_rows()
    assert len(rows) == 1
    assert rows[0]["status"] == "success"
    assert rows[0]["pipeline"] == "ledger-test"
    assert rows[0]["env_class"] == "standalone"
    assert rows[0]["hours"] >= 0.0


def test_failed_run_appends_row_with_status_failed(
    ledger_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Force the in-process Data handler to raise; the ledger row
    must be written with ``status=failed`` and the duration accrued
    so far."""
    from ophelian.providers.standalone import StandaloneProvider

    def boom(self: Any, *_a: Any, **_kw: Any) -> Any:
        raise RuntimeError("synthetic step failure")

    monkeypatch.setattr(StandaloneProvider, "_handle_data", boom)
    _toy_pipeline(name="ledger-fail").run(env=Standalone(local=True, container=False))
    rows = read_rows()
    assert len(rows) == 1
    assert rows[0]["status"] == "failed"
    assert rows[0]["pipeline"] == "ledger-fail"


def test_context_propagates_to_ledger(ledger_file: Path) -> None:
    ctx = {"team": "ml", "tenant_id": "acme"}
    _toy_pipeline(context=ctx).run(env=Standalone(local=True, container=False))
    rows = read_rows()
    assert len(rows) == 1
    assert rows[0]["context"] == ctx


def test_router_quote_threads_into_actual_usd(ledger_file: Path) -> None:
    """When the env carries a router quote, ``actual_usd`` is
    populated as ``hourly_usd * hours``."""
    env = Standalone(local=True, container=False)
    env._router_quote_hourly_usd = 3.6  # type: ignore[attr-defined]
    _toy_pipeline().run(env=env)
    rows = read_rows()
    assert rows[0]["hourly_usd"] == pytest.approx(3.6)
    if rows[0]["hours"] > 0:
        assert rows[0]["actual_usd"] == pytest.approx(3.6 * rows[0]["hours"], rel=1e-3)


# ----------------------------------------------------------------------
# Concurrent-write safety
# ----------------------------------------------------------------------


def test_concurrent_appends_dont_interleave(ledger_file: Path) -> None:
    """Many threads write rows in parallel — every line must parse as
    valid JSON (no torn writes)."""

    def worker(i: int) -> None:
        for j in range(20):
            row = LedgerRow(
                schema_version=SCHEMA_VERSION,
                run_id=f"r-{i}-{j}",
                pipeline="p",
                timestamp=1700000000.0 + i + j * 0.001,
                env_class="standalone",
                provider="standalone",
                region="local",
                gpu_type=None,
                instance=None,
                hours=0.01,
                hourly_usd=1.0,
                estimated_usd=0.01,
                actual_usd=0.01,
                status="success",
                context={"thread": i},
            )
            append_row(row)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    raw = ledger_file.read_text().splitlines()
    assert len(raw) == 8 * 20
    for line in raw:
        json.loads(line)  # raises if any row was torn


# ----------------------------------------------------------------------
# CLI: filters, grouping, output formats
# ----------------------------------------------------------------------


def _seed(ledger_file: Path) -> None:
    base = 1700000000.0
    rows = [
        LedgerRow(
            schema_version=SCHEMA_VERSION,
            run_id="r1",
            pipeline="alpha",
            timestamp=base,
            env_class="aws",
            provider="aws",
            region="us-east-1",
            gpu_type="A100",
            instance="p4d.24xlarge",
            hours=1.0,
            hourly_usd=4.0,
            estimated_usd=4.0,
            actual_usd=4.0,
            status="success",
            context={"team": "ml"},
        ),
        LedgerRow(
            schema_version=SCHEMA_VERSION,
            run_id="r2",
            pipeline="beta",
            timestamp=base + 86400,  # +1 day
            env_class="gcp",
            provider="gcp",
            region="us-central1",
            gpu_type="H100",
            instance="a3-highgpu-8g",
            hours=2.0,
            hourly_usd=10.0,
            estimated_usd=20.0,
            actual_usd=20.0,
            status="failed",
            context={"team": "ml"},
        ),
        LedgerRow(
            schema_version=SCHEMA_VERSION,
            run_id="r3",
            pipeline="gamma",
            timestamp=base + 86400 * 7,  # +7 days
            env_class="standalone",
            provider="standalone",
            region="local",
            gpu_type=None,
            instance=None,
            hours=0.5,
            hourly_usd=None,
            estimated_usd=None,
            actual_usd=None,
            status="success",
            context={"team": "ops"},
        ),
    ]
    for row in rows:
        append_row(row)


def test_cli_default_table_lists_all_rows(ledger_file: Path) -> None:
    _seed(ledger_file)
    result = runner.invoke(app, ["costs"])
    assert result.exit_code == 0, result.stdout
    for needle in ("alpha", "beta", "gamma"):
        assert needle in result.stdout


def test_cli_filter_since(ledger_file: Path) -> None:
    _seed(ledger_file)
    # base + 86400 keeps r2 + r3 only
    import datetime as _dt

    cutoff = _dt.datetime.fromtimestamp(1700000000.0 + 86400, tz=_dt.UTC).strftime(
        "%Y-%m-%dT%H:%M:%S"
    )
    result = runner.invoke(app, ["costs", "--since", cutoff, "--format", "json"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    pipelines = {r["pipeline"] for r in payload}
    assert pipelines == {"beta", "gamma"}


def test_cli_filter_until(ledger_file: Path) -> None:
    _seed(ledger_file)
    # epoch-seconds form is also accepted
    result = runner.invoke(app, ["costs", "--until", str(1700000000.0 + 1), "--format", "json"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert {r["pipeline"] for r in payload} == {"alpha"}


def test_cli_group_by_context_key(ledger_file: Path) -> None:
    _seed(ledger_file)
    result = runner.invoke(app, ["costs", "--by", "team", "--format", "json"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    by_team = {r["team"]: r for r in payload}
    assert set(by_team) == {"ml", "ops"}
    assert by_team["ml"]["runs"] == 2
    assert by_team["ml"]["actual_usd"] == pytest.approx(24.0)
    assert by_team["ops"]["runs"] == 1
    assert by_team["ops"]["actual_usd"] == 0.0


def test_cli_group_by_builtin(ledger_file: Path) -> None:
    _seed(ledger_file)
    result = runner.invoke(app, ["costs", "--by", "provider", "--format", "csv"])
    assert result.exit_code == 0, result.stdout
    lines = result.stdout.strip().splitlines()
    header = lines[0].split(",")
    assert header[0] == "provider"
    body = [line.split(",") for line in lines[1:]]
    providers = {row[0] for row in body}
    assert providers == {"aws", "gcp", "standalone"}


def test_cli_format_markdown(ledger_file: Path) -> None:
    _seed(ledger_file)
    result = runner.invoke(app, ["costs", "--format", "markdown"])
    assert result.exit_code == 0, result.stdout
    assert "| timestamp |" in result.stdout
    assert "alpha" in result.stdout


def test_cli_empty_ledger_default_format(ledger_file: Path) -> None:
    result = runner.invoke(app, ["costs"])
    assert result.exit_code == 0, result.stdout
    assert "Ledger empty" in result.stdout


def test_cli_bad_since_raises(ledger_file: Path) -> None:
    _seed(ledger_file)
    result = runner.invoke(app, ["costs", "--since", "not-a-date"])
    assert result.exit_code != 0
