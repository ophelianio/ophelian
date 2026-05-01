"""Tests for the observability surface — run_id, JSON logger, summary."""

from __future__ import annotations

import io
import json
import logging

import pytest
from ophelian.core.nodes import StepResult
from ophelian.observability import (
    JSONFormatter,
    bind_run,
    configure_logging,
    emit_event,
    get_run_id,
    set_run_id,
)
from ophelian.observability.summary import emit_run_summary

# ---------------------------------------------------------------------------
# run_id ContextVar
# ---------------------------------------------------------------------------


def test_get_run_id_starts_unset() -> None:
    set_run_id(None)
    assert get_run_id() is None


def test_bind_run_sets_and_resets() -> None:
    set_run_id(None)
    with bind_run("run-alpha"):
        assert get_run_id() == "run-alpha"
        with bind_run("run-beta"):
            assert get_run_id() == "run-beta"
        assert get_run_id() == "run-alpha"
    assert get_run_id() is None


def test_set_run_id_persists_until_overwritten() -> None:
    set_run_id("run-sticky")
    try:
        assert get_run_id() == "run-sticky"
    finally:
        set_run_id(None)


# ---------------------------------------------------------------------------
# JSON formatter
# ---------------------------------------------------------------------------


def test_json_formatter_emits_required_keys() -> None:
    record = logging.LogRecord(
        name="ophelian.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="hello %s",
        args=("world",),
        exc_info=None,
    )
    line = JSONFormatter().format(record)
    payload = json.loads(line)
    assert payload["level"] == "INFO"
    assert payload["logger"] == "ophelian.test"
    assert payload["msg"] == "hello world"
    assert "ts" in payload


def test_json_formatter_includes_run_id_when_bound() -> None:
    record = logging.LogRecord(
        name="ophelian.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="x",
        args=(),
        exc_info=None,
    )
    with bind_run("run-correlated"):
        payload = json.loads(JSONFormatter().format(record))
    assert payload["run_id"] == "run-correlated"


def test_json_formatter_carries_extra_fields() -> None:
    record = logging.LogRecord(
        name="ophelian.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="x",
        args=(),
        exc_info=None,
    )
    record.step = "trainer"
    record.duration_s = 1.25
    payload = json.loads(JSONFormatter().format(record))
    assert payload["extra"]["step"] == "trainer"
    assert payload["extra"]["duration_s"] == 1.25


def test_json_formatter_repr_falls_back_for_unserialisable() -> None:
    class _NotJsonable:
        def __repr__(self) -> str:
            return "<thing>"

    record = logging.LogRecord(
        name="ophelian.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="x",
        args=(),
        exc_info=None,
    )
    record.thing = _NotJsonable()
    payload = json.loads(JSONFormatter().format(record))
    assert payload["extra"]["thing"] == "<thing>"


# ---------------------------------------------------------------------------
# configure_logging
# ---------------------------------------------------------------------------


def test_configure_logging_replaces_handler_each_call() -> None:
    log = configure_logging(json=False)
    first_handlers = list(log.handlers)
    log = configure_logging(json=True)
    second_handlers = list(log.handlers)
    assert len(first_handlers) == 1
    assert len(second_handlers) == 1
    assert first_handlers[0] is not second_handlers[0]


def test_configure_logging_env_var_picks_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPHELIAN_LOG_FORMAT", "json")
    log = configure_logging()
    handler = log.handlers[0]
    assert isinstance(handler.formatter, JSONFormatter)


def test_configure_logging_emits_json_lines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = io.StringIO()
    log = configure_logging(json=True, stream=stream)
    with bind_run("run-emit"):
        log.info("step done", extra={"step": "trainer", "metric": 0.99})
    line = stream.getvalue().strip().splitlines()[-1]
    payload = json.loads(line)
    assert payload["msg"] == "step done"
    assert payload["run_id"] == "run-emit"
    assert payload["extra"]["step"] == "trainer"


def test_configure_logging_default_streams() -> None:
    """JSON logs go to stdout (parseable channel), human logs to stderr."""
    import sys

    json_log = configure_logging(json=True)
    assert json_log.handlers[0].stream is sys.stdout
    human_log = configure_logging(json=False)
    assert human_log.handlers[0].stream is sys.stderr


# ---------------------------------------------------------------------------
# OTel hook is silent when not installed
# ---------------------------------------------------------------------------


def test_emit_event_is_a_silent_noop_without_otel() -> None:
    # Installing opentelemetry isn't part of the default test deps,
    # so the only contract is "doesn't raise".
    emit_event("something", run_id="run-x", duration_s=1.0)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def _step(name: str, status: str = "success", metrics: dict | None = None) -> StepResult:
    return StepResult(
        name=name,
        kind="train",
        status=status,  # type: ignore[arg-type]
        metrics=metrics or {},
        artifacts={"model": f"s3://b/{name}"},
    )


def test_emit_run_summary_human_writes_to_stream() -> None:
    stream = io.StringIO()
    emit_run_summary(
        provider="aws",
        run_id="run-1",
        pipeline="demo",
        steps=[_step("trainer", metrics={"accuracy": 0.91})],
        stream=stream,
    )
    body = stream.getvalue()
    assert "trainer" in body
    assert "demo" in body
    assert "run-1" in body


def test_emit_run_summary_json_mode_emits_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPHELIAN_LOG_FORMAT", "json")

    captured: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append(record)

    summary_logger = logging.getLogger("ophelian.observability.summary")
    handler = _Capture(level=logging.INFO)
    summary_logger.addHandler(handler)
    previous_level = summary_logger.level
    summary_logger.setLevel(logging.INFO)
    try:
        emit_run_summary(
            provider="gcp",
            run_id="run-2",
            pipeline="demo",
            steps=[_step("ds"), _step("trainer", metrics={"accuracy": 0.5})],
        )
    finally:
        summary_logger.removeHandler(handler)
        summary_logger.setLevel(previous_level)

    records = [r for r in captured if r.msg == "run.summary"]
    assert records, "summary record was not emitted"
    record = records[-1]
    assert record.run_id == "run-2"  # type: ignore[attr-defined]
    assert record.provider == "gcp"  # type: ignore[attr-defined]
    assert {s["name"] for s in record.steps} == {"ds", "trainer"}  # type: ignore[attr-defined]


def test_emit_run_summary_skips_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPHELIAN_NO_SUMMARY", "1")
    stream = io.StringIO()
    emit_run_summary(
        provider="aws",
        run_id="r",
        pipeline="p",
        steps=[_step("trainer")],
        stream=stream,
    )
    assert stream.getvalue() == ""


def test_emit_run_summary_json_includes_duration_and_cost_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """duration_seconds + cost_estimate_usd + gpu_utilization must round-trip."""
    monkeypatch.setenv("OPHELIAN_LOG_FORMAT", "json")
    from ophelian.core.nodes import StepResult

    captured: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append(record)

    summary_logger = logging.getLogger("ophelian.observability.summary")
    handler = _Capture(level=logging.INFO)
    summary_logger.addHandler(handler)
    previous_level = summary_logger.level
    summary_logger.setLevel(logging.INFO)
    try:
        step = StepResult(
            name="trainer",
            kind="train",
            status="success",
            metrics={"accuracy": 0.9},
            artifacts={"model": "s3://b/m.pkl"},
            duration_seconds=120.0,
        )
        emit_run_summary(
            provider="aws",
            run_id="run-3",
            pipeline="demo",
            steps=[step],
            hourly_usd=12.0,  # router quote
        )
    finally:
        summary_logger.removeHandler(handler)
        summary_logger.setLevel(previous_level)

    records = [r for r in captured if r.msg == "run.summary"]
    assert records
    payload = records[-1].steps[0]  # type: ignore[attr-defined]
    assert payload["duration_seconds"] == 120.0
    assert payload["cost_estimate_usd"] == pytest.approx(12.0 * (120.0 / 3600.0))
    assert payload["gpu_utilization"] is None
    assert records[-1].hourly_usd == 12.0  # type: ignore[attr-defined]


def test_emit_run_summary_human_renders_duration_and_cost_columns() -> None:
    from ophelian.core.nodes import StepResult

    stream = io.StringIO()
    step = StepResult(
        name="trainer",
        kind="train",
        status="success",
        metrics={"accuracy": 0.9},
        duration_seconds=42.0,
    )
    emit_run_summary(
        provider="aws",
        run_id="run-4",
        pipeline="demo",
        steps=[step],
        stream=stream,
        hourly_usd=3.6,
    )
    body = stream.getvalue()
    assert "Duration" in body or "DURATION" in body
    assert "Cost" in body or "COST" in body
    assert "GPU util" in body
