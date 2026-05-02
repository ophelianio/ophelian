"""Shared pytest fixtures.

The OpenTelemetry SDK's ``set_tracer_provider`` / ``set_meter_provider``
are one-shot by design — calling them a second time logs a warning and
keeps the original provider. Each test module that wants to assert on
spans / metrics therefore has to share the same in-memory providers.

This module installs them once per test session and yields the
in-memory exporters so individual tests can ``clear()`` between runs.
``OPHELIAN_OTEL_DISABLE=1`` is set so ``auto_configure_from_env`` sees
that we already own the providers and stays out of the way.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

import pytest


@pytest.fixture(scope="session")
def _otel_in_memory_providers() -> Iterator[dict[str, Any]]:
    os.environ["OPHELIAN_OTEL_DISABLE"] = "1"

    from opentelemetry import metrics, trace
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )
    from ophelian.observability.otel import _reset_auto_configuration_for_tests

    span_exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    metric_reader = InMemoryMetricReader()
    meter_provider = MeterProvider(metric_readers=[metric_reader])
    trace.set_tracer_provider(tracer_provider)
    metrics.set_meter_provider(meter_provider)
    _reset_auto_configuration_for_tests()
    try:
        yield {"spans": span_exporter, "metrics": metric_reader}
    finally:
        os.environ.pop("OPHELIAN_OTEL_DISABLE", None)
