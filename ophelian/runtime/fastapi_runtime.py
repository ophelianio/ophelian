"""FastAPI inference runtime.

`build_app` returns a FastAPI app bound to a `ModelAdapter`, exposing
`/health` and `/predict`. Heavier runtimes (Triton, BentoML, Ray Serve) will
plug in here in later releases.

Every served request is wrapped in an OpenTelemetry span and contributes
to the ``ophelian.serve.requests`` counter and ``ophelian.serve.latency``
histogram. When the ``[otel]`` extra is not installed, the helpers are
silent no-ops and the app behaves exactly as before.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, Response

from ophelian.models import registry
from ophelian.models.base import ModelAdapter
from ophelian.observability.otel import (
    ATTR_FRAMEWORK,
    record_serve_outcome,
    serve_request_span,
)


def _install_otel_middleware(app: FastAPI, *, framework: str) -> None:
    """Wrap every request in an Ophelian serve span + metrics."""

    @app.middleware("http")
    async def _otel_middleware(request: Request, call_next: Any) -> Response:
        method = request.method
        # Prefer the route template (e.g. ``/predict``) over the raw
        # path so the metric cardinality stays bounded; fall back to
        # the raw URL path if the route is unknown (404, etc.).
        route_obj = request.scope.get("route")
        route = getattr(route_obj, "path", None) or request.url.path
        started = time.monotonic()
        with serve_request_span(method=method, route=route, framework=framework) as span:
            try:
                response: Response = await call_next(request)
            except Exception:
                record_serve_outcome(
                    method=method,
                    route=route,
                    status_code=500,
                    duration_seconds=time.monotonic() - started,
                    span=span,
                )
                raise
            record_serve_outcome(
                method=method,
                route=route,
                status_code=response.status_code,
                duration_seconds=time.monotonic() - started,
                span=span,
            )
            return response


def build_app(*, framework: str, model_path: str | Path) -> FastAPI:
    """Construct a FastAPI app that serves the model under `model_path`."""
    adapter_cls = registry.get(framework)
    adapter: ModelAdapter = adapter_cls()
    model = adapter.load(Path(model_path))
    app = FastAPI(title=f"ophelian-inference[{framework}]", version="0.1.0")
    _install_otel_middleware(app, framework=framework)

    # Expose the framework name as a span attribute on every span the
    # caller creates inside a request handler — useful for downstream
    # consumers that fan out by ML framework.
    app.state.ophelian_attrs = {ATTR_FRAMEWORK: framework}

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "framework": framework, "model": str(model_path)}

    @app.post("/predict")
    def predict(payload: dict[str, Any]) -> dict[str, Any]:
        prediction = adapter.predict(model, payload.get("inputs"))
        return {"prediction": prediction}

    return app


def app_from_env() -> FastAPI:
    """Zero-arg factory used by ``uvicorn --factory`` inside containers.

    Reads ``OPHELIAN_FRAMEWORK`` and ``OPHELIAN_MODEL_PATH`` from the
    environment so the Standalone provider can launch::

        python -m uvicorn --factory ophelian.runtime.fastapi_runtime:app_from_env

    inside the deploy container without having to inject keyword arguments.
    """
    framework = os.environ.get("OPHELIAN_FRAMEWORK")
    model_path = os.environ.get("OPHELIAN_MODEL_PATH")
    if not framework or not model_path:
        raise RuntimeError(
            "app_from_env requires OPHELIAN_FRAMEWORK and OPHELIAN_MODEL_PATH "
            "environment variables to be set."
        )
    return build_app(framework=framework, model_path=model_path)


class FastAPIRuntime:
    """Convenience wrapper used by the standalone provider's deploy step."""

    def __init__(self, *, framework: str, model_path: str | Path) -> None:
        self.framework = framework
        self.model_path = Path(model_path)

    def app(self) -> FastAPI:
        return build_app(framework=self.framework, model_path=self.model_path)

    def serve(self, *, host: str = "0.0.0.0", port: int = 8000) -> None:  # pragma: no cover
        import uvicorn

        uvicorn.run(self.app(), host=host, port=port)
