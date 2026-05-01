"""FastAPI inference runtime.

`build_app` returns a FastAPI app bound to a `ModelAdapter`, exposing
`/health` and `/predict`. Heavier runtimes (Triton, BentoML, Ray Serve) will
plug in here in later releases.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI

from ophelian.models import registry
from ophelian.models.base import ModelAdapter


def build_app(*, framework: str, model_path: str | Path) -> FastAPI:
    """Construct a FastAPI app that serves the model under `model_path`."""
    adapter_cls = registry.get(framework)
    adapter: ModelAdapter = adapter_cls()
    model = adapter.load(Path(model_path))
    app = FastAPI(title=f"ophelian-inference[{framework}]", version="0.1.0")

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
