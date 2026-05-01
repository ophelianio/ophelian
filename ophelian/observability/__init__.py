"""Observability surface — lightweight structured logging used across the lib.

Real OpenTelemetry / metrics integration ships in a later release; this module
just provides a configured logger so all subpackages emit consistent records.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("ophelian")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

__all__ = ["logger"]
