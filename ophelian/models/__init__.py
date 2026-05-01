"""Pluggable model adapters — one per supported framework."""

from __future__ import annotations

from ophelian.models.base import ModelAdapter, register_adapter, registry

__all__ = ["ModelAdapter", "register_adapter", "registry"]
