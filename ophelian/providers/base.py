"""Abstract `Provider` interface every execution backend implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ophelian.core.compiler import ExecutionPlan
    from ophelian.core.nodes import Pipeline, PipelineResult


class Provider(ABC):
    """Backend that knows how to materialise an `ExecutionPlan`."""

    name: str = "abstract"

    @abstractmethod
    def execute(self, pipeline: Pipeline, plan: ExecutionPlan) -> PipelineResult:
        """Run every step in `plan` and return the aggregated result."""

    def describe(self) -> str:
        """Short human-readable description used by the CLI."""
        return self.name
