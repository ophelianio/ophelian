"""Compile a declarative `Pipeline` into an ordered execution plan."""

from __future__ import annotations

from collections import defaultdict, deque

from pydantic import BaseModel, ConfigDict, Field
from rich.console import Console
from rich.table import Table
from rich.tree import Tree

from ophelian.core.nodes import Node, Pipeline


class PlannedStep(BaseModel):
    """A node ready to run, annotated with its resolved upstream dependencies."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    order: int
    node: Node
    depends_on: tuple[str, ...]

    @property
    def name(self) -> str:
        return self.node.name

    @property
    def kind(self) -> str:
        return self.node.kind


class ExecutionPlan(BaseModel):
    """Ordered list of `PlannedStep`s the runtime will execute."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    pipeline: str
    steps: list[PlannedStep] = Field(default_factory=list)

    def render(self, console: Console | None = None) -> None:
        """Render the plan as a rich table + dependency tree."""
        console = console or Console()
        table = Table(title=f"Ophelian execution plan — {self.pipeline}")
        table.add_column("#", justify="right", style="cyan", no_wrap=True)
        table.add_column("Node", style="bold")
        table.add_column("Kind", style="magenta")
        table.add_column("Depends on", style="green")
        for step in self.steps:
            deps = ", ".join(step.depends_on) if step.depends_on else "—"
            table.add_row(str(step.order), step.name, step.kind, deps)
        console.print(table)

        tree = Tree(f"[bold]{self.pipeline}[/bold]")
        children: dict[str, list[str]] = defaultdict(list)
        roots: list[str] = []
        for step in self.steps:
            if not step.depends_on:
                roots.append(step.name)
            for dep in step.depends_on:
                children[dep].append(step.name)
        seen: set[str] = set()

        def add(node_name: str, branch: Tree) -> None:
            if node_name in seen:
                branch.add(f"[dim]{node_name} (already shown)[/dim]")
                return
            seen.add(node_name)
            sub = branch.add(node_name)
            for child in children.get(node_name, []):
                add(child, sub)

        for root in roots:
            add(root, tree)
        console.print(tree)


class GraphCompiler:
    """Topologically order pipeline nodes into a deterministic plan."""

    def compile(self, pipeline: Pipeline) -> ExecutionPlan:
        nodes_by_name: dict[str, Node] = {n.name: n for n in pipeline.steps}
        indegree: dict[str, int] = dict.fromkeys(nodes_by_name, 0)
        adjacency: dict[str, list[str]] = defaultdict(list)
        for node in pipeline.steps:
            for dep in node.depends_on:
                adjacency[dep].append(node.name)
                indegree[node.name] += 1

        ordering: list[str] = []
        # Preserve user-declared order among nodes with the same indegree.
        ready: deque[str] = deque(name for name in nodes_by_name if indegree[name] == 0)
        while ready:
            current = ready.popleft()
            ordering.append(current)
            for child in adjacency[current]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)

        if len(ordering) != len(nodes_by_name):
            cycle = [name for name, deg in indegree.items() if deg > 0]
            raise ValueError(f"Cycle detected in pipeline {pipeline.name!r} involving: {cycle}")

        steps = [
            PlannedStep(
                order=i,
                node=nodes_by_name[name],
                depends_on=nodes_by_name[name].depends_on,
            )
            for i, name in enumerate(ordering)
        ]
        return ExecutionPlan(pipeline=pipeline.name, steps=steps)
