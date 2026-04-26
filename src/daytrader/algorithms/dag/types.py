"""DAG data model: nodes, edges, and the DAGDefinition container."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DAGNode:
    """One node in a composed strategy DAG."""

    node_id: str
    node_type: str  # "algorithm" | "combinator" | "risk_filter"
    algorithm_id: str | None = None  # For algorithm nodes: registry key
    combinator_type: str | None = None  # For combinator nodes
    params: dict[str, Any] = field(default_factory=dict)
    position: tuple[float, float] = (0.0, 0.0)  # Canvas x, y
    weight: float = 1.0


@dataclass(frozen=True)
class DAGEdge:
    """Directed edge: source node output feeds into target node input."""

    source_id: str
    target_id: str
    source_slot: int = 0
    target_slot: int = 0


@dataclass
class DAGDefinition:
    """Complete DAG specification — serializable to YAML."""

    id: str
    name: str
    version: str = "0.1.0"
    description: str = ""
    nodes: list[DAGNode] = field(default_factory=list)
    edges: list[DAGEdge] = field(default_factory=list)
    root_node_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_node(self, node_id: str) -> DAGNode:
        """Retrieve a node by ID. Raises KeyError if not found."""
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        raise KeyError(f"Node {node_id!r} not found in DAG {self.id!r}")

    def children_of(self, node_id: str) -> list[str]:
        """Return source node IDs that feed into the given node."""
        return [e.source_id for e in self.edges if e.target_id == node_id]

    def parents_of(self, node_id: str) -> list[str]:
        """Return target node IDs that the given node feeds into."""
        return [e.target_id for e in self.edges if e.source_id == node_id]

    def leaf_nodes(self) -> list[DAGNode]:
        """Return nodes with no incoming edges (algorithm leaf nodes)."""
        targets = {e.target_id for e in self.edges}
        {e.source_id for e in self.edges}
        # Leaves are nodes that are sources but never targets,
        # or nodes with no edges at all (isolated)
        leaf_ids = set()
        for node in self.nodes:
            if node.node_id not in targets:
                leaf_ids.add(node.node_id)
        return [n for n in self.nodes if n.node_id in leaf_ids]
