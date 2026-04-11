"""DAG structural validation: cycle detection, root checks, integrity."""

from __future__ import annotations

from collections import deque

from .types import DAGDefinition


def validate(dag: DAGDefinition) -> list[str]:
    """Return a list of validation errors (empty means valid)."""
    errors: list[str] = []

    if not dag.nodes:
        errors.append("DAG has no nodes")
        return errors

    node_ids = {n.node_id for n in dag.nodes}

    # Check for duplicate node IDs
    if len(node_ids) != len(dag.nodes):
        errors.append("Duplicate node IDs detected")

    # Check edge endpoints exist
    for edge in dag.edges:
        if edge.source_id not in node_ids:
            errors.append(f"Edge source {edge.source_id!r} not in nodes")
        if edge.target_id not in node_ids:
            errors.append(f"Edge target {edge.target_id!r} not in nodes")

    # Check root node
    if dag.root_node_id is None:
        errors.append("No root_node_id specified")
    elif dag.root_node_id not in node_ids:
        errors.append(f"Root node {dag.root_node_id!r} not in nodes")
    else:
        root = dag.get_node(dag.root_node_id)
        if root.node_type == "algorithm":
            errors.append(
                f"Root node {dag.root_node_id!r} is an algorithm node; "
                "root must be a combinator or risk_filter"
            )

    # Check algorithm nodes have algorithm_id
    for node in dag.nodes:
        if node.node_type == "algorithm" and not node.algorithm_id:
            errors.append(f"Algorithm node {node.node_id!r} has no algorithm_id")
        if node.node_type == "combinator" and not node.combinator_type:
            errors.append(f"Combinator node {node.node_id!r} has no combinator_type")

    # Check for cycles
    if _has_cycle(dag):
        errors.append("DAG contains a cycle")

    # Check combinator nodes have at least one input
    if not errors:  # only if no structural errors
        for node in dag.nodes:
            if node.node_type in ("combinator", "risk_filter"):
                children = dag.children_of(node.node_id)
                if not children:
                    errors.append(
                        f"Combinator/filter {node.node_id!r} has no inputs"
                    )

    return errors


def topological_order(dag: DAGDefinition) -> list[str]:
    """Return node IDs in topological order (leaves first, root last).

    Uses Kahn's algorithm (BFS-based). Raises ValueError if the DAG
    has a cycle.
    """
    # Build adjacency and in-degree maps
    in_degree: dict[str, int] = {n.node_id: 0 for n in dag.nodes}
    adjacency: dict[str, list[str]] = {n.node_id: [] for n in dag.nodes}

    for edge in dag.edges:
        adjacency[edge.source_id].append(edge.target_id)
        in_degree[edge.target_id] += 1

    # Start with nodes that have no incoming edges (leaves)
    queue: deque[str] = deque()
    for nid, deg in in_degree.items():
        if deg == 0:
            queue.append(nid)

    order: list[str] = []
    while queue:
        nid = queue.popleft()
        order.append(nid)
        for child in adjacency[nid]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    if len(order) != len(dag.nodes):
        raise ValueError("DAG contains a cycle — topological sort failed")

    return order


def _has_cycle(dag: DAGDefinition) -> bool:
    """Check for cycles using Kahn's algorithm."""
    try:
        topological_order(dag)
        return False
    except ValueError:
        return True
