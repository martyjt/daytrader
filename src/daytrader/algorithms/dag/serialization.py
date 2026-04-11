"""YAML serialization/deserialization for DAG definitions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .types import DAGDefinition, DAGEdge, DAGNode


def dag_to_dict(dag: DAGDefinition) -> dict[str, Any]:
    """Convert a DAGDefinition to a plain dict for YAML serialization."""
    return {
        "dag": {
            "id": dag.id,
            "name": dag.name,
            "version": dag.version,
            "description": dag.description,
            "nodes": [
                {
                    "node_id": n.node_id,
                    "node_type": n.node_type,
                    **({"algorithm_id": n.algorithm_id} if n.algorithm_id else {}),
                    **({"combinator_type": n.combinator_type} if n.combinator_type else {}),
                    **({"params": dict(n.params)} if n.params else {}),
                    "position": list(n.position),
                    "weight": n.weight,
                }
                for n in dag.nodes
            ],
            "edges": [
                {"source_id": e.source_id, "target_id": e.target_id}
                for e in dag.edges
            ],
            "root_node_id": dag.root_node_id,
            **({"metadata": dag.metadata} if dag.metadata else {}),
        }
    }


def dag_from_dict(data: dict[str, Any]) -> DAGDefinition:
    """Parse a DAGDefinition from a plain dict (YAML-loaded)."""
    d = data["dag"]
    nodes = [
        DAGNode(
            node_id=n["node_id"],
            node_type=n["node_type"],
            algorithm_id=n.get("algorithm_id"),
            combinator_type=n.get("combinator_type"),
            params=n.get("params", {}),
            position=tuple(n.get("position", [0, 0])),
            weight=n.get("weight", 1.0),
        )
        for n in d.get("nodes", [])
    ]
    edges = [
        DAGEdge(
            source_id=e["source_id"],
            target_id=e["target_id"],
            source_slot=e.get("source_slot", 0),
            target_slot=e.get("target_slot", 0),
        )
        for e in d.get("edges", [])
    ]
    return DAGDefinition(
        id=d["id"],
        name=d["name"],
        version=d.get("version", "0.1.0"),
        description=d.get("description", ""),
        nodes=nodes,
        edges=edges,
        root_node_id=d.get("root_node_id"),
        metadata=d.get("metadata", {}),
    )


def dag_to_yaml(dag: DAGDefinition) -> str:
    """Serialize a DAGDefinition to a YAML string."""
    return yaml.dump(dag_to_dict(dag), default_flow_style=False, sort_keys=False)


def dag_from_yaml(yaml_str: str) -> DAGDefinition:
    """Deserialize a DAGDefinition from a YAML string."""
    data = yaml.safe_load(yaml_str)
    return dag_from_dict(data)


def save_dag(dag: DAGDefinition, path: Path | str) -> None:
    """Save a DAGDefinition to a YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dag_to_yaml(dag), encoding="utf-8")


def load_dag(path: Path | str) -> DAGDefinition:
    """Load a DAGDefinition from a YAML file."""
    path = Path(path)
    return dag_from_yaml(path.read_text(encoding="utf-8"))
