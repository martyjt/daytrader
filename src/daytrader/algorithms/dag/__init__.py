"""DAG composition engine — tier-2 strategy composer.

Allows composing multiple algorithms into a single strategy via
a directed acyclic graph of algorithm and combinator nodes.
"""

from .composite import CompositeAlgorithm
from .types import DAGDefinition, DAGEdge, DAGNode

__all__ = [
    "CompositeAlgorithm",
    "DAGDefinition",
    "DAGEdge",
    "DAGNode",
]
