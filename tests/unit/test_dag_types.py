"""Tests for DAG types and validation."""

from daytrader.algorithms.dag.types import DAGDefinition, DAGEdge, DAGNode
from daytrader.algorithms.dag.validation import topological_order, validate

import pytest


def _simple_dag() -> DAGDefinition:
    """Two algo leaves → one majority_vote combinator."""
    return DAGDefinition(
        id="test_dag",
        name="Test DAG",
        nodes=[
            DAGNode("ema_1", "algorithm", algorithm_id="ema_crossover", position=(100, 200)),
            DAGNode("rsi_1", "algorithm", algorithm_id="rsi_mean_reversion", position=(100, 400)),
            DAGNode("vote_0", "combinator", combinator_type="majority_vote", position=(400, 300)),
        ],
        edges=[
            DAGEdge("ema_1", "vote_0"),
            DAGEdge("rsi_1", "vote_0"),
        ],
        root_node_id="vote_0",
    )


def test_get_node():
    dag = _simple_dag()
    node = dag.get_node("ema_1")
    assert node.algorithm_id == "ema_crossover"


def test_get_node_missing():
    dag = _simple_dag()
    with pytest.raises(KeyError, match="not_here"):
        dag.get_node("not_here")


def test_children_of():
    dag = _simple_dag()
    children = dag.children_of("vote_0")
    assert set(children) == {"ema_1", "rsi_1"}


def test_leaf_nodes():
    dag = _simple_dag()
    leaves = dag.leaf_nodes()
    leaf_ids = {n.node_id for n in leaves}
    assert leaf_ids == {"ema_1", "rsi_1"}


def test_validate_valid_dag():
    dag = _simple_dag()
    errors = validate(dag)
    assert errors == []


def test_validate_no_nodes():
    dag = DAGDefinition(id="empty", name="Empty")
    errors = validate(dag)
    assert "no nodes" in errors[0].lower()


def test_validate_no_root():
    dag = DAGDefinition(
        id="no_root",
        name="No Root",
        nodes=[DAGNode("a", "algorithm", algorithm_id="ema_crossover")],
        root_node_id=None,
    )
    errors = validate(dag)
    assert any("root" in e.lower() for e in errors)


def test_validate_algorithm_root_rejected():
    dag = DAGDefinition(
        id="algo_root",
        name="Algo Root",
        nodes=[DAGNode("a", "algorithm", algorithm_id="ema_crossover")],
        root_node_id="a",
    )
    errors = validate(dag)
    assert any("algorithm node" in e.lower() for e in errors)


def test_validate_cycle_detected():
    dag = DAGDefinition(
        id="cycle",
        name="Cycle",
        nodes=[
            DAGNode("a", "combinator", combinator_type="weighted_average"),
            DAGNode("b", "combinator", combinator_type="weighted_average"),
        ],
        edges=[
            DAGEdge("a", "b"),
            DAGEdge("b", "a"),
        ],
        root_node_id="a",
    )
    errors = validate(dag)
    assert any("cycle" in e.lower() for e in errors)


def test_validate_missing_algorithm_id():
    dag = DAGDefinition(
        id="missing_algo",
        name="Missing Algo",
        nodes=[
            DAGNode("a", "algorithm"),  # no algorithm_id!
            DAGNode("c", "combinator", combinator_type="majority_vote"),
        ],
        edges=[DAGEdge("a", "c")],
        root_node_id="c",
    )
    errors = validate(dag)
    assert any("algorithm_id" in e.lower() for e in errors)


def test_topological_order():
    dag = _simple_dag()
    order = topological_order(dag)
    # ema_1 and rsi_1 should come before vote_0
    assert order.index("vote_0") > order.index("ema_1")
    assert order.index("vote_0") > order.index("rsi_1")


def test_topological_order_cycle_raises():
    dag = DAGDefinition(
        id="cycle",
        name="Cycle",
        nodes=[
            DAGNode("a", "combinator", combinator_type="weighted_average"),
            DAGNode("b", "combinator", combinator_type="weighted_average"),
        ],
        edges=[DAGEdge("a", "b"), DAGEdge("b", "a")],
        root_node_id="a",
    )
    with pytest.raises(ValueError, match="cycle"):
        topological_order(dag)
