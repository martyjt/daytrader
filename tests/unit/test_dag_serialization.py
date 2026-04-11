"""Tests for DAG YAML serialization/deserialization."""

import tempfile
from pathlib import Path

from daytrader.algorithms.dag.serialization import (
    dag_from_yaml,
    dag_to_yaml,
    load_dag,
    save_dag,
)
from daytrader.algorithms.dag.types import DAGDefinition, DAGEdge, DAGNode


def _sample_dag() -> DAGDefinition:
    return DAGDefinition(
        id="trend_ensemble_v1",
        name="Trend Ensemble",
        version="0.1.0",
        description="EMA crossover + RSI mean reversion, majority vote",
        nodes=[
            DAGNode(
                "ema_1", "algorithm",
                algorithm_id="ema_crossover",
                params={"fast_period": 9, "slow_period": 21},
                position=(100, 200),
                weight=1.0,
            ),
            DAGNode(
                "rsi_1", "algorithm",
                algorithm_id="rsi_mean_reversion",
                params={"period": 14, "oversold": 30, "overbought": 70},
                position=(100, 400),
                weight=1.0,
            ),
            DAGNode(
                "vote_0", "combinator",
                combinator_type="majority_vote",
                params={"min_agreement": 0.5},
                position=(400, 300),
            ),
        ],
        edges=[
            DAGEdge("ema_1", "vote_0"),
            DAGEdge("rsi_1", "vote_0"),
        ],
        root_node_id="vote_0",
    )


def test_yaml_round_trip():
    """Serialize to YAML and back should produce equivalent DAG."""
    original = _sample_dag()
    yaml_str = dag_to_yaml(original)
    restored = dag_from_yaml(yaml_str)

    assert restored.id == original.id
    assert restored.name == original.name
    assert restored.version == original.version
    assert restored.root_node_id == original.root_node_id
    assert len(restored.nodes) == len(original.nodes)
    assert len(restored.edges) == len(original.edges)


def test_yaml_preserves_params():
    original = _sample_dag()
    yaml_str = dag_to_yaml(original)
    restored = dag_from_yaml(yaml_str)

    ema = restored.get_node("ema_1")
    assert ema.params["fast_period"] == 9
    assert ema.params["slow_period"] == 21

    rsi = restored.get_node("rsi_1")
    assert rsi.params["period"] == 14


def test_yaml_preserves_positions():
    original = _sample_dag()
    yaml_str = dag_to_yaml(original)
    restored = dag_from_yaml(yaml_str)

    ema = restored.get_node("ema_1")
    assert ema.position == (100, 200)


def test_yaml_preserves_edges():
    original = _sample_dag()
    yaml_str = dag_to_yaml(original)
    restored = dag_from_yaml(yaml_str)

    children = restored.children_of("vote_0")
    assert set(children) == {"ema_1", "rsi_1"}


def test_yaml_file_round_trip():
    """Save to file and load back should work."""
    original = _sample_dag()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_dag.yaml"
        save_dag(original, path)
        assert path.exists()

        loaded = load_dag(path)
        assert loaded.id == original.id
        assert len(loaded.nodes) == len(original.nodes)


def test_yaml_output_is_readable():
    """YAML output should contain expected keys."""
    dag = _sample_dag()
    yaml_str = dag_to_yaml(dag)
    assert "dag:" in yaml_str
    assert "trend_ensemble_v1" in yaml_str
    assert "ema_crossover" in yaml_str
    assert "majority_vote" in yaml_str
