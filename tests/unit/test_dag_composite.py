"""Tests for CompositeAlgorithm — the DAG execution engine."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from daytrader.algorithms.dag.composite import CompositeAlgorithm
from daytrader.algorithms.dag.types import DAGDefinition, DAGEdge, DAGNode
from daytrader.algorithms.registry import AlgorithmRegistry
from daytrader.backtest.engine import BacktestEngine
from daytrader.core.types.bars import Timeframe
from daytrader.core.types.symbols import AssetClass, Symbol


def _ensure_registry():
    if not AlgorithmRegistry.available():
        AlgorithmRegistry.auto_register()


def _simple_dag() -> DAGDefinition:
    """Two algo leaves → weighted_average combinator."""
    return DAGDefinition(
        id="test_composite",
        name="Test Composite",
        nodes=[
            DAGNode("ema_1", "algorithm", algorithm_id="ema_crossover"),
            DAGNode("rsi_1", "algorithm", algorithm_id="rsi_mean_reversion"),
            DAGNode("avg_0", "combinator", combinator_type="weighted_average"),
        ],
        edges=[
            DAGEdge("ema_1", "avg_0"),
            DAGEdge("rsi_1", "avg_0"),
        ],
        root_node_id="avg_0",
    )


def _vote_dag() -> DAGDefinition:
    """Three algo leaves → majority_vote combinator."""
    return DAGDefinition(
        id="vote_dag",
        name="Vote DAG",
        nodes=[
            DAGNode("ema_1", "algorithm", algorithm_id="ema_crossover"),
            DAGNode("rsi_1", "algorithm", algorithm_id="rsi_mean_reversion"),
            DAGNode("bb_1", "algorithm", algorithm_id="bollinger_bands"),
            DAGNode("vote_0", "combinator", combinator_type="majority_vote"),
        ],
        edges=[
            DAGEdge("ema_1", "vote_0"),
            DAGEdge("rsi_1", "vote_0"),
            DAGEdge("bb_1", "vote_0"),
        ],
        root_node_id="vote_0",
    )


def _symbol() -> Symbol:
    return Symbol("TEST", "USD", AssetClass.CRYPTO)


def _trending_ohlcv(n: int = 300) -> pl.DataFrame:
    np.random.seed(42)
    prices = 100.0 + np.cumsum(np.random.randn(n) * 0.5 + 0.1)
    prices = np.maximum(prices, 1.0)
    return pl.DataFrame({
        "timestamp": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n)],
        "open": prices.tolist(),
        "high": (prices + np.abs(np.random.randn(n)) * 2).tolist(),
        "low": (prices - np.abs(np.random.randn(n)) * 2).tolist(),
        "close": prices.tolist(),
        "volume": (1000 + np.abs(np.random.randn(n)) * 500).tolist(),
    })


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_composite_construction():
    _ensure_registry()
    algo = CompositeAlgorithm(_simple_dag())
    assert algo.manifest.id == "dag:test_composite"
    assert algo.manifest.name == "Test Composite"


def test_composite_invalid_dag_raises():
    bad_dag = DAGDefinition(id="bad", name="Bad")
    with pytest.raises(ValueError, match="Invalid DAG"):
        CompositeAlgorithm(bad_dag)


def test_composite_warmup():
    _ensure_registry()
    algo = CompositeAlgorithm(_simple_dag())
    # Should be max of child warmup bars
    assert algo.warmup_bars() > 0


# ---------------------------------------------------------------------------
# End-to-end with backtest engine
# ---------------------------------------------------------------------------


async def test_composite_backtest():
    """CompositeAlgorithm should run through BacktestEngine cleanly."""
    _ensure_registry()
    algo = CompositeAlgorithm(_simple_dag())
    data = _trending_ohlcv(300)

    result = await BacktestEngine().run(
        algorithm=algo,
        symbol=_symbol(),
        timeframe=Timeframe.D1,
        start=datetime(2024, 1, 1),
        end=datetime(2024, 10, 27),
        data=data,
        commission_bps=0,
    )
    assert len(result.equity_curve) == 300
    assert result.final_equity > 0


async def test_vote_dag_backtest():
    """Three-algo majority vote DAG should run cleanly."""
    _ensure_registry()
    algo = CompositeAlgorithm(_vote_dag())
    data = _trending_ohlcv(300)

    result = await BacktestEngine().run(
        algorithm=algo,
        symbol=_symbol(),
        timeframe=Timeframe.D1,
        start=datetime(2024, 1, 1),
        end=datetime(2024, 10, 27),
        data=data,
        commission_bps=0,
    )
    assert len(result.equity_curve) == 300
    assert result.final_equity > 0


async def test_composite_signal_attribution():
    """Signals from CompositeAlgorithm should have dag_id in metadata."""
    _ensure_registry()
    algo = CompositeAlgorithm(_simple_dag())
    data = _trending_ohlcv(300)

    result = await BacktestEngine().run(
        algorithm=algo,
        symbol=_symbol(),
        timeframe=Timeframe.D1,
        start=datetime(2024, 1, 1),
        end=datetime(2024, 10, 27),
        data=data,
        commission_bps=0,
    )
    # If any signals were emitted, check metadata
    for sig in result.signals:
        assert "dag_id" in sig.metadata
        assert sig.metadata["dag_id"] == "test_composite"


async def test_composite_with_custom_params():
    """Node-level param overrides should be respected."""
    _ensure_registry()
    dag = DAGDefinition(
        id="param_test",
        name="Param Test",
        nodes=[
            DAGNode(
                "ema_1", "algorithm",
                algorithm_id="ema_crossover",
                params={"fast_period": 5, "slow_period": 15},
            ),
            DAGNode("rsi_1", "algorithm", algorithm_id="rsi_mean_reversion"),
            DAGNode("avg_0", "combinator", combinator_type="weighted_average"),
        ],
        edges=[
            DAGEdge("ema_1", "avg_0"),
            DAGEdge("rsi_1", "avg_0"),
        ],
        root_node_id="avg_0",
    )
    algo = CompositeAlgorithm(dag)
    data = _trending_ohlcv(300)

    result = await BacktestEngine().run(
        algorithm=algo,
        symbol=_symbol(),
        timeframe=Timeframe.D1,
        start=datetime(2024, 1, 1),
        end=datetime(2024, 10, 27),
        data=data,
        commission_bps=0,
    )
    assert len(result.equity_curve) == 300
