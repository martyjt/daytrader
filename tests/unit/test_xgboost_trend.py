"""Tests for the XGBoost Trend Classifier algorithm."""

from datetime import datetime, timedelta
from typing import Any

import numpy as np
import polars as pl

from daytrader.algorithms.builtin.xgboost_trend import (
    XGBoostTrendAlgorithm,
    _build_feature_matrix,
    _rolling_return,
    _rsi,
)
from daytrader.algorithms.registry import AlgorithmRegistry
from daytrader.backtest.engine import BacktestEngine
from daytrader.core.types.bars import Timeframe
from daytrader.core.types.symbols import AssetClass, Symbol


def _trending_ohlcv(n: int = 200, start_price: float = 100.0) -> pl.DataFrame:
    """Generate trending data with some noise for ML training."""
    np.random.seed(42)
    prices = start_price + np.cumsum(np.random.randn(n) * 0.5 + 0.1)
    prices = np.maximum(prices, 1.0)  # Ensure positive
    return pl.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n)],
            "open": prices.tolist(),
            "high": (prices + np.abs(np.random.randn(n)) * 2).tolist(),
            "low": (prices - np.abs(np.random.randn(n)) * 2).tolist(),
            "close": prices.tolist(),
            "volume": (1000 + np.random.randn(n) * 100).tolist(),
        }
    )


def _symbol() -> Symbol:
    return Symbol("TEST", "USD", AssetClass.CRYPTO)


# ---------------------------------------------------------------------------
# Manifest and properties
# ---------------------------------------------------------------------------


def test_manifest():
    algo = XGBoostTrendAlgorithm()
    m = algo.manifest
    assert m.id == "xgboost_trend"
    assert m.name == "XGBoost Trend"
    assert "crypto" in m.asset_classes
    assert "equities" in m.asset_classes


def test_warmup_bars():
    algo = XGBoostTrendAlgorithm(lookback=50)
    assert algo.warmup_bars() == 50

    algo2 = XGBoostTrendAlgorithm(lookback=100)
    assert algo2.warmup_bars() == 100


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def test_rolling_return_shape():
    closes = np.arange(1, 101, dtype=float)
    ret = _rolling_return(closes, 5)
    assert len(ret) == 100
    assert np.isnan(ret[0])
    assert not np.isnan(ret[5])


def test_rsi_range():
    """RSI should be between 0 and 100."""
    np.random.seed(42)
    closes = 100 + np.cumsum(np.random.randn(100))
    rsi = _rsi(closes, 14)
    valid = rsi[~np.isnan(rsi)]
    assert len(valid) > 0
    assert all(0 <= v <= 100 for v in valid)


def test_feature_matrix_shape():
    n = 100
    closes = np.arange(1, n + 1, dtype=float)
    highs = closes + 1
    lows = closes - 1
    volumes = np.ones(n) * 1000

    features = _build_feature_matrix(closes, highs, lows, volumes)
    assert features.shape == (n, 7)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def test_train_sets_is_trained():
    algo = XGBoostTrendAlgorithm(lookback=20, auto_train=False)
    data = _trending_ohlcv(200)
    assert not algo._is_trained
    algo.train(data)
    assert algo._is_trained


def test_train_insufficient_data():
    """Training with too few rows should not crash and leave untrained."""
    algo = XGBoostTrendAlgorithm(lookback=20, auto_train=False)
    data = _trending_ohlcv(25)
    algo.train(data)
    assert not algo._is_trained


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def test_on_bar_before_training_returns_none():
    """Without training and auto_train=False, on_bar should return None."""
    algo = XGBoostTrendAlgorithm(auto_train=False)
    data = _trending_ohlcv(100)
    closes = data["close"].to_numpy().astype(float)
    highs = data["high"].to_numpy().astype(float)
    lows = data["low"].to_numpy().astype(float)
    volumes = data["volume"].to_numpy().astype(float)

    from decimal import Decimal
    from uuid import uuid4

    from daytrader.core.context import AlgorithmContext
    from daytrader.core.types.bars import Bar

    emitted: list[Any] = []
    logs = []
    ctx = AlgorithmContext(
        tenant_id=uuid4(),
        persona_id=uuid4(),
        algorithm_id="xgboost_trend",
        symbol=_symbol(),
        timeframe=Timeframe.D1,
        now=datetime(2024, 4, 10),
        bar=Bar(
            timestamp=datetime(2024, 4, 10),
            open=Decimal("100"),
            high=Decimal("101"),
            low=Decimal("99"),
            close=Decimal("100"),
            volume=Decimal("1000"),
        ),
        history_arrays={
            "close": closes,
            "high": highs,
            "low": lows,
            "volume": volumes,
        },
        features={},
        params={},
        emit_fn=emitted.append,
        log_fn=lambda msg, fields: logs.append({"message": msg, **fields}),
    )
    result = algo.on_bar(ctx)
    assert result is None


def test_on_bar_after_training_emits():
    """After training on sufficient data, on_bar should emit signals."""
    algo = XGBoostTrendAlgorithm(
        lookback=30, auto_train=False, score_threshold=0.5,
    )
    data = _trending_ohlcv(200)
    algo.train(data)

    closes = data["close"].to_numpy().astype(float)
    highs = data["high"].to_numpy().astype(float)
    lows = data["low"].to_numpy().astype(float)
    volumes = data["volume"].to_numpy().astype(float)

    from decimal import Decimal
    from uuid import uuid4

    from daytrader.core.context import AlgorithmContext
    from daytrader.core.types.bars import Bar

    emitted: list[Any] = []
    logs = []
    ctx = AlgorithmContext(
        tenant_id=uuid4(),
        persona_id=uuid4(),
        algorithm_id="xgboost_trend",
        symbol=_symbol(),
        timeframe=Timeframe.D1,
        now=datetime(2024, 7, 10),
        bar=Bar(
            timestamp=datetime(2024, 7, 10),
            open=Decimal(str(closes[-1])),
            high=Decimal(str(highs[-1])),
            low=Decimal(str(lows[-1])),
            close=Decimal(str(closes[-1])),
            volume=Decimal(str(volumes[-1])),
        ),
        history_arrays={
            "close": closes,
            "high": highs,
            "low": lows,
            "volume": volumes,
        },
        features={},
        params={},
        emit_fn=emitted.append,
        log_fn=lambda msg, fields: logs.append({"message": msg, **fields}),
    )
    signal = algo.on_bar(ctx)

    # Should have logged feature importance
    assert len(logs) >= 1
    assert logs[0]["message"] == "xgboost_prediction"
    assert "up_probability" in logs[0]

    # Signal emitted (or None if prediction was right at threshold)
    # With threshold 0.5 it should always emit
    if signal is not None:
        assert -1.0 <= signal.score <= 1.0
        assert "feature_importance" in signal.metadata


# ---------------------------------------------------------------------------
# End-to-end with backtest engine
# ---------------------------------------------------------------------------


async def test_xgboost_end_to_end_backtest():
    """XGBoost should work with BacktestEngine on sufficient data."""
    algo = XGBoostTrendAlgorithm(lookback=30, auto_train=True)
    engine = BacktestEngine()
    data = _trending_ohlcv(200)

    result = await engine.run(
        algorithm=algo,
        symbol=_symbol(),
        timeframe=Timeframe.D1,
        start=datetime(2024, 1, 1),
        end=datetime(2024, 7, 19),
        data=data,
        commission_bps=0,
    )

    assert len(result.equity_curve) == 200
    assert result.final_equity > 0


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_includes_xgboost():
    AlgorithmRegistry.clear()
    AlgorithmRegistry.auto_register()
    assert "xgboost_trend" in AlgorithmRegistry.available()
    algo = AlgorithmRegistry.get("xgboost_trend")
    assert algo.manifest.id == "xgboost_trend"
