"""Tests for the Transformer Trend Classifier algorithm."""

from datetime import datetime, timedelta
from decimal import Decimal
from uuid import uuid4

import numpy as np
import polars as pl
import pytest

from daytrader.algorithms.builtin.transformer_trend import TransformerTrendAlgorithm
from daytrader.algorithms.registry import AlgorithmRegistry
from daytrader.backtest.engine import BacktestEngine
from daytrader.core.context import AlgorithmContext
from daytrader.core.types.bars import Bar, Timeframe
from daytrader.core.types.symbols import AssetClass, Symbol


def _trending_ohlcv(n: int = 200, start_price: float = 100.0) -> pl.DataFrame:
    np.random.seed(42)
    prices = start_price + np.cumsum(np.random.randn(n) * 0.5 + 0.1)
    prices = np.maximum(prices, 1.0)
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
    algo = TransformerTrendAlgorithm()
    m = algo.manifest
    assert m.id == "transformer_trend"
    assert m.name == "Transformer Trend"
    assert "crypto" in m.asset_classes
    assert "equities" in m.asset_classes


def test_warmup_bars():
    algo = TransformerTrendAlgorithm(lookback=50)
    assert algo.warmup_bars() == 50


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def test_train_sets_is_trained():
    algo = TransformerTrendAlgorithm(
        lookback=20, sequence_length=10, epochs=5, auto_train=False,
    )
    data = _trending_ohlcv(200)
    assert not algo._is_trained
    algo.train(data)
    assert algo._is_trained


def test_train_insufficient_data():
    algo = TransformerTrendAlgorithm(
        lookback=20, sequence_length=10, auto_train=False,
    )
    data = _trending_ohlcv(25)
    algo.train(data)
    assert not algo._is_trained


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def test_on_bar_before_training_returns_none():
    algo = TransformerTrendAlgorithm(auto_train=False)
    data = _trending_ohlcv(100)
    closes = data["close"].to_numpy().astype(float)
    highs = data["high"].to_numpy().astype(float)
    lows = data["low"].to_numpy().astype(float)
    volumes = data["volume"].to_numpy().astype(float)

    ctx = AlgorithmContext(
        tenant_id=uuid4(),
        persona_id=uuid4(),
        algorithm_id="transformer_trend",
        symbol=_symbol(),
        timeframe=Timeframe.D1,
        now=datetime(2024, 4, 10),
        bar=Bar(
            timestamp=datetime(2024, 4, 10),
            open=Decimal("100"), high=Decimal("101"),
            low=Decimal("99"), close=Decimal("100"),
            volume=Decimal("1000"),
        ),
        history_arrays={
            "close": closes, "open": closes,
            "high": highs, "low": lows, "volume": volumes,
        },
        features={},
        params={},
        emit_fn=lambda s: None,
        log_fn=lambda msg, fields: None,
    )
    result = algo.on_bar(ctx)
    assert result is None


def test_on_bar_after_training_emits():
    algo = TransformerTrendAlgorithm(
        lookback=30, sequence_length=10, epochs=10,
        auto_train=False, score_threshold=0.5,
    )
    data = _trending_ohlcv(200)
    algo.train(data)

    closes = data["close"].to_numpy().astype(float)
    opens = data["open"].to_numpy().astype(float)
    highs = data["high"].to_numpy().astype(float)
    lows = data["low"].to_numpy().astype(float)
    volumes = data["volume"].to_numpy().astype(float)

    emitted = []
    logs = []
    ctx = AlgorithmContext(
        tenant_id=uuid4(),
        persona_id=uuid4(),
        algorithm_id="transformer_trend",
        symbol=_symbol(),
        timeframe=Timeframe.D1,
        now=datetime(2024, 7, 10),
        bar=Bar(
            timestamp=datetime(2024, 7, 10),
            open=Decimal(str(opens[-1])),
            high=Decimal(str(highs[-1])),
            low=Decimal(str(lows[-1])),
            close=Decimal(str(closes[-1])),
            volume=Decimal(str(volumes[-1])),
        ),
        history_arrays={
            "close": closes, "open": opens,
            "high": highs, "low": lows, "volume": volumes,
        },
        features={},
        params={},
        emit_fn=emitted.append,
        log_fn=lambda msg, fields: logs.append({"message": msg, **fields}),
    )
    signal = algo.on_bar(ctx)

    assert len(logs) >= 1
    assert "up_probability" in logs[0]

    if signal is not None:
        assert -1.0 <= signal.score <= 1.0


# ---------------------------------------------------------------------------
# End-to-end with backtest engine
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_transformer_end_to_end_backtest():
    algo = TransformerTrendAlgorithm(
        lookback=30, sequence_length=10, epochs=5, auto_train=True,
    )
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


def test_registry_includes_transformer():
    AlgorithmRegistry.clear()
    AlgorithmRegistry.auto_register()
    assert "transformer_trend" in AlgorithmRegistry.available()
    algo = AlgorithmRegistry.get("transformer_trend")
    assert algo.manifest.id == "transformer_trend"
