"""Tests for explainability: debug log capture in backtest runs."""

from datetime import datetime, timedelta

import polars as pl

from daytrader.algorithms.base import Algorithm, AlgorithmManifest
from daytrader.algorithms.builtin.buy_hold import BuyHoldAlgorithm
from daytrader.backtest.engine import BacktestEngine
from daytrader.core.context import AlgorithmContext
from daytrader.core.types.bars import Timeframe
from daytrader.core.types.signals import Signal
from daytrader.core.types.symbols import AssetClass, Symbol


def _linear_ohlcv(n: int = 20, start_price: float = 100.0) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n)],
            "open": [start_price + i for i in range(n)],
            "high": [start_price + i + 1 for i in range(n)],
            "low": [start_price + i - 1 for i in range(n)],
            "close": [start_price + i for i in range(n)],
            "volume": [1000.0] * n,
        }
    )


def _symbol() -> Symbol:
    return Symbol("TEST", "USD", AssetClass.CRYPTO)


class LoggingAlgorithm(Algorithm):
    """Algorithm that emits debug logs on every bar."""

    @property
    def manifest(self) -> AlgorithmManifest:
        return AlgorithmManifest(
            id="logging_test",
            name="Logging Test",
            version="0.1.0",
        )

    def on_bar(self, ctx: AlgorithmContext) -> Signal | None:
        ctx.log("processing bar", close=float(ctx.bar.close), bar_index=ctx.now)
        return ctx.emit(score=1.0, confidence=1.0, reason="test")


async def test_debug_logs_captured():
    """Algorithm log calls should appear in result.debug_logs."""
    engine = BacktestEngine()
    result = await engine.run(
        algorithm=LoggingAlgorithm(),
        symbol=_symbol(),
        timeframe=Timeframe.D1,
        start=datetime(2024, 1, 1),
        end=datetime(2024, 1, 10),
        data=_linear_ohlcv(10),
    )
    assert len(result.debug_logs) == 10
    assert result.debug_logs[0]["message"] == "processing bar"
    assert "close" in result.debug_logs[0]
    assert "bar" in result.debug_logs[0]
    assert "timestamp" in result.debug_logs[0]


async def test_debug_logs_empty_when_no_logging():
    """Buy & Hold never calls ctx.log(), so debug_logs should be empty."""
    engine = BacktestEngine()
    result = await engine.run(
        algorithm=BuyHoldAlgorithm(),
        symbol=_symbol(),
        timeframe=Timeframe.D1,
        start=datetime(2024, 1, 1),
        end=datetime(2024, 1, 10),
        data=_linear_ohlcv(10),
    )
    assert result.debug_logs == []


async def test_debug_logs_contain_bar_index_and_timestamp():
    """Each log entry should have bar index and timestamp."""
    engine = BacktestEngine()
    result = await engine.run(
        algorithm=LoggingAlgorithm(),
        symbol=_symbol(),
        timeframe=Timeframe.D1,
        start=datetime(2024, 1, 1),
        end=datetime(2024, 1, 5),
        data=_linear_ohlcv(5),
    )
    for i, log in enumerate(result.debug_logs):
        assert log["bar"] == i
        assert isinstance(log["timestamp"], str)
