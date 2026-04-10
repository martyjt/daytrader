"""Tests for the backtest engine using synthetic data."""

from datetime import datetime, timedelta

import polars as pl

from daytrader.algorithms.builtin.buy_hold import BuyHoldAlgorithm
from daytrader.backtest.engine import BacktestEngine
from daytrader.core.types.bars import Timeframe
from daytrader.core.types.symbols import AssetClass, Symbol


def _linear_ohlcv(n: int = 20, start_price: float = 100.0) -> pl.DataFrame:
    """Generate N bars with linearly increasing prices."""
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


async def test_buy_hold_on_rising_market():
    """Buy & hold on linearly rising prices should produce positive return."""
    engine = BacktestEngine()
    result = await engine.run(
        algorithm=BuyHoldAlgorithm(),
        symbol=_symbol(),
        timeframe=Timeframe.D1,
        start=datetime(2024, 1, 1),
        end=datetime(2024, 1, 20),
        initial_capital=10_000.0,
        commission_bps=10.0,
        data=_linear_ohlcv(20, 100.0),
    )

    assert result.final_equity > result.initial_capital
    assert result.kpis["total_return_pct"] > 0
    assert result.kpis["max_drawdown_pct"] <= 0  # Drawdown is ≤ 0 (negative)
    assert len(result.equity_curve) == 20
    assert len(result.trades) >= 1  # At least one buy


async def test_equity_curve_length_matches_bars():
    data = _linear_ohlcv(50)
    engine = BacktestEngine()
    result = await engine.run(
        algorithm=BuyHoldAlgorithm(),
        symbol=_symbol(),
        timeframe=Timeframe.D1,
        start=datetime(2024, 1, 1),
        end=datetime(2024, 2, 19),
        data=data,
    )
    assert len(result.equity_curve) == 50


async def test_signals_are_captured():
    engine = BacktestEngine()
    result = await engine.run(
        algorithm=BuyHoldAlgorithm(),
        symbol=_symbol(),
        timeframe=Timeframe.D1,
        start=datetime(2024, 1, 1),
        end=datetime(2024, 1, 10),
        data=_linear_ohlcv(10),
    )
    # Buy & hold emits one signal per bar
    assert len(result.signals) == 10
    assert all(s.score == 1.0 for s in result.signals)


async def test_commission_reduces_equity():
    data = _linear_ohlcv(10, 100.0)
    engine = BacktestEngine()

    zero_comm = await engine.run(
        algorithm=BuyHoldAlgorithm(),
        symbol=_symbol(),
        timeframe=Timeframe.D1,
        start=datetime(2024, 1, 1),
        end=datetime(2024, 1, 10),
        data=data,
        commission_bps=0.0,
    )

    with_comm = await engine.run(
        algorithm=BuyHoldAlgorithm(),
        symbol=_symbol(),
        timeframe=Timeframe.D1,
        start=datetime(2024, 1, 1),
        end=datetime(2024, 1, 10),
        data=data,
        commission_bps=50.0,  # 0.5% — significant
    )

    assert with_comm.final_equity < zero_comm.final_equity


async def test_empty_data_returns_empty_result():
    engine = BacktestEngine()
    result = await engine.run(
        algorithm=BuyHoldAlgorithm(),
        symbol=_symbol(),
        timeframe=Timeframe.D1,
        start=datetime(2024, 1, 1),
        end=datetime(2024, 1, 10),
        data=pl.DataFrame(
            schema={
                "timestamp": pl.Datetime,
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Float64,
            }
        ),
    )
    assert result.equity_curve == []
    assert result.final_equity == 10_000.0


async def test_kpis_are_populated():
    engine = BacktestEngine()
    result = await engine.run(
        algorithm=BuyHoldAlgorithm(),
        symbol=_symbol(),
        timeframe=Timeframe.D1,
        start=datetime(2024, 1, 1),
        end=datetime(2024, 1, 20),
        data=_linear_ohlcv(20),
    )
    kpis = result.kpis
    assert "total_return_pct" in kpis
    assert "sharpe_ratio" in kpis
    assert "max_drawdown_pct" in kpis
    assert "num_trades" in kpis
    assert kpis["num_trades"] >= 1
