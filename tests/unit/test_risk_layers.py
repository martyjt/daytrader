"""Tests for risk layers: ATR, stop-loss, take-profit, daily loss limits."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl

from daytrader.algorithms.builtin.buy_hold import BuyHoldAlgorithm
from daytrader.backtest.engine import BacktestEngine
from daytrader.backtest.risk import (
    DailyPnLTracker,
    RiskConfig,
    check_stop_loss,
    check_take_profit,
    compute_atr,
    stop_loss_price,
    take_profit_price,
)
from daytrader.core.types.bars import Timeframe
from daytrader.core.types.symbols import AssetClass, Symbol


def _symbol() -> Symbol:
    return Symbol("TEST", "USD", AssetClass.CRYPTO)


# ---------------------------------------------------------------------------
# compute_atr tests
# ---------------------------------------------------------------------------


def test_atr_with_known_data():
    """ATR on constant-range bars should equal the constant range."""
    n = 30
    highs = np.array([110.0] * n)
    lows = np.array([90.0] * n)
    closes = np.array([100.0] * n)
    atr = compute_atr(highs, lows, closes, period=14)
    # True range = 20 for every bar, ATR should converge to 20
    assert abs(atr - 20.0) < 1.0


def test_atr_insufficient_data():
    """ATR returns 0 with fewer bars than period + 1."""
    highs = np.array([110.0] * 10)
    lows = np.array([90.0] * 10)
    closes = np.array([100.0] * 10)
    atr = compute_atr(highs, lows, closes, period=14)
    assert atr == 0.0


def test_atr_positive_for_volatile_data():
    """ATR should be positive for data with price movement."""
    np.random.seed(42)
    closes = 100 + np.cumsum(np.random.randn(50))
    highs = closes + np.abs(np.random.randn(50)) * 2
    lows = closes - np.abs(np.random.randn(50)) * 2
    atr = compute_atr(highs, lows, closes, period=14)
    assert atr > 0


# ---------------------------------------------------------------------------
# check_stop_loss / check_take_profit tests
# ---------------------------------------------------------------------------


def test_stop_loss_triggered():
    """Price drops below entry - ATR*mult should trigger stop."""
    assert check_stop_loss(
        current_low=90.0, entry_price=100.0, atr=5.0, multiplier=2.0
    )


def test_stop_loss_not_triggered():
    """Price stays above stop level."""
    assert not check_stop_loss(
        current_low=95.0, entry_price=100.0, atr=5.0, multiplier=2.0
    )


def test_take_profit_triggered():
    """Price rises above entry + ATR*mult should trigger TP."""
    assert check_take_profit(
        current_high=125.0, entry_price=100.0, atr=5.0, multiplier=4.0
    )


def test_take_profit_not_triggered():
    """Price stays below TP level."""
    assert not check_take_profit(
        current_high=115.0, entry_price=100.0, atr=5.0, multiplier=4.0
    )


def test_stop_loss_price_calculation():
    assert stop_loss_price(100.0, 5.0, 2.0) == 90.0


def test_take_profit_price_calculation():
    assert take_profit_price(100.0, 5.0, 4.0) == 120.0


def test_stop_loss_zero_atr():
    """Zero ATR should not trigger stop-loss."""
    assert not check_stop_loss(current_low=50.0, entry_price=100.0, atr=0.0, multiplier=2.0)


# ---------------------------------------------------------------------------
# DailyPnLTracker tests
# ---------------------------------------------------------------------------


def test_daily_tracker_halts_on_loss():
    """Tracker should halt when daily loss exceeds limit."""
    tracker = DailyPnLTracker(initial_equity=10000.0, daily_loss_limit_pct=5.0)
    t1 = datetime(2024, 1, 1, 10, 0)
    t2 = datetime(2024, 1, 1, 11, 0)
    t3 = datetime(2024, 1, 1, 12, 0)

    assert not tracker.update(10000.0, t1)  # Start of day
    assert not tracker.update(9600.0, t2)   # 4% loss — still within limit
    assert tracker.update(9400.0, t3)       # 6% loss — halted


def test_daily_tracker_resets_on_new_day():
    """Tracker should reset halt status on a new calendar day."""
    tracker = DailyPnLTracker(initial_equity=10000.0, daily_loss_limit_pct=5.0)
    day1_t1 = datetime(2024, 1, 1, 10, 0)
    day1_t2 = datetime(2024, 1, 1, 12, 0)
    day2 = datetime(2024, 1, 2, 10, 0)

    tracker.update(10000.0, day1_t1)  # Initializes day
    tracker.update(9400.0, day1_t2)   # 6% loss — halted
    assert tracker.is_halted

    halted = tracker.update(9400.0, day2)  # New day — reset
    assert not halted
    assert not tracker.is_halted


def test_daily_tracker_stays_halted_same_day():
    """Once halted, tracker stays halted for the rest of the day."""
    tracker = DailyPnLTracker(initial_equity=10000.0, daily_loss_limit_pct=5.0)
    t1 = datetime(2024, 1, 1, 10, 0)
    t2 = datetime(2024, 1, 1, 11, 0)
    t3 = datetime(2024, 1, 1, 14, 0)

    tracker.update(10000.0, t1)  # Initializes day
    tracker.update(9400.0, t2)   # 6% loss — halted
    assert tracker.update(9800.0, t3)  # Still halted even if equity recovers


# ---------------------------------------------------------------------------
# RiskConfig tests
# ---------------------------------------------------------------------------


def test_risk_config_disabled():
    config = RiskConfig.disabled()
    assert not config.enabled


def test_risk_config_defaults():
    config = RiskConfig()
    assert config.stop_loss_atr_mult == 2.0
    assert config.take_profit_atr_mult == 4.0
    assert config.max_hold_bars == 500
    assert config.daily_loss_limit_pct == 5.0
    assert config.enabled


# ---------------------------------------------------------------------------
# Engine integration: risk layers
# ---------------------------------------------------------------------------


def _crash_then_recover(n: int = 50) -> pl.DataFrame:
    """Generate prices that rise, crash sharply, then recover.
    Designed to trigger stop-loss."""
    prices = []
    for i in range(n):
        if i < 20:
            prices.append(100.0 + i * 2)  # Rising: 100 → 138
        elif i < 25:
            prices.append(138.0 - (i - 20) * 15)  # Crash: 138 → 63
        else:
            prices.append(63.0 + (i - 25) * 1)  # Slow recovery

    return pl.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n)],
            "open": prices,
            "high": [p + 2 for p in prices],
            "low": [p - 2 for p in prices],
            "close": prices,
            "volume": [1000.0] * n,
        }
    )


def _steady_rise(n: int = 50) -> pl.DataFrame:
    """Generate steadily rising prices. Designed to trigger take-profit."""
    prices = [100.0 + i * 3 for i in range(n)]
    return pl.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n)],
            "open": prices,
            "high": [p + 5 for p in prices],
            "low": [p - 1 for p in prices],
            "close": prices,
            "volume": [1000.0] * n,
        }
    )


async def test_stop_loss_triggers_exit():
    """With risk enabled, a price crash should trigger a stop-loss exit."""
    engine = BacktestEngine()
    risk = RiskConfig(stop_loss_atr_mult=1.5, take_profit_atr_mult=10.0, max_hold_bars=500)
    result = await engine.run(
        algorithm=BuyHoldAlgorithm(),
        symbol=_symbol(),
        timeframe=Timeframe.D1,
        start=datetime(2024, 1, 1),
        end=datetime(2024, 2, 19),
        data=_crash_then_recover(50),
        risk_config=risk,
        commission_bps=0,
    )
    sl_events = [e for e in result.risk_events if e["type"] == "stop_loss"]
    assert len(sl_events) >= 1
    sl_trades = [t for t in result.trades if t.get("exit_reason") == "stop_loss"]
    assert len(sl_trades) >= 1


async def test_take_profit_triggers_exit():
    """With risk enabled, a strong rise should trigger take-profit."""
    engine = BacktestEngine()
    risk = RiskConfig(stop_loss_atr_mult=10.0, take_profit_atr_mult=1.5, max_hold_bars=500)
    result = await engine.run(
        algorithm=BuyHoldAlgorithm(),
        symbol=_symbol(),
        timeframe=Timeframe.D1,
        start=datetime(2024, 1, 1),
        end=datetime(2024, 2, 19),
        data=_steady_rise(50),
        risk_config=risk,
        commission_bps=0,
    )
    tp_events = [e for e in result.risk_events if e["type"] == "take_profit"]
    assert len(tp_events) >= 1
    tp_trades = [t for t in result.trades if t.get("exit_reason") == "take_profit"]
    assert len(tp_trades) >= 1


async def test_risk_disabled_no_forced_exits():
    """With risk disabled, no forced exits should occur."""
    engine = BacktestEngine()
    result = await engine.run(
        algorithm=BuyHoldAlgorithm(),
        symbol=_symbol(),
        timeframe=Timeframe.D1,
        start=datetime(2024, 1, 1),
        end=datetime(2024, 2, 19),
        data=_crash_then_recover(50),
        risk_config=RiskConfig.disabled(),
        commission_bps=0,
    )
    assert result.risk_events == []
    risk_trades = [t for t in result.trades if "exit_reason" in t]
    assert len(risk_trades) == 0


async def test_backward_compat_no_risk_config():
    """Without risk_config param, engine should work as before (no risk exits)."""
    engine = BacktestEngine()
    result = await engine.run(
        algorithm=BuyHoldAlgorithm(),
        symbol=_symbol(),
        timeframe=Timeframe.D1,
        start=datetime(2024, 1, 1),
        end=datetime(2024, 2, 19),
        data=_crash_then_recover(50),
        commission_bps=0,
    )
    assert result.risk_events == []
