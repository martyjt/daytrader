"""Tests for the global risk layer 3."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from daytrader.risk.global_risk import GlobalRiskConfig, GlobalRiskMonitor

# ---------------------------------------------------------------------------
# GlobalRiskConfig
# ---------------------------------------------------------------------------


def test_config_defaults():
    cfg = GlobalRiskConfig()
    assert cfg.kill_switch_enabled is True
    assert cfg.max_drawdown_pct == 20.0
    assert cfg.data_staleness_seconds == 120


# ---------------------------------------------------------------------------
# Drawdown detection
# ---------------------------------------------------------------------------


async def test_no_breach_below_threshold():
    monitor = GlobalRiskMonitor(GlobalRiskConfig(max_drawdown_pct=20.0))

    assert await monitor.check_drawdown(100_000) is False
    assert await monitor.check_drawdown(95_000) is False  # 5% DD
    assert monitor.is_breached is False
    assert monitor.peak_equity == 100_000


async def test_breach_at_threshold():
    monitor = GlobalRiskMonitor(GlobalRiskConfig(max_drawdown_pct=20.0))

    await monitor.check_drawdown(100_000)
    result = await monitor.check_drawdown(80_000)  # Exactly 20%

    assert result is True
    assert monitor.is_breached is True


async def test_breach_above_threshold():
    monitor = GlobalRiskMonitor(GlobalRiskConfig(max_drawdown_pct=10.0))

    await monitor.check_drawdown(50_000)
    result = await monitor.check_drawdown(40_000)  # 20% DD > 10% limit

    assert result is True
    assert monitor.is_breached is True


async def test_peak_tracks_high_water_mark():
    monitor = GlobalRiskMonitor(GlobalRiskConfig(max_drawdown_pct=50.0))

    await monitor.check_drawdown(100_000)
    await monitor.check_drawdown(120_000)  # New peak
    await monitor.check_drawdown(110_000)  # Drawdown from 120k

    assert monitor.peak_equity == 120_000
    assert monitor.is_breached is False


async def test_zero_equity_no_crash():
    monitor = GlobalRiskMonitor(GlobalRiskConfig())
    assert await monitor.check_drawdown(0) is False


async def test_breach_callback_invoked():
    callback = AsyncMock()
    monitor = GlobalRiskMonitor(
        GlobalRiskConfig(max_drawdown_pct=10.0),
        on_breach=callback,
    )

    await monitor.check_drawdown(100_000)
    await monitor.check_drawdown(85_000)  # 15% DD > 10%

    callback.assert_called_once()
    args = callback.call_args
    assert args[0][0] == "max_drawdown"
    assert args[0][1]["drawdown_pct"] == pytest.approx(15.0)


# ---------------------------------------------------------------------------
# Data staleness
# ---------------------------------------------------------------------------


async def test_no_staleness_when_fresh():
    monitor = GlobalRiskMonitor(GlobalRiskConfig(data_staleness_seconds=120))
    now = datetime.now(UTC)
    monitor.record_data_timestamp("BTC/USDT", now)

    stale = await monitor.check_staleness()
    assert stale == []


async def test_staleness_detected():
    monitor = GlobalRiskMonitor(GlobalRiskConfig(data_staleness_seconds=60))
    old = datetime.now(UTC) - timedelta(seconds=120)
    monitor.record_data_timestamp("BTC/USDT", old)

    stale = await monitor.check_staleness()
    assert "BTC/USDT" in stale


async def test_staleness_callback_invoked():
    callback = AsyncMock()
    monitor = GlobalRiskMonitor(
        GlobalRiskConfig(data_staleness_seconds=10),
        on_breach=callback,
    )
    old = datetime.now(UTC) - timedelta(seconds=60)
    monitor.record_data_timestamp("ETH/USDT", old)

    await monitor.check_staleness()

    callback.assert_called_once()
    assert callback.call_args[0][0] == "data_staleness"


async def test_empty_timestamps_no_staleness():
    monitor = GlobalRiskMonitor(GlobalRiskConfig())
    assert await monitor.check_staleness() == []


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


async def test_reset_clears_state():
    monitor = GlobalRiskMonitor(GlobalRiskConfig(max_drawdown_pct=5.0))

    await monitor.check_drawdown(100_000)
    await monitor.check_drawdown(90_000)  # Breach
    assert monitor.is_breached is True

    monitor.reset()
    assert monitor.is_breached is False
    assert monitor.peak_equity == 0.0  # type: ignore[unreachable]


# ---------------------------------------------------------------------------
# current_drawdown_pct
# ---------------------------------------------------------------------------


async def test_current_drawdown_pct():
    monitor = GlobalRiskMonitor(GlobalRiskConfig(max_drawdown_pct=50.0))

    await monitor.check_drawdown(100_000)
    await monitor.check_drawdown(90_000)

    assert monitor.current_drawdown_pct == pytest.approx(10.0)
