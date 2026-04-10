"""Tests for shared technical indicator functions."""

import numpy as np

from daytrader.algorithms.indicators import (
    atr,
    ema,
    rolling_std,
    rsi,
    sma,
    stochastic,
    true_range,
)


def test_sma_known_values():
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = sma(data, 3)
    assert np.isnan(result[0])
    assert np.isnan(result[1])
    assert result[2] == 2.0  # (1+2+3)/3
    assert result[3] == 3.0  # (2+3+4)/3
    assert result[4] == 4.0  # (3+4+5)/3


def test_sma_insufficient_data():
    data = np.array([1.0, 2.0])
    result = sma(data, 5)
    assert np.all(np.isnan(result))


def test_ema_converges():
    """EMA on constant data should converge to that constant."""
    data = np.ones(50) * 100.0
    result = ema(data, 10)
    assert not np.isnan(result[-1])
    assert abs(result[-1] - 100.0) < 0.01


def test_ema_responds_to_trend():
    """EMA on rising data should be below current price."""
    data = np.arange(1, 51, dtype=float)
    result = ema(data, 10)
    assert result[-1] < data[-1]
    assert result[-1] > result[-2]


def test_rsi_range():
    """RSI should always be in [0, 100]."""
    np.random.seed(42)
    closes = 100 + np.cumsum(np.random.randn(100))
    result = rsi(closes, 14)
    valid = result[~np.isnan(result)]
    assert len(valid) > 0
    assert all(0 <= v <= 100 for v in valid)


def test_rsi_trending_up():
    """Strongly rising prices should produce high RSI."""
    closes = np.arange(100, 200, dtype=float)  # 100 bars rising
    result = rsi(closes, 14)
    assert result[-1] > 90.0


def test_rsi_trending_down():
    """Strongly falling prices should produce low RSI."""
    closes = np.arange(200, 100, -1, dtype=float)  # 100 bars falling
    result = rsi(closes, 14)
    assert result[-1] < 10.0


def test_true_range_constant():
    """True range on constant prices should be the bar range."""
    n = 20
    highs = np.ones(n) * 110.0
    lows = np.ones(n) * 90.0
    closes = np.ones(n) * 100.0
    tr = true_range(highs, lows, closes)
    assert np.isnan(tr[0])
    assert tr[1] == 20.0


def test_atr_known():
    """ATR on constant-range bars should converge to the range."""
    n = 50
    highs = np.ones(n) * 110.0
    lows = np.ones(n) * 90.0
    closes = np.ones(n) * 100.0
    result = atr(highs, lows, closes, period=14)
    assert not np.isnan(result[-1])
    assert abs(result[-1] - 20.0) < 1.0


def test_stochastic_range():
    """Stochastic %K should be in [0, 1]."""
    np.random.seed(42)
    data = 100 + np.cumsum(np.random.randn(100))
    result = stochastic(data, 14)
    valid = result[~np.isnan(result)]
    assert len(valid) > 0
    assert all(0 <= v <= 1 for v in valid)


def test_stochastic_at_high():
    """When price is at the high of the period, stochastic should be 1."""
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = stochastic(data, 5)
    assert result[-1] == 1.0


def test_rolling_std_known():
    """Rolling std of constant data should be 0."""
    data = np.ones(20) * 50.0
    result = rolling_std(data, 10)
    assert not np.isnan(result[-1])
    assert result[-1] == 0.0
