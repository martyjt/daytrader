"""Tests for shared technical indicator functions."""

import numpy as np

from daytrader.algorithms.indicators import (
    atr,
    cci,
    ema,
    ichimoku_lines,
    obv,
    rolling_std,
    rsi,
    sma,
    stochastic,
    tema,
    true_range,
    williams_r,
    zscore,
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


# ---------------------------------------------------------------------------
# Phase 3 indicators
# ---------------------------------------------------------------------------


def test_ichimoku_lines_shape():
    n = 100
    highs = np.arange(101.0, 101.0 + n)
    lows = np.arange(99.0, 99.0 + n)
    tenkan, kijun, span_a, span_b = ichimoku_lines(highs, lows)
    assert len(tenkan) == n
    assert len(kijun) == n
    assert len(span_a) == n
    assert len(span_b) == n


def test_ichimoku_lines_warmup():
    """span_b should have NaN for first span_b_period - 1 bars."""
    n = 100
    highs = np.arange(101.0, 101.0 + n)
    lows = np.arange(99.0, 99.0 + n)
    _, _, _, span_b = ichimoku_lines(highs, lows, span_b_period=52)
    assert np.isnan(span_b[50])
    assert not np.isnan(span_b[51])


def test_williams_r_range():
    """Williams %R should be in [-100, 0]."""
    np.random.seed(42)
    n = 100
    closes = 100 + np.cumsum(np.random.randn(n))
    highs = closes + np.abs(np.random.randn(n))
    lows = closes - np.abs(np.random.randn(n))
    result = williams_r(highs, lows, closes, 14)
    valid = result[~np.isnan(result)]
    assert len(valid) > 0
    assert all(-100.0 <= v <= 0.0 for v in valid)


def test_williams_r_at_high():
    """Price at period high → %R should be 0."""
    highs = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
    lows = np.array([9.0, 10.0, 11.0, 12.0, 13.0])
    closes = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
    result = williams_r(highs, lows, closes, 5)
    assert result[-1] == 0.0


def test_cci_known():
    """CCI should be around 0 for flat data."""
    data = np.ones(50) * 100.0
    result = cci(data, data, data, 20)
    valid = result[~np.isnan(result)]
    assert len(valid) > 0
    assert all(abs(v) < 1.0 for v in valid)


def test_cci_trending():
    """CCI should be positive for strongly rising data."""
    n = 50
    closes = np.arange(100.0, 100.0 + n)
    highs = closes + 1
    lows = closes - 1
    result = cci(highs, lows, closes, 20)
    assert result[-1] > 50  # positive during uptrend


def test_obv_rising():
    """OBV should increase when price rises."""
    closes = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
    volumes = np.array([1000.0, 1000.0, 1000.0, 1000.0, 1000.0])
    result = obv(closes, volumes)
    assert result[0] == 0.0
    assert result[-1] == 4000.0  # 4 up bars × 1000


def test_obv_falling():
    """OBV should decrease when price falls."""
    closes = np.array([104.0, 103.0, 102.0, 101.0, 100.0])
    volumes = np.array([1000.0, 1000.0, 1000.0, 1000.0, 1000.0])
    result = obv(closes, volumes)
    assert result[-1] == -4000.0


def test_tema_converges():
    """TEMA on constant data should converge to that constant."""
    data = np.ones(100) * 50.0
    result = tema(data, 10)
    valid = result[~np.isnan(result)]
    assert len(valid) > 0
    assert abs(valid[-1] - 50.0) < 0.01


def test_tema_responsive():
    """TEMA should be closer to current price than EMA."""
    data = np.arange(1.0, 101.0)
    t = tema(data, 10)
    e = ema(data, 10)
    # TEMA is more responsive (less lag) so should be closer to current
    t_valid = t[~np.isnan(t)]
    e_valid = e[~np.isnan(e)]
    assert abs(data[-1] - t_valid[-1]) < abs(data[-1] - e_valid[-1])


def test_zscore_zero_for_constant():
    """Z-score of constant data should be 0 (or NaN if std==0)."""
    data = np.ones(30) * 100.0
    result = zscore(data, 10)
    # std is 0, so zscore should be NaN (division by zero guarded)
    valid = result[~np.isnan(result)]
    assert len(valid) == 0  # all NaN because std == 0


def test_zscore_range():
    """Z-score should be reasonable for normal data."""
    np.random.seed(42)
    data = 100 + np.cumsum(np.random.randn(100) * 0.5)
    result = zscore(data, 20)
    valid = result[~np.isnan(result)]
    assert len(valid) > 0
    # Z-scores should be within reasonable bounds
    assert all(abs(v) < 10 for v in valid)
