"""Shared technical indicator functions — pure numpy, no external deps.

Used by built-in algorithms for signal generation. Every function
takes raw numpy arrays and returns a numpy array of the same length
(with NaN for warmup bars where the indicator is undefined).
"""

from __future__ import annotations

import numpy as np


def sma(data: np.ndarray, period: int) -> np.ndarray:
    """Simple moving average."""
    out = np.full(len(data), np.nan)
    if len(data) < period:
        return out
    cumsum = np.cumsum(data)
    cumsum[period:] = cumsum[period:] - cumsum[:-period]
    out[period - 1 :] = cumsum[period - 1 :] / period
    return out


def ema(data: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average (EMA).

    Initializes with the SMA of the first ``period`` valid (non-NaN)
    values, then applies the standard multiplier ``k = 2 / (period + 1)``.
    Leading NaN values are skipped so chaining EMAs (e.g. signal line in
    MACD) works correctly.
    """
    out = np.full(len(data), np.nan)

    # Find the first index where we have `period` consecutive non-NaN values
    valid_mask = ~np.isnan(data)
    if valid_mask.sum() < period:
        return out

    # Find the start of the first run of non-NaN values that's long enough
    start = 0
    while start < len(data) and np.isnan(data[start]):
        start += 1

    if start + period > len(data):
        return out

    k = 2.0 / (period + 1)
    # Seed with SMA of first `period` valid values
    out[start + period - 1] = float(np.mean(data[start : start + period]))
    for i in range(start + period, len(data)):
        if np.isnan(data[i]):
            out[i] = out[i - 1]  # hold last value through any NaN gaps
        else:
            out[i] = data[i] * k + out[i - 1] * (1 - k)
    return out


def rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    """Relative Strength Index using Wilder smoothing."""
    out = np.full(len(closes), np.nan)
    if len(closes) < period + 1:
        return out

    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            out[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            out[i + 1] = 100.0 - 100.0 / (1.0 + rs)

    return out


def true_range(
    highs: np.ndarray, lows: np.ndarray, closes: np.ndarray
) -> np.ndarray:
    """True Range series. First element is NaN (needs previous close)."""
    n = len(closes)
    tr = np.full(n, np.nan)
    if n < 2:
        return tr
    for i in range(1, n):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        tr[i] = max(hl, hc, lc)
    return tr


def atr(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """Average True Range series using Wilder smoothing."""
    tr = true_range(highs, lows, closes)
    out = np.full(len(closes), np.nan)
    if len(closes) < period + 1:
        return out

    # SMA of first `period` true ranges (starting from index 1)
    first_atr = np.mean(tr[1 : period + 1])
    out[period] = first_atr
    alpha = 1.0 / period
    for i in range(period + 1, len(closes)):
        out[i] = out[i - 1] * (1 - alpha) + tr[i] * alpha
    return out


def stochastic(data: np.ndarray, period: int) -> np.ndarray:
    """Stochastic oscillator %K (raw, unsmoothed) in [0, 1]."""
    out = np.full(len(data), np.nan)
    if len(data) < period:
        return out
    for i in range(period - 1, len(data)):
        window = data[i - period + 1 : i + 1]
        lo = np.min(window)
        hi = np.max(window)
        if hi - lo > 0:
            out[i] = (data[i] - lo) / (hi - lo)
        else:
            out[i] = 0.5
    return out


def rolling_std(data: np.ndarray, period: int) -> np.ndarray:
    """Rolling standard deviation."""
    out = np.full(len(data), np.nan)
    if len(data) < period:
        return out
    for i in range(period - 1, len(data)):
        out[i] = np.std(data[i - period + 1 : i + 1], ddof=0)
    return out


# ---------------------------------------------------------------------------
# Phase 3 indicators
# ---------------------------------------------------------------------------


def ichimoku_lines(
    highs: np.ndarray,
    lows: np.ndarray,
    tenkan_period: int = 9,
    kijun_period: int = 26,
    span_b_period: int = 52,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Ichimoku Cloud lines: tenkan, kijun, span_a, span_b.

    Returns four arrays of the same length as input. NaN for warmup bars.
    Senkou spans are **not** displaced — the caller handles displacement.
    """
    n = len(highs)

    def _midpoint(h: np.ndarray, l: np.ndarray, period: int) -> np.ndarray:
        out = np.full(n, np.nan)
        for i in range(period - 1, n):
            window_h = h[i - period + 1 : i + 1]
            window_l = l[i - period + 1 : i + 1]
            out[i] = (np.max(window_h) + np.min(window_l)) / 2.0
        return out

    tenkan = _midpoint(highs, lows, tenkan_period)
    kijun = _midpoint(highs, lows, kijun_period)
    span_a = np.full(n, np.nan)
    span_b = _midpoint(highs, lows, span_b_period)
    # span_a = (tenkan + kijun) / 2 where both are valid
    valid = ~(np.isnan(tenkan) | np.isnan(kijun))
    span_a[valid] = (tenkan[valid] + kijun[valid]) / 2.0
    return tenkan, kijun, span_a, span_b


def williams_r(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """Williams %R in range [-100, 0]."""
    out = np.full(len(closes), np.nan)
    if len(closes) < period:
        return out
    for i in range(period - 1, len(closes)):
        hh = np.max(highs[i - period + 1 : i + 1])
        ll = np.min(lows[i - period + 1 : i + 1])
        if hh - ll > 0:
            out[i] = -100.0 * (hh - closes[i]) / (hh - ll)
        else:
            out[i] = -50.0
    return out


def cci(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = 20,
) -> np.ndarray:
    """Commodity Channel Index."""
    tp = (highs + lows + closes) / 3.0
    out = np.full(len(closes), np.nan)
    if len(closes) < period:
        return out
    for i in range(period - 1, len(closes)):
        window = tp[i - period + 1 : i + 1]
        mean = np.mean(window)
        mean_dev = np.mean(np.abs(window - mean))
        if mean_dev > 0:
            out[i] = (tp[i] - mean) / (0.015 * mean_dev)
        else:
            out[i] = 0.0
    return out


def obv(closes: np.ndarray, volumes: np.ndarray) -> np.ndarray:
    """On-Balance Volume (cumulative)."""
    n = len(closes)
    out = np.full(n, np.nan)
    if n < 2:
        return out
    out[0] = 0.0
    for i in range(1, n):
        if closes[i] > closes[i - 1]:
            out[i] = out[i - 1] + volumes[i]
        elif closes[i] < closes[i - 1]:
            out[i] = out[i - 1] - volumes[i]
        else:
            out[i] = out[i - 1]
    return out


def tema(data: np.ndarray, period: int) -> np.ndarray:
    """Triple Exponential Moving Average: 3*EMA1 - 3*EMA2 + EMA3.

    Requires roughly ``3 * period`` bars before producing valid values
    because each EMA layer needs its own warmup.
    """
    n = len(data)
    out = np.full(n, np.nan)
    if n < period:
        return out

    ema1 = ema(data, period)

    # For EMA2, feed only the valid portion of ema1
    start1 = period - 1  # first valid index in ema1
    if n - start1 < period:
        return out
    ema2_slice = ema(ema1[start1:], period)
    ema2 = np.full(n, np.nan)
    ema2[start1:] = ema2_slice

    # For EMA3, feed only the valid portion of ema2
    start2 = start1 + (period - 1)
    if n - start2 < period:
        return out
    ema3_slice = ema(ema2[start2:], period)
    ema3 = np.full(n, np.nan)
    ema3[start2:] = ema3_slice

    valid = ~(np.isnan(ema1) | np.isnan(ema2) | np.isnan(ema3))
    out[valid] = 3.0 * ema1[valid] - 3.0 * ema2[valid] + ema3[valid]
    return out


def zscore(data: np.ndarray, period: int) -> np.ndarray:
    """Rolling z-score: (data - rolling_mean) / rolling_std."""
    out = np.full(len(data), np.nan)
    if len(data) < period:
        return out
    mean = sma(data, period)
    std = rolling_std(data, period)
    valid = (~np.isnan(mean)) & (~np.isnan(std)) & (std > 0)
    out[valid] = (data[valid] - mean[valid]) / std[valid]
    return out
