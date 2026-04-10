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

    Initializes with the SMA of the first ``period`` values, then
    applies the standard multiplier ``k = 2 / (period + 1)``.
    """
    out = np.full(len(data), np.nan)
    if len(data) < period:
        return out

    k = 2.0 / (period + 1)
    out[period - 1] = np.mean(data[:period])
    for i in range(period, len(data)):
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
