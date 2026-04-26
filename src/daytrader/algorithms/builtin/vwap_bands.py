"""VWAP Bands — volume-weighted average price with deviation bands.

Computes a rolling VWAP from typical price and volume, then adds
standard-deviation bands around it. Price at or below the lower band
triggers a bullish (mean-reversion) signal; price at or above the
upper band triggers a bearish signal. Score scales with the distance
beyond the band.
"""

from __future__ import annotations

import numpy as np

from ...core.context import AlgorithmContext
from ...core.types.signals import Signal
from ..base import Algorithm, AlgorithmManifest, AlgorithmParam


def _rolling_vwap(
    typical: np.ndarray, volume: np.ndarray, window: int
) -> np.ndarray:
    """Rolling VWAP = sum(typical * volume) / sum(volume) over *window*."""
    n = len(typical)
    out = np.full(n, np.nan)
    if n < window:
        return out
    tv = typical * volume
    cum_tv = np.cumsum(tv)
    cum_v = np.cumsum(volume)
    cum_tv[window:] = cum_tv[window:] - cum_tv[:-window]
    cum_v[window:] = cum_v[window:] - cum_v[:-window]
    # First valid index is window - 1
    valid = slice(window - 1, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        out[valid] = np.where(
            cum_v[valid] > 0,
            cum_tv[valid] / cum_v[valid],
            np.nan,
        )
    return out


def _rolling_std_dev(
    data: np.ndarray, ref: np.ndarray, window: int
) -> np.ndarray:
    """Rolling standard deviation of (data - ref) over *window*."""
    n = len(data)
    out = np.full(n, np.nan)
    if n < window:
        return out
    diff = data - ref
    for i in range(window - 1, n):
        segment = diff[i - window + 1 : i + 1]
        if np.any(np.isnan(segment)):
            continue
        out[i] = np.std(segment, ddof=0)
    return out


class VWAPBandsAlgorithm(Algorithm):
    """Mean-reversion strategy using rolling VWAP with deviation bands."""

    def __init__(
        self,
        *,
        window: int = 20,
        std_dev: float = 2.0,
    ) -> None:
        self._window = window
        self._std_dev = std_dev

    @property
    def manifest(self) -> AlgorithmManifest:
        return AlgorithmManifest(
            id="vwap_bands",
            name="VWAP Bands",
            version="1.0.0",
            description=(
                "Rolling VWAP with standard-deviation bands. Emits "
                "bullish signals when price drops to the lower band "
                "and bearish signals at the upper band. Score scales "
                "with distance beyond the band."
            ),
            asset_classes=["crypto", "equities", "forex", "commodities"],
            timeframes=["5m", "15m", "30m", "1h", "4h", "1d"],
            params=[
                AlgorithmParam(
                    "window", "int", self._window,
                    min=5, max=100,
                    description="Rolling VWAP lookback window",
                ),
                AlgorithmParam(
                    "std_dev", "float", self._std_dev,
                    min=0.5, max=4.0, step=0.5,
                    description="Number of standard deviations for bands",
                ),
            ],
            author="Daytrader built-in",
        )

    def warmup_bars(self) -> int:
        return self._window

    def on_bar(self, ctx: AlgorithmContext) -> Signal | None:
        window = ctx.param("window", self._window)
        std_dev = ctx.param("std_dev", self._std_dev)

        closes = ctx.history_arrays["close"]
        highs = ctx.history_arrays["high"]
        lows = ctx.history_arrays["low"]
        volumes = ctx.history_arrays["volume"]

        if (
            closes is None
            or highs is None
            or lows is None
            or volumes is None
            or len(closes) < window
        ):
            return None

        # Step 1: Typical price
        typical = (highs + lows + closes) / 3.0

        # Step 2: Rolling VWAP
        vwap = _rolling_vwap(typical, volumes, window)

        # Step 3: Rolling std dev of (typical - VWAP)
        std = _rolling_std_dev(typical, vwap, window)

        vwap_val = vwap[-1]
        std_val = std[-1]
        if np.isnan(vwap_val) or np.isnan(std_val) or std_val == 0:
            return None

        # Step 4: Bands
        upper = vwap_val + std_dev * std_val
        lower = vwap_val - std_dev * std_val
        price = float(closes[-1])

        ctx.log(
            "vwap_bands",
            price=price,
            vwap=float(vwap_val),
            upper_band=float(upper),
            lower_band=float(lower),
            std=float(std_val),
        )

        band_width = upper - lower
        if band_width <= 0:
            return None

        # Step 5: Signals
        if price <= lower:
            # Bullish mean-reversion
            distance = (lower - price) / band_width
            score = min(1.0, 0.5 + distance)
            return ctx.emit(
                score=score,
                confidence=min(1.0, distance + 0.5),
                reason=(
                    f"Price {price:.4f} at/below VWAP lower band "
                    f"{lower:.4f} (VWAP {vwap_val:.4f}, {std_dev}σ)"
                ),
            )

        if price >= upper:
            # Bearish mean-reversion
            distance = (price - upper) / band_width
            score = max(-1.0, -(0.5 + distance))
            return ctx.emit(
                score=score,
                confidence=min(1.0, distance + 0.5),
                reason=(
                    f"Price {price:.4f} at/above VWAP upper band "
                    f"{upper:.4f} (VWAP {vwap_val:.4f}, {std_dev}σ)"
                ),
            )

        # Price inside bands — no signal
        return None
