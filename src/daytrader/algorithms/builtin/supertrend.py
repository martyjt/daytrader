"""Supertrend — ATR-based dynamic support/resistance trend follower.

Computes a Supertrend indicator from ATR and median price. The upper and
lower bands tighten monotonically (final upper can only move down while
price stays below it; final lower can only move up while price stays
above it). When price crosses through a band the trend flips, and the
algorithm emits a signal only on that flip.

Best suited for trending markets on higher timeframes; will whipsaw in
choppy, sideways conditions.
"""

from __future__ import annotations

import numpy as np

from ...core.context import AlgorithmContext
from ...core.types.signals import Signal
from ..base import Algorithm, AlgorithmManifest, AlgorithmParam
from ..indicators import atr


class SupertrendAlgorithm(Algorithm):
    def __init__(self) -> None:
        self._prev_trend: int | None = None  # +1 = UP, -1 = DOWN

    @property
    def manifest(self) -> AlgorithmManifest:
        return AlgorithmManifest(
            id="supertrend",
            name="Supertrend",
            version="1.0.0",
            description=(
                "ATR-based dynamic support/resistance that emits signals "
                "only on trend flips. Bullish when price crosses above the "
                "upper band, bearish when it crosses below the lower band."
            ),
            asset_classes=["crypto", "equities"],
            timeframes=["1h", "4h", "1d"],
            params=[
                AlgorithmParam(
                    name="atr_period",
                    type="int",
                    default=10,
                    min=5,
                    max=50,
                    description="ATR lookback period",
                ),
                AlgorithmParam(
                    name="multiplier",
                    type="float",
                    default=3.0,
                    min=1.0,
                    max=5.0,
                    step=0.5,
                    description="ATR multiplier for band width",
                ),
            ],
            author="Daytrader built-in",
        )

    def warmup_bars(self) -> int:
        return int(self.manifest.param_defaults()["atr_period"]) + 1

    def on_bar(self, ctx: AlgorithmContext) -> Signal | None:
        closes = ctx.history_arrays["close"]
        highs = ctx.history_arrays["high"]
        lows = ctx.history_arrays["low"]

        atr_period = int(ctx.param("atr_period", 10))
        multiplier = float(ctx.param("multiplier", 3.0))

        if len(closes) < atr_period + 1:
            return None

        # Compute ATR series
        atr_series = atr(highs, lows, closes, atr_period)

        # Compute basic bands
        hl2 = (highs + lows) / 2.0
        basic_upper = hl2 + multiplier * atr_series
        basic_lower = hl2 - multiplier * atr_series

        n = len(closes)

        # Build final upper and lower band arrays
        final_upper = np.full(n, np.nan)
        final_lower = np.full(n, np.nan)
        trend = np.zeros(n, dtype=int)  # +1 = UP, -1 = DOWN

        # Initialize at the first valid ATR bar
        start = atr_period
        if np.isnan(basic_upper[start]) or np.isnan(basic_lower[start]):
            return None

        final_upper[start] = basic_upper[start]
        final_lower[start] = basic_lower[start]
        trend[start] = 1  # assume UP initially

        for i in range(start + 1, n):
            if np.isnan(basic_upper[i]) or np.isnan(basic_lower[i]):
                final_upper[i] = final_upper[i - 1]
                final_lower[i] = final_lower[i - 1]
                trend[i] = trend[i - 1]
                continue

            # Final upper band: tighten down if price stayed below
            if closes[i - 1] <= final_upper[i - 1]:
                final_upper[i] = min(basic_upper[i], final_upper[i - 1])
            else:
                final_upper[i] = basic_upper[i]

            # Final lower band: tighten up if price stayed above
            if closes[i - 1] >= final_lower[i - 1]:
                final_lower[i] = max(basic_lower[i], final_lower[i - 1])
            else:
                final_lower[i] = basic_lower[i]

            # Determine trend
            if closes[i] > final_upper[i]:
                trend[i] = 1  # UP
            elif closes[i] < final_lower[i]:
                trend[i] = -1  # DOWN
            else:
                trend[i] = trend[i - 1]

        current_trend = int(trend[-1])
        supertrend_value = float(
            final_lower[-1] if current_trend == 1 else final_upper[-1]
        )

        ctx.log(
            "supertrend",
            trend="UP" if current_trend == 1 else "DOWN",
            supertrend_value=supertrend_value,
            final_upper=float(final_upper[-1]),
            final_lower=float(final_lower[-1]),
            atr_value=float(atr_series[-1]) if not np.isnan(atr_series[-1]) else 0.0,
        )

        # Only emit on trend FLIP
        if self._prev_trend is not None and current_trend != self._prev_trend:
            self._prev_trend = current_trend
            if current_trend == 1:
                return ctx.emit(
                    score=0.8,
                    confidence=0.8,
                    reason=(
                        f"Supertrend flipped UP — close={closes[-1]:.4f} "
                        f"broke above upper band, supertrend={supertrend_value:.4f}"
                    ),
                )
            else:
                return ctx.emit(
                    score=-0.8,
                    confidence=0.8,
                    reason=(
                        f"Supertrend flipped DOWN — close={closes[-1]:.4f} "
                        f"broke below lower band, supertrend={supertrend_value:.4f}"
                    ),
                )

        self._prev_trend = current_trend
        return None
