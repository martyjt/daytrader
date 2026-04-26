"""ADX Trend Filter — directional movement strength filter.

Computes the Average Directional Index (ADX) along with +DI and -DI
to measure trend strength and direction. Only emits signals when ADX
exceeds a configurable threshold (indicating a trending market) and
the directional dominance changes.

ADX below threshold means the market lacks a clear trend — the
algorithm stays flat. Higher ADX values produce stronger conviction
scores, with a separate ``strong_adx`` level for high-confidence
signals.
"""

from __future__ import annotations

import numpy as np

from ...core.context import AlgorithmContext
from ...core.types.signals import Signal
from ..base import Algorithm, AlgorithmManifest, AlgorithmParam
from ..indicators import true_range


class ADXTrendFilterAlgorithm(Algorithm):
    def __init__(self) -> None:
        self._prev_direction: int | None = None  # +1 = bullish, -1 = bearish

    @property
    def manifest(self) -> AlgorithmManifest:
        return AlgorithmManifest(
            id="adx_trend_filter",
            name="ADX Trend Filter",
            version="1.0.0",
            description=(
                "Directional movement index filter that only trades when "
                "ADX indicates a strong trend. Bullish when +DI > -DI, "
                "bearish when -DI > +DI. Emits only on direction changes."
            ),
            asset_classes=["crypto", "equities"],
            timeframes=["1h", "4h", "1d"],
            params=[
                AlgorithmParam(
                    name="period",
                    type="int",
                    default=14,
                    min=5,
                    max=50,
                    description="DI/ADX smoothing period",
                ),
                AlgorithmParam(
                    name="adx_threshold",
                    type="float",
                    default=20,
                    min=10,
                    max=50,
                    description="Minimum ADX to consider a trend present",
                ),
                AlgorithmParam(
                    name="strong_adx",
                    type="float",
                    default=30,
                    min=20,
                    max=60,
                    description="ADX level considered a strong trend",
                ),
            ],
            author="Daytrader built-in",
        )

    def warmup_bars(self) -> int:
        return int(self.manifest.param_defaults()["period"]) * 3

    def on_bar(self, ctx: AlgorithmContext) -> Signal | None:
        highs = ctx.history_arrays["high"]
        lows = ctx.history_arrays["low"]
        closes = ctx.history_arrays["close"]

        period = int(ctx.param("period", 14))
        adx_threshold = float(ctx.param("adx_threshold", 20))
        strong_adx = float(ctx.param("strong_adx", 30))

        n = len(closes)
        if n < period * 3:
            return None

        # Compute +DM and -DM
        plus_dm = np.zeros(n)
        minus_dm = np.zeros(n)
        for i in range(1, n):
            up_move = highs[i] - highs[i - 1]
            down_move = lows[i - 1] - lows[i]
            if up_move > down_move and up_move > 0:
                plus_dm[i] = up_move
            if down_move > up_move and down_move > 0:
                minus_dm[i] = down_move

        # Compute True Range
        tr = true_range(highs, lows, closes)
        # Replace NaN at index 0 with 0 for smoothing
        tr[0] = 0.0

        # Wilder's smoothing for +DM, -DM, TR
        smoothed_plus_dm = np.zeros(n)
        smoothed_minus_dm = np.zeros(n)
        smoothed_tr = np.zeros(n)

        # Initialize with sum of first `period` values (starting from bar 1)
        if n < period + 1:
            return None
        smoothed_plus_dm[period] = np.sum(plus_dm[1 : period + 1])
        smoothed_minus_dm[period] = np.sum(minus_dm[1 : period + 1])
        smoothed_tr[period] = np.sum(tr[1 : period + 1])

        for i in range(period + 1, n):
            smoothed_plus_dm[i] = (
                smoothed_plus_dm[i - 1]
                - smoothed_plus_dm[i - 1] / period
                + plus_dm[i]
            )
            smoothed_minus_dm[i] = (
                smoothed_minus_dm[i - 1]
                - smoothed_minus_dm[i - 1] / period
                + minus_dm[i]
            )
            smoothed_tr[i] = (
                smoothed_tr[i - 1] - smoothed_tr[i - 1] / period + tr[i]
            )

        # +DI and -DI
        plus_di = np.zeros(n)
        minus_di = np.zeros(n)
        for i in range(period, n):
            if smoothed_tr[i] > 0:
                plus_di[i] = 100.0 * smoothed_plus_dm[i] / smoothed_tr[i]
                minus_di[i] = 100.0 * smoothed_minus_dm[i] / smoothed_tr[i]

        # DX
        dx = np.zeros(n)
        for i in range(period, n):
            di_sum = plus_di[i] + minus_di[i]
            if di_sum > 0:
                dx[i] = abs(plus_di[i] - minus_di[i]) / di_sum * 100.0

        # ADX — Wilder smooth of DX
        adx = np.zeros(n)
        adx_start = period * 2
        if n <= adx_start:
            return None
        adx[adx_start] = np.mean(dx[period : adx_start + 1])
        for i in range(adx_start + 1, n):
            adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period

        current_adx = float(adx[-1])
        current_plus_di = float(plus_di[-1])
        current_minus_di = float(minus_di[-1])

        ctx.log(
            "adx_trend_filter",
            adx=current_adx,
            plus_di=current_plus_di,
            minus_di=current_minus_di,
            adx_threshold=adx_threshold,
        )

        # No trend — ADX below threshold
        if current_adx < adx_threshold:
            return None

        # Determine direction
        if current_plus_di > current_minus_di:
            direction = 1  # bullish
        elif current_minus_di > current_plus_di:
            direction = -1  # bearish
        else:
            return None  # equal — no signal

        # Only emit on direction CHANGE
        if self._prev_direction is not None and direction == self._prev_direction:
            return None

        self._prev_direction = direction

        # Confidence scales with ADX
        if current_adx >= strong_adx:
            score_magnitude = 0.9
        else:
            # Linear interpolation between 0.7 and 0.9
            t = (current_adx - adx_threshold) / (strong_adx - adx_threshold)
            score_magnitude = 0.7 + 0.2 * min(1.0, max(0.0, t))

        score = score_magnitude * direction
        confidence = min(1.0, current_adx / 50.0)

        if direction == 1:
            reason = (
                f"ADX bullish: +DI={current_plus_di:.2f} > "
                f"-DI={current_minus_di:.2f}, ADX={current_adx:.2f}"
            )
        else:
            reason = (
                f"ADX bearish: -DI={current_minus_di:.2f} > "
                f"+DI={current_plus_di:.2f}, ADX={current_adx:.2f}"
            )

        return ctx.emit(score=score, confidence=confidence, reason=reason)
