"""EMA Crossover — fast/slow exponential moving average crossover.

Computes a fast EMA and a slow EMA on close prices. When the fast EMA
crosses above the slow EMA, the algorithm emits a bullish signal; when
it crosses below, a bearish signal. The score is proportional to the
spread between the two EMAs (normalized by the slow EMA value),
providing stronger signals when the crossover is more decisive.

This is a classic trend-following strategy that works best in trending
markets and tends to underperform in choppy, range-bound conditions.
"""

from __future__ import annotations

import numpy as np

from ..base import Algorithm, AlgorithmManifest, AlgorithmParam
from ..indicators import ema
from ...core.context import AlgorithmContext
from ...core.types.signals import Signal
from ...core.types.visualize import (
    PlotTrace,
    VisualizeContext,
    nan_array_to_jsonable,
)


class EMACrossoverAlgorithm(Algorithm):
    @property
    def manifest(self) -> AlgorithmManifest:
        return AlgorithmManifest(
            id="ema_crossover",
            name="EMA Crossover",
            version="1.0.0",
            description=(
                "Trend-following strategy based on fast/slow EMA crossover. "
                "Emits bullish signals when the fast EMA crosses above the "
                "slow EMA, and bearish signals on the opposite cross."
            ),
            asset_classes=["crypto", "equities"],
            timeframes=["1h", "4h", "1d"],
            params=[
                AlgorithmParam(
                    name="fast_period",
                    type="int",
                    default=9,
                    min=2,
                    max=200,
                    description="Period for the fast EMA",
                ),
                AlgorithmParam(
                    name="slow_period",
                    type="int",
                    default=21,
                    min=2,
                    max=500,
                    description="Period for the slow EMA",
                ),
            ],
            author="Daytrader built-in",
            suitable_regimes=["bull", "bear"],
        )

    def warmup_bars(self) -> int:
        return int(self.manifest.param_defaults()["slow_period"]) + 1

    def on_bar(self, ctx: AlgorithmContext) -> Signal | None:
        closes = ctx.history_arrays["close"]
        fast_period = int(ctx.param("fast_period", 9))
        slow_period = int(ctx.param("slow_period", 21))

        if len(closes) < slow_period + 1:
            return None

        fast_ema = ema(closes, fast_period)
        slow_ema = ema(closes, slow_period)

        # Need at least two valid values to detect a crossover
        if np.isnan(fast_ema[-1]) or np.isnan(fast_ema[-2]):
            return None
        if np.isnan(slow_ema[-1]) or np.isnan(slow_ema[-2]):
            return None

        spread = float(fast_ema[-1] - slow_ema[-1])
        prev_spread = float(fast_ema[-2] - slow_ema[-2])

        ctx.log(
            "ema_crossover",
            fast_ema=float(fast_ema[-1]),
            slow_ema=float(slow_ema[-1]),
            spread=spread,
            prev_spread=prev_spread,
        )

        # Bullish crossover: fast crosses above slow
        if fast_ema[-1] > slow_ema[-1] and fast_ema[-2] <= slow_ema[-2]:
            # Score proportional to spread magnitude, normalized by slow EMA
            normalizer = abs(float(slow_ema[-1])) if slow_ema[-1] != 0 else 1.0
            score = min(1.0, abs(spread) / normalizer * 50)
            score = max(0.51, score)  # at least above the BUY threshold
            return ctx.emit(
                score=score,
                confidence=min(1.0, abs(spread) / normalizer * 100),
                reason=f"bullish EMA cross: fast({fast_period})={fast_ema[-1]:.4f} > slow({slow_period})={slow_ema[-1]:.4f}, spread={spread:.4f}",
            )

        # Bearish crossover: fast crosses below slow
        if fast_ema[-1] < slow_ema[-1] and fast_ema[-2] >= slow_ema[-2]:
            normalizer = abs(float(slow_ema[-1])) if slow_ema[-1] != 0 else 1.0
            score = min(1.0, abs(spread) / normalizer * 50)
            score = max(0.51, score)
            return ctx.emit(
                score=-score,
                confidence=min(1.0, abs(spread) / normalizer * 100),
                reason=f"bearish EMA cross: fast({fast_period})={fast_ema[-1]:.4f} < slow({slow_period})={slow_ema[-1]:.4f}, spread={spread:.4f}",
            )

        return None

    def visualize(self, vctx: VisualizeContext) -> list[PlotTrace]:
        fast_period = int(vctx.params.get("fast_period", 9))
        slow_period = int(vctx.params.get("slow_period", 21))
        fast = ema(vctx.closes, fast_period)
        slow = ema(vctx.closes, slow_period)
        return [
            PlotTrace(
                name=f"Fast EMA ({fast_period})",
                kind="line",
                data=nan_array_to_jsonable(fast),
                panel="price",
                color="#22b8cf",
            ),
            PlotTrace(
                name=f"Slow EMA ({slow_period})",
                kind="line",
                data=nan_array_to_jsonable(slow),
                panel="price",
                color="#f76707",
            ),
        ]
