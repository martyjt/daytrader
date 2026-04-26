"""MACD Signal — histogram crossover strategy.

Uses the Moving Average Convergence/Divergence (MACD) indicator to
detect momentum shifts. The MACD line is the difference between a fast
and slow EMA; the signal line is an EMA of the MACD line; the histogram
is MACD minus signal.

When the histogram flips from negative to positive (MACD crossing above
its signal line), the algorithm emits a bullish signal. When it flips
from positive to negative, it emits a bearish signal. The score is
proportional to the histogram magnitude, providing stronger signals for
more decisive momentum shifts.

This is a momentum/trend strategy that combines aspects of trend
following and mean reversion through the signal-line crossover.
"""

from __future__ import annotations

import numpy as np

from ...core.context import AlgorithmContext
from ...core.types.signals import Signal
from ...core.types.visualize import (
    PlotTrace,
    VisualizeContext,
    nan_array_to_jsonable,
)
from ..base import Algorithm, AlgorithmManifest, AlgorithmParam
from ..indicators import ema


class MACDSignalAlgorithm(Algorithm):
    @property
    def manifest(self) -> AlgorithmManifest:
        return AlgorithmManifest(
            id="macd_signal",
            name="MACD Signal",
            version="1.0.0",
            description=(
                "Momentum strategy based on MACD histogram crossovers. "
                "Emits bullish signals when the MACD histogram flips "
                "from negative to positive, and bearish on the reverse."
            ),
            asset_classes=["crypto", "equities"],
            timeframes=["1h", "4h", "1d"],
            params=[
                AlgorithmParam(
                    name="fast_period",
                    type="int",
                    default=12,
                    min=2,
                    max=100,
                    description="Period for the fast EMA in MACD calculation",
                ),
                AlgorithmParam(
                    name="slow_period",
                    type="int",
                    default=26,
                    min=2,
                    max=200,
                    description="Period for the slow EMA in MACD calculation",
                ),
                AlgorithmParam(
                    name="signal_period",
                    type="int",
                    default=9,
                    min=2,
                    max=50,
                    description="Period for the signal line (EMA of MACD)",
                ),
            ],
            author="Daytrader built-in",
            suitable_regimes=["bull", "bear"],
        )

    def warmup_bars(self) -> int:
        defaults = self.manifest.param_defaults()
        return int(defaults["slow_period"]) + int(defaults["signal_period"])

    def on_bar(self, ctx: AlgorithmContext) -> Signal | None:
        closes = ctx.history_arrays["close"]
        fast_period = int(ctx.param("fast_period", 12))
        slow_period = int(ctx.param("slow_period", 26))
        signal_period = int(ctx.param("signal_period", 9))

        min_bars = slow_period + signal_period
        if len(closes) < min_bars:
            return None

        # MACD line = fast EMA - slow EMA
        fast_ema = ema(closes, fast_period)
        slow_ema = ema(closes, slow_period)
        macd_line = fast_ema - slow_ema

        # Signal line = EMA of the MACD line
        # We need to compute EMA only on the valid (non-NaN) portion of MACD
        signal_line = ema(macd_line, signal_period)

        # Histogram = MACD - signal
        histogram = macd_line - signal_line

        # Need at least two valid histogram values for crossover detection
        if np.isnan(histogram[-1]) or np.isnan(histogram[-2]):
            return None

        hist_now = float(histogram[-1])
        hist_prev = float(histogram[-2])
        macd_now = float(macd_line[-1])
        signal_now = float(signal_line[-1])

        ctx.log(
            "macd_signal",
            macd=macd_now,
            signal=signal_now,
            histogram=hist_now,
            prev_histogram=hist_prev,
        )

        # Normalize histogram magnitude for scoring.
        # Use the current close as a reference to make the score
        # scale-independent across different asset price levels.
        close_price = float(closes[-1])
        normalizer = abs(close_price) if close_price != 0 else 1.0

        # Bullish crossover: histogram flips from negative to positive
        if hist_now > 0 and hist_prev <= 0:
            score = min(1.0, abs(hist_now) / normalizer * 100)
            score = max(0.51, score)  # ensure above BUY threshold
            return ctx.emit(
                score=score,
                confidence=min(1.0, abs(hist_now) / normalizer * 200),
                reason=(
                    f"bullish MACD crossover: MACD={macd_now:.4f}, "
                    f"signal={signal_now:.4f}, histogram={hist_now:.4f}"
                ),
            )

        # Bearish crossover: histogram flips from positive to negative
        if hist_now < 0 and hist_prev >= 0:
            score = min(1.0, abs(hist_now) / normalizer * 100)
            score = max(0.51, score)
            return ctx.emit(
                score=-score,
                confidence=min(1.0, abs(hist_now) / normalizer * 200),
                reason=(
                    f"bearish MACD crossover: MACD={macd_now:.4f}, "
                    f"signal={signal_now:.4f}, histogram={hist_now:.4f}"
                ),
            )

        return None

    def visualize(self, vctx: VisualizeContext) -> list[PlotTrace]:
        fast_period = int(vctx.params.get("fast_period", 12))
        slow_period = int(vctx.params.get("slow_period", 26))
        signal_period = int(vctx.params.get("signal_period", 9))
        fast_ema = ema(vctx.closes, fast_period)
        slow_ema = ema(vctx.closes, slow_period)
        macd_line = fast_ema - slow_ema
        signal_line = ema(macd_line, signal_period)
        histogram = macd_line - signal_line
        return [
            PlotTrace(
                name="MACD",
                kind="line",
                data=nan_array_to_jsonable(macd_line),
                panel="own",
                color="#5c7cfa",
            ),
            PlotTrace(
                name="Signal",
                kind="line",
                data=nan_array_to_jsonable(signal_line),
                panel="own",
                color="#f76707",
            ),
            PlotTrace(
                name="Histogram",
                kind="histogram",
                data=nan_array_to_jsonable(histogram),
                panel="own",
                color="#868e96",
                reference_lines=(("0", 0.0, "#868e96"),),
            ),
        ]
