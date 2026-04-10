"""RSI Mean Reversion — oversold/overbought threshold strategy.

Uses the Relative Strength Index (RSI) to detect extreme price
conditions and bets on reversion to the mean. When RSI drops below the
oversold threshold, the algorithm emits a bullish signal (expecting a
bounce); when RSI rises above the overbought threshold, it emits a
bearish signal (expecting a pullback).

The signal score scales proportionally to how far the RSI has pushed
beyond the threshold: a barely-oversold reading produces a modest
signal, while an extremely oversold reading produces a strong one.

This strategy works best in range-bound markets and tends to
underperform during strong trends.
"""

from __future__ import annotations

from ..base import Algorithm, AlgorithmManifest, AlgorithmParam
from ..indicators import rsi as compute_rsi
from ...core.context import AlgorithmContext
from ...core.types.signals import Signal


class RSIMeanReversionAlgorithm(Algorithm):
    @property
    def manifest(self) -> AlgorithmManifest:
        return AlgorithmManifest(
            id="rsi_mean_reversion",
            name="RSI Mean Reversion",
            version="1.0.0",
            description=(
                "Mean-reversion strategy using RSI extremes. Emits bullish "
                "signals when RSI falls below the oversold threshold and "
                "bearish signals when RSI rises above the overbought threshold."
            ),
            asset_classes=["crypto", "equities"],
            timeframes=["1h", "4h", "1d"],
            params=[
                AlgorithmParam(
                    name="period",
                    type="int",
                    default=14,
                    min=2,
                    max=100,
                    description="RSI lookback period",
                ),
                AlgorithmParam(
                    name="oversold",
                    type="float",
                    default=30.0,
                    min=5.0,
                    max=50.0,
                    description="RSI level below which the asset is considered oversold",
                ),
                AlgorithmParam(
                    name="overbought",
                    type="float",
                    default=70.0,
                    min=50.0,
                    max=95.0,
                    description="RSI level above which the asset is considered overbought",
                ),
            ],
            author="Daytrader built-in",
        )

    def warmup_bars(self) -> int:
        return int(self.manifest.param_defaults()["period"]) + 1

    def on_bar(self, ctx: AlgorithmContext) -> Signal | None:
        closes = ctx.history_arrays["close"]
        period = int(ctx.param("period", 14))
        oversold = float(ctx.param("oversold", 30.0))
        overbought = float(ctx.param("overbought", 70.0))

        if len(closes) < period + 1:
            return None

        rsi_values = compute_rsi(closes, period)
        current_rsi = float(rsi_values[-1])

        # RSI might still be NaN if there is insufficient data
        if current_rsi != current_rsi:  # NaN check
            return None

        ctx.log("rsi_mean_reversion", rsi=current_rsi)

        # Oversold — bullish signal
        if current_rsi < oversold:
            # Distance below oversold, scaled so RSI=0 gives score=1.0
            distance = oversold - current_rsi
            score = min(1.0, distance / oversold)
            score = max(0.51, score)  # ensure above BUY threshold
            confidence = min(1.0, distance / oversold)
            return ctx.emit(
                score=score,
                confidence=confidence,
                reason=f"RSI oversold: RSI({period})={current_rsi:.1f} < {oversold:.0f}, distance={distance:.1f}",
            )

        # Overbought — bearish signal
        if current_rsi > overbought:
            # Distance above overbought, scaled so RSI=100 gives score=-1.0
            distance = current_rsi - overbought
            max_distance = 100.0 - overbought
            score = min(1.0, distance / max_distance) if max_distance > 0 else 1.0
            score = max(0.51, score)
            confidence = min(1.0, distance / max_distance) if max_distance > 0 else 1.0
            return ctx.emit(
                score=-score,
                confidence=confidence,
                reason=f"RSI overbought: RSI({period})={current_rsi:.1f} > {overbought:.0f}, distance={distance:.1f}",
            )

        # Neutral zone — no signal
        return None
