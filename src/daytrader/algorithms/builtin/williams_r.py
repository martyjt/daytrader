"""Williams %R oscillator algorithm.

Mean-reversion strategy using the Williams %R indicator.
Emits bullish signals in oversold territory and bearish in overbought.
"""

from __future__ import annotations

from ..base import Algorithm, AlgorithmManifest, AlgorithmParam
from ..indicators import williams_r as _williams_r
from ...core.context import AlgorithmContext
from ...core.types.signals import Signal


class WilliamsRAlgorithm(Algorithm):
    """Williams %R oversold/overbought mean reversion."""

    def __init__(
        self,
        *,
        period: int = 14,
        oversold: float = -80.0,
        overbought: float = -20.0,
    ) -> None:
        self._period = period
        self._oversold = oversold
        self._overbought = overbought

    @property
    def manifest(self) -> AlgorithmManifest:
        return AlgorithmManifest(
            id="williams_r",
            name="Williams %R",
            version="1.0.0",
            description="Mean-reversion: oversold/overbought via Williams %R [-100, 0].",
            asset_classes=["crypto", "equities"],
            timeframes=["5m", "15m", "1h", "4h", "1d"],
            params=[
                AlgorithmParam("period", "int", self._period, min=5, max=50, description="Lookback period"),
                AlgorithmParam("oversold", "float", self._oversold, min=-95.0, max=-60.0, step=5.0, description="Oversold threshold"),
                AlgorithmParam("overbought", "float", self._overbought, min=-40.0, max=-5.0, step=5.0, description="Overbought threshold"),
            ],
            author="Daytrader built-in",
        )

    def warmup_bars(self) -> int:
        return self._period + 1

    def on_bar(self, ctx: AlgorithmContext) -> Signal | None:
        highs = ctx.history_arrays.get("high")
        lows = ctx.history_arrays.get("low")
        closes = ctx.history_arrays.get("close")
        if closes is None or len(closes) < self.warmup_bars():
            return None

        wr = _williams_r(highs, lows, closes, self._period)
        val = wr[-1]
        if val != val:  # NaN check
            return None

        if val <= self._oversold:
            depth = (self._oversold - val) / 100.0
            score = min(0.5 + depth * 5.0, 1.0)
            return ctx.emit(
                score=score,
                confidence=min(0.6 + depth * 3.0, 0.95),
                reason=f"Williams %R={val:.1f} oversold (<{self._oversold})",
            )

        if val >= self._overbought:
            depth = (val - self._overbought) / 100.0
            score = -min(0.5 + depth * 5.0, 1.0)
            return ctx.emit(
                score=score,
                confidence=min(0.6 + depth * 3.0, 0.95),
                reason=f"Williams %R={val:.1f} overbought (>{self._overbought})",
            )

        return None
