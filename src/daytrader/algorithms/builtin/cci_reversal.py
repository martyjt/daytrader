"""Commodity Channel Index reversal algorithm.

Mean-reversion strategy using CCI at +/-100 thresholds.
"""

from __future__ import annotations

from ...core.context import AlgorithmContext
from ...core.types.signals import Signal
from ..base import Algorithm, AlgorithmManifest, AlgorithmParam
from ..indicators import cci as _cci


class CCIReversalAlgorithm(Algorithm):
    """CCI mean reversion at extreme levels."""

    def __init__(
        self,
        *,
        period: int = 20,
        oversold: float = -100.0,
        overbought: float = 100.0,
    ) -> None:
        self._period = period
        self._oversold = oversold
        self._overbought = overbought

    @property
    def manifest(self) -> AlgorithmManifest:
        return AlgorithmManifest(
            id="cci_reversal",
            name="CCI Reversal",
            version="1.0.0",
            description="Mean-reversion at CCI +/-100 thresholds.",
            asset_classes=["crypto", "equities"],
            timeframes=["5m", "15m", "1h", "4h", "1d"],
            params=[
                AlgorithmParam("period", "int", self._period, min=5, max=50, description="CCI period"),
                AlgorithmParam("oversold", "float", self._oversold, min=-300.0, max=-50.0, step=25.0, description="Oversold threshold"),
                AlgorithmParam("overbought", "float", self._overbought, min=50.0, max=300.0, step=25.0, description="Overbought threshold"),
            ],
            author="Daytrader built-in",
        )

    def warmup_bars(self) -> int:
        return self._period + 1

    def on_bar(self, ctx: AlgorithmContext) -> Signal | None:
        highs = ctx.history_arrays["high"]
        lows = ctx.history_arrays["low"]
        closes = ctx.history_arrays["close"]
        if closes is None or len(closes) < self.warmup_bars():
            return None

        vals = _cci(highs, lows, closes, self._period)
        val = vals[-1]
        if val != val:  # NaN
            return None

        if val <= self._oversold:
            depth = abs(val - self._oversold) / 100.0
            score = min(0.5 + depth * 0.5, 1.0)
            return ctx.emit(
                score=score,
                confidence=min(0.55 + depth * 0.3, 0.90),
                reason=f"CCI={val:.1f} oversold (<{self._oversold})",
            )

        if val >= self._overbought:
            depth = abs(val - self._overbought) / 100.0
            score = -min(0.5 + depth * 0.5, 1.0)
            return ctx.emit(
                score=score,
                confidence=min(0.55 + depth * 0.3, 0.90),
                reason=f"CCI={val:.1f} overbought (>{self._overbought})",
            )

        return None
