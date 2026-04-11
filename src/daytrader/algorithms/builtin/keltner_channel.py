"""Keltner Channel algorithm.

ATR-based channel mean reversion. Like Bollinger Bands but uses
ATR instead of standard deviation, giving smoother bands.
"""

from __future__ import annotations

import numpy as np

from ..base import Algorithm, AlgorithmManifest, AlgorithmParam
from ..indicators import ema as _ema, atr as _atr
from ...core.context import AlgorithmContext
from ...core.types.signals import Signal


class KeltnerChannelAlgorithm(Algorithm):
    """Keltner Channel mean reversion at ATR-based bands."""

    def __init__(
        self,
        *,
        ema_period: int = 20,
        atr_period: int = 14,
        atr_mult: float = 2.0,
    ) -> None:
        self._ema_period = ema_period
        self._atr_period = atr_period
        self._atr_mult = atr_mult

    @property
    def manifest(self) -> AlgorithmManifest:
        return AlgorithmManifest(
            id="keltner_channel",
            name="Keltner Channel",
            version="1.0.0",
            description="Mean-reversion at ATR-based Keltner Channel bands.",
            asset_classes=["crypto", "equities"],
            timeframes=["5m", "15m", "1h", "4h", "1d"],
            params=[
                AlgorithmParam("ema_period", "int", self._ema_period, min=5, max=50, description="EMA center period"),
                AlgorithmParam("atr_period", "int", self._atr_period, min=5, max=50, description="ATR period"),
                AlgorithmParam("atr_mult", "float", self._atr_mult, min=0.5, max=5.0, step=0.5, description="ATR band multiplier"),
            ],
            author="Daytrader built-in",
        )

    def warmup_bars(self) -> int:
        return max(self._ema_period, self._atr_period) + 1

    def on_bar(self, ctx: AlgorithmContext) -> Signal | None:
        closes = ctx.history_arrays.get("close")
        highs = ctx.history_arrays.get("high")
        lows = ctx.history_arrays.get("low")
        if closes is None or len(closes) < self.warmup_bars():
            return None

        mid = _ema(closes, self._ema_period)
        atr_vals = _atr(highs, lows, closes, self._atr_period)

        i = len(closes) - 1
        if np.isnan(mid[i]) or np.isnan(atr_vals[i]):
            return None

        upper = mid[i] + self._atr_mult * atr_vals[i]
        lower = mid[i] - self._atr_mult * atr_vals[i]
        close = closes[i]

        if close <= lower:
            dist = (lower - close) / atr_vals[i] if atr_vals[i] > 0 else 0.0
            score = min(0.5 + dist * 0.3, 1.0)
            return ctx.emit(
                score=score,
                confidence=min(0.55 + dist * 0.15, 0.90),
                reason=f"Price below Keltner lower band ({lower:.2f})",
            )

        if close >= upper:
            dist = (close - upper) / atr_vals[i] if atr_vals[i] > 0 else 0.0
            score = -min(0.5 + dist * 0.3, 1.0)
            return ctx.emit(
                score=score,
                confidence=min(0.55 + dist * 0.15, 0.90),
                reason=f"Price above Keltner upper band ({upper:.2f})",
            )

        return None
