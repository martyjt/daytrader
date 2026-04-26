"""Triple EMA (TEMA) crossover algorithm.

Trend-following strategy using TEMA crossover, which is more
responsive than standard EMA with less lag.
"""

from __future__ import annotations

import numpy as np

from ...core.context import AlgorithmContext
from ...core.types.signals import Signal
from ..base import Algorithm, AlgorithmManifest, AlgorithmParam
from ..indicators import tema as _tema


class TripleEMACrossoverAlgorithm(Algorithm):
    """TEMA fast/slow crossover trend strategy."""

    def __init__(
        self,
        *,
        fast_period: int = 9,
        slow_period: int = 21,
    ) -> None:
        self._fast_period = fast_period
        self._slow_period = slow_period

    @property
    def manifest(self) -> AlgorithmManifest:
        return AlgorithmManifest(
            id="triple_ema",
            name="Triple EMA Crossover",
            version="1.0.0",
            description="Trend-following via fast/slow TEMA crossover with reduced lag.",
            asset_classes=["crypto", "equities"],
            timeframes=["5m", "15m", "1h", "4h", "1d"],
            params=[
                AlgorithmParam("fast_period", "int", self._fast_period, min=3, max=50, description="Fast TEMA period"),
                AlgorithmParam("slow_period", "int", self._slow_period, min=10, max=200, description="Slow TEMA period"),
            ],
            author="Daytrader built-in",
        )

    def warmup_bars(self) -> int:
        # TEMA needs 3x the period for triple smoothing
        return self._slow_period * 3 + 1

    def on_bar(self, ctx: AlgorithmContext) -> Signal | None:
        closes = ctx.history_arrays["close"]
        if closes is None or len(closes) < self.warmup_bars():
            return None

        fast = _tema(closes, self._fast_period)
        slow = _tema(closes, self._slow_period)

        i = len(closes) - 1
        if np.isnan(fast[i]) or np.isnan(slow[i]) or np.isnan(fast[i - 1]) or np.isnan(slow[i - 1]):
            return None

        # Detect crossover
        prev_diff = fast[i - 1] - slow[i - 1]
        curr_diff = fast[i] - slow[i]

        if prev_diff <= 0 and curr_diff > 0:
            # Bullish crossover
            spread = curr_diff / slow[i] if slow[i] > 0 else 0.0
            score = min(0.6 + spread * 10.0, 1.0)
            return ctx.emit(
                score=score,
                confidence=min(0.55 + spread * 5.0, 0.90),
                reason=f"TEMA bullish crossover: fast={fast[i]:.2f} > slow={slow[i]:.2f}",
            )

        if prev_diff >= 0 and curr_diff < 0:
            # Bearish crossover
            spread = abs(curr_diff) / slow[i] if slow[i] > 0 else 0.0
            score = -min(0.6 + spread * 10.0, 1.0)
            return ctx.emit(
                score=score,
                confidence=min(0.55 + spread * 5.0, 0.90),
                reason=f"TEMA bearish crossover: fast={fast[i]:.2f} < slow={slow[i]:.2f}",
            )

        return None
