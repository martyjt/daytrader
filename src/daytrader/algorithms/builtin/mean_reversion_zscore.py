"""Z-score mean reversion algorithm.

Statistical mean reversion: buy when z-score drops below -2,
sell when it rises above +2. Foundation for pairs trading.
"""

from __future__ import annotations

from ...core.context import AlgorithmContext
from ...core.types.signals import Signal
from ..base import Algorithm, AlgorithmManifest, AlgorithmParam
from ..indicators import zscore as _zscore


class MeanReversionZScoreAlgorithm(Algorithm):
    """Z-score based statistical mean reversion."""

    def __init__(
        self,
        *,
        period: int = 20,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
    ) -> None:
        self._period = period
        self._entry_z = entry_z
        self._exit_z = exit_z

    @property
    def manifest(self) -> AlgorithmManifest:
        return AlgorithmManifest(
            id="mean_reversion_zscore",
            name="Z-Score Mean Reversion",
            version="1.0.0",
            description="Statistical mean reversion using rolling z-score of price.",
            asset_classes=["crypto", "equities"],
            timeframes=["5m", "15m", "1h", "4h", "1d"],
            params=[
                AlgorithmParam("period", "int", self._period, min=10, max=100, description="Rolling window period"),
                AlgorithmParam("entry_z", "float", self._entry_z, min=1.0, max=4.0, step=0.25, description="Z-score entry threshold"),
                AlgorithmParam("exit_z", "float", self._exit_z, min=0.0, max=1.5, step=0.25, description="Z-score exit threshold"),
            ],
            author="Daytrader built-in",
        )

    def warmup_bars(self) -> int:
        return self._period + 1

    def on_bar(self, ctx: AlgorithmContext) -> Signal | None:
        closes = ctx.history_arrays["close"]
        if closes is None or len(closes) < self.warmup_bars():
            return None

        z_vals = _zscore(closes, self._period)
        z = z_vals[-1]
        if z != z:  # NaN
            return None

        if z <= -self._entry_z:
            depth = abs(z) - self._entry_z
            score = min(0.5 + depth * 0.25, 1.0)
            return ctx.emit(
                score=score,
                confidence=min(0.55 + depth * 0.15, 0.90),
                reason=f"Z-score={z:.2f} below -{self._entry_z} (mean reversion buy)",
            )

        if z >= self._entry_z:
            depth = z - self._entry_z
            score = -min(0.5 + depth * 0.25, 1.0)
            return ctx.emit(
                score=score,
                confidence=min(0.55 + depth * 0.15, 0.90),
                reason=f"Z-score={z:.2f} above +{self._entry_z} (mean reversion sell)",
            )

        return None
