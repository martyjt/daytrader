"""Buy & Hold — the simplest possible strategy.

Emits ``score=1.0`` (max long) on every bar. The backtest engine buys
on the first signal and holds until the run ends. Useful as a baseline
benchmark: any real strategy should outperform buy-and-hold on a
risk-adjusted basis.
"""

from __future__ import annotations

from ...core.context import AlgorithmContext
from ...core.types.signals import Signal
from ..base import Algorithm, AlgorithmManifest


class BuyHoldAlgorithm(Algorithm):
    @property
    def manifest(self) -> AlgorithmManifest:
        return AlgorithmManifest(
            id="buy_hold",
            name="Buy & Hold",
            version="1.0.0",
            description=(
                "Always long. Emits a max-conviction buy signal on every "
                "bar. Serves as the baseline benchmark."
            ),
            asset_classes=["crypto", "equities", "forex", "commodities"],
            timeframes=["1m", "5m", "15m", "30m", "1h", "1d", "1w"],
            author="Daytrader built-in",
        )

    def on_bar(self, ctx: AlgorithmContext) -> Signal | None:
        return ctx.emit(
            score=1.0,
            confidence=1.0,
            reason="buy and hold — always long",
        )
