"""Example Buy & Hold plugin.

Emits a max-conviction long signal on every bar. The backtest engine
buys on the first signal and holds. Useful as a baseline benchmark when
forking your first real strategy.
"""

from __future__ import annotations

from daytrader.algorithms.base import Algorithm, AlgorithmManifest
from daytrader.core.context import AlgorithmContext
from daytrader.core.types.signals import Signal


class ExampleBuyHold(Algorithm):
    @property
    def manifest(self) -> AlgorithmManifest:
        return AlgorithmManifest(
            id="example_buy_hold",
            name="Example — Buy & Hold",
            version="0.1.0",
            description="Always long. Baseline benchmark.",
            asset_classes=["crypto", "equities", "forex", "commodities"],
            timeframes=["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"],
            author="Daytrader examples",
        )

    def on_bar(self, ctx: AlgorithmContext) -> Signal | None:
        return ctx.emit(score=1.0, confidence=1.0, reason="buy and hold")
