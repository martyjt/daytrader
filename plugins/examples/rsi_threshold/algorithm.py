"""Example RSI threshold plugin.

Computes the Relative Strength Index from the price history exposed by
``ctx.history``. Emits a bullish signal when RSI drops below the
oversold threshold and a bearish signal when it climbs above the
overbought threshold. The score scales with distance past the threshold
so a barely-oversold reading produces a modest signal and an extreme
reading produces a strong one.

This is a self-contained example: the RSI helper is implemented inline
so the plugin doesn't need anything beyond NumPy. Real plugins are
welcome to import from ``daytrader.algorithms.indicators`` if they
prefer.
"""

from __future__ import annotations

import numpy as np

from daytrader.algorithms.base import (
    Algorithm,
    AlgorithmManifest,
    AlgorithmParam,
)
from daytrader.core.context import AlgorithmContext
from daytrader.core.types.signals import Signal


def _rsi(closes: np.ndarray, period: int) -> float:
    """Standard Wilder RSI over the last ``period`` returns."""
    if len(closes) < period + 1:
        return float("nan")
    diffs = np.diff(closes[-(period + 1):])
    gains = np.where(diffs > 0, diffs, 0.0)
    losses = np.where(diffs < 0, -diffs, 0.0)
    avg_gain = float(gains.mean())
    avg_loss = float(losses.mean())
    if avg_loss == 0.0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


class ExampleRsiThreshold(Algorithm):
    @property
    def manifest(self) -> AlgorithmManifest:
        return AlgorithmManifest(
            id="example_rsi_threshold",
            name="Example — RSI Threshold",
            version="0.1.0",
            description=(
                "Mean-reversion using RSI extremes. Long when oversold, "
                "short when overbought."
            ),
            asset_classes=["crypto", "equities"],
            timeframes=["1h", "4h", "1d"],
            params=[
                AlgorithmParam(name="period", type="int", default=14, min=2, max=100),
                AlgorithmParam(
                    name="oversold", type="float", default=30.0, min=5.0, max=50.0
                ),
                AlgorithmParam(
                    name="overbought", type="float", default=70.0, min=50.0, max=95.0
                ),
            ],
            author="Daytrader examples",
            suitable_regimes=["sideways"],
        )

    def warmup_bars(self) -> int:
        return int(self.manifest.param_defaults()["period"]) + 1

    def on_bar(self, ctx: AlgorithmContext) -> Signal | None:
        period = int(ctx.param("period", 14))
        oversold = float(ctx.param("oversold", 30.0))
        overbought = float(ctx.param("overbought", 70.0))

        closes = ctx.history(period + 1, "close")
        rsi = _rsi(closes, period)
        if not np.isfinite(rsi):
            return None

        ctx.log("rsi", value=rsi, period=period)

        if rsi < oversold:
            distance = oversold - rsi
            score = max(0.51, min(1.0, distance / oversold))
            return ctx.emit(
                score=score,
                confidence=score,
                reason=f"oversold rsi={rsi:.1f} < {oversold:.0f}",
            )
        if rsi > overbought:
            distance = rsi - overbought
            span = max(1.0, 100.0 - overbought)
            score = max(0.51, min(1.0, distance / span))
            return ctx.emit(
                score=-score,
                confidence=score,
                reason=f"overbought rsi={rsi:.1f} > {overbought:.0f}",
            )
        return None
