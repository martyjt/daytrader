"""RSI divergence algorithm.

Detects divergence between price and RSI:
- Bearish: price makes new high but RSI lower high
- Bullish: price makes new low but RSI higher low
"""

from __future__ import annotations

import numpy as np

from ..base import Algorithm, AlgorithmManifest, AlgorithmParam
from ..indicators import rsi as _rsi
from ...core.context import AlgorithmContext
from ...core.types.signals import Signal


class RSIDivergenceAlgorithm(Algorithm):
    """RSI divergence detection for reversal signals."""

    def __init__(
        self,
        *,
        rsi_period: int = 14,
        lookback: int = 20,
        divergence_bars: int = 5,
    ) -> None:
        self._rsi_period = rsi_period
        self._lookback = lookback
        self._divergence_bars = divergence_bars

    @property
    def manifest(self) -> AlgorithmManifest:
        return AlgorithmManifest(
            id="rsi_divergence",
            name="RSI Divergence",
            version="1.0.0",
            description="Detects price/RSI divergence as reversal signals.",
            asset_classes=["crypto", "equities"],
            timeframes=["1h", "4h", "1d"],
            params=[
                AlgorithmParam("rsi_period", "int", self._rsi_period, min=5, max=30, description="RSI period"),
                AlgorithmParam("lookback", "int", self._lookback, min=10, max=100, description="Window for peak detection"),
                AlgorithmParam("divergence_bars", "int", self._divergence_bars, min=2, max=20, description="Min bars between peaks"),
            ],
            author="Daytrader built-in",
        )

    def warmup_bars(self) -> int:
        return self._rsi_period + self._lookback + 1

    def on_bar(self, ctx: AlgorithmContext) -> Signal | None:
        closes = ctx.history_arrays.get("close")
        if closes is None or len(closes) < self.warmup_bars():
            return None

        rsi_vals = _rsi(closes, self._rsi_period)
        window = self._lookback

        price_win = closes[-window:]
        rsi_win = rsi_vals[-window:]

        if np.isnan(rsi_win).any():
            return None

        recent = self._divergence_bars

        # Bearish divergence: price new high, RSI lower high
        price_recent_high = np.max(price_win[-recent:])
        price_prev_high = np.max(price_win[:-recent])
        rsi_recent_high = np.max(rsi_win[-recent:])
        rsi_prev_high = np.max(rsi_win[:-recent])

        if price_recent_high > price_prev_high and rsi_recent_high < rsi_prev_high:
            return ctx.emit(
                score=-0.65,
                confidence=0.60,
                reason=f"Bearish RSI divergence: RSI={rsi_win[-1]:.1f}, price high but RSI declining",
            )

        # Bullish divergence: price new low, RSI higher low
        price_recent_low = np.min(price_win[-recent:])
        price_prev_low = np.min(price_win[:-recent])
        rsi_recent_low = np.min(rsi_win[-recent:])
        rsi_prev_low = np.min(rsi_win[:-recent])

        if price_recent_low < price_prev_low and rsi_recent_low > rsi_prev_low:
            return ctx.emit(
                score=0.65,
                confidence=0.60,
                reason=f"Bullish RSI divergence: RSI={rsi_win[-1]:.1f}, price low but RSI rising",
            )

        return None
