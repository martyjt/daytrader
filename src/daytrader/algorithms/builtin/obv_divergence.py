"""On-Balance Volume divergence algorithm.

Detects divergence between price and OBV:
- Bearish: price makes new high but OBV does not
- Bullish: price makes new low but OBV does not
"""

from __future__ import annotations

import numpy as np

from ...core.context import AlgorithmContext
from ...core.types.signals import Signal
from ..base import Algorithm, AlgorithmManifest, AlgorithmParam
from ..indicators import obv as _obv


class OBVDivergenceAlgorithm(Algorithm):
    """On-Balance Volume divergence detection."""

    def __init__(
        self,
        *,
        lookback: int = 20,
        divergence_bars: int = 5,
    ) -> None:
        self._lookback = lookback
        self._divergence_bars = divergence_bars

    @property
    def manifest(self) -> AlgorithmManifest:
        return AlgorithmManifest(
            id="obv_divergence",
            name="OBV Divergence",
            version="1.0.0",
            description="Detects price/OBV divergence as reversal signals.",
            asset_classes=["crypto", "equities"],
            timeframes=["1h", "4h", "1d"],
            params=[
                AlgorithmParam("lookback", "int", self._lookback, min=10, max=100, description="Window for peak/trough detection"),
                AlgorithmParam("divergence_bars", "int", self._divergence_bars, min=2, max=20, description="Min bars between peaks"),
            ],
            author="Daytrader built-in",
        )

    def warmup_bars(self) -> int:
        return self._lookback + 1

    def on_bar(self, ctx: AlgorithmContext) -> Signal | None:
        closes = ctx.history_arrays["close"]
        volumes = ctx.history_arrays["volume"]
        if closes is None or len(closes) < self.warmup_bars():
            return None

        obv_vals = _obv(closes, volumes)
        window = self._lookback

        price_win = closes[-window:]
        obv_win = obv_vals[-window:]

        if np.isnan(obv_win).any():
            return None

        # Check for bearish divergence: price new high, OBV lower high
        recent = self._divergence_bars
        price_recent_high = np.max(price_win[-recent:])
        price_prev_high = np.max(price_win[:-recent])
        obv_recent_high = np.max(obv_win[-recent:])
        obv_prev_high = np.max(obv_win[:-recent])

        if price_recent_high > price_prev_high and obv_recent_high < obv_prev_high:
            return ctx.emit(
                score=-0.65,
                confidence=0.60,
                reason=f"Bearish OBV divergence: price high={price_recent_high:.2f} vs OBV declining",
            )

        # Check for bullish divergence: price new low, OBV higher low
        price_recent_low = np.min(price_win[-recent:])
        price_prev_low = np.min(price_win[:-recent])
        obv_recent_low = np.min(obv_win[-recent:])
        obv_prev_low = np.min(obv_win[:-recent])

        if price_recent_low < price_prev_low and obv_recent_low > obv_prev_low:
            return ctx.emit(
                score=0.65,
                confidence=0.60,
                reason=f"Bullish OBV divergence: price low={price_recent_low:.2f} vs OBV rising",
            )

        return None
