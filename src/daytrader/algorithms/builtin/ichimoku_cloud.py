"""Ichimoku Cloud breakout algorithm.

Emits signals when price breaks above/below the displaced cloud
with tenkan/kijun confirmation. Salvaged from CryptoTrader and
adapted to the Daytrader Algorithm ABC.
"""

from __future__ import annotations

import numpy as np

from ..base import Algorithm, AlgorithmManifest, AlgorithmParam
from ..indicators import ichimoku_lines
from ...core.context import AlgorithmContext
from ...core.types.signals import Signal


class IchimokuCloudAlgorithm(Algorithm):
    """Ichimoku Cloud trend-following strategy."""

    def __init__(
        self,
        *,
        tenkan_period: int = 9,
        kijun_period: int = 26,
        span_b_period: int = 52,
        displacement: int = 26,
    ) -> None:
        self._tenkan_period = tenkan_period
        self._kijun_period = kijun_period
        self._span_b_period = span_b_period
        self._displacement = displacement
        self._prev_state: float = 0.0  # +1 bullish, -1 bearish, 0 neutral

    @property
    def manifest(self) -> AlgorithmManifest:
        return AlgorithmManifest(
            id="ichimoku_cloud",
            name="Ichimoku Cloud",
            version="1.0.0",
            description=(
                "Trend strategy: emits on cloud breakout with "
                "tenkan/kijun confirmation. Displaced cloud acts as "
                "support/resistance."
            ),
            asset_classes=["crypto", "equities"],
            timeframes=["1h", "4h", "1d"],
            params=[
                AlgorithmParam("tenkan_period", "int", self._tenkan_period, min=5, max=30, description="Conversion line period"),
                AlgorithmParam("kijun_period", "int", self._kijun_period, min=10, max=60, description="Base line period"),
                AlgorithmParam("span_b_period", "int", self._span_b_period, min=20, max=120, description="Senkou Span B period"),
                AlgorithmParam("displacement", "int", self._displacement, min=10, max=52, description="Cloud displacement"),
            ],
            author="Daytrader built-in",
        )

    def warmup_bars(self) -> int:
        return self._span_b_period + self._displacement

    def on_bar(self, ctx: AlgorithmContext) -> Signal | None:
        highs = ctx.history_arrays.get("high")
        lows = ctx.history_arrays.get("low")
        closes = ctx.history_arrays.get("close")
        if closes is None or len(closes) < self.warmup_bars():
            return None

        tenkan, kijun, span_a, span_b = ichimoku_lines(
            highs, lows,
            self._tenkan_period,
            self._kijun_period,
            self._span_b_period,
        )

        i = len(closes) - 1
        disp = self._displacement

        # Displaced cloud: look back by displacement
        if i - disp < 0 or np.isnan(span_a[i - disp]) or np.isnan(span_b[i - disp]):
            return None

        cloud_a = span_a[i - disp]
        cloud_b = span_b[i - disp]
        cloud_top = max(cloud_a, cloud_b)
        cloud_bottom = min(cloud_a, cloud_b)

        close = float(ctx.bar.close)
        t = tenkan[i]
        k = kijun[i]
        if np.isnan(t) or np.isnan(k):
            return None

        # Determine state
        bullish_confirm = t > k
        bearish_confirm = t < k
        above_cloud = close > cloud_top
        below_cloud = close < cloud_bottom

        state = 0.0
        if above_cloud and bullish_confirm:
            state = 1.0
        elif below_cloud and bearish_confirm:
            state = -1.0

        # Only emit on state change
        if state == 0.0 or state == self._prev_state:
            self._prev_state = state
            return None
        self._prev_state = state

        # Distance beyond cloud as confidence
        if state > 0:
            dist = (close - cloud_top) / close if close > 0 else 0.0
        else:
            dist = (cloud_bottom - close) / close if close > 0 else 0.0

        confidence = min(0.60 + dist * 15.0, 0.90)
        score = state * min(0.6 + dist * 5.0, 1.0)

        return ctx.emit(
            score=score,
            confidence=confidence,
            reason=(
                f"Ichimoku cloud break: close={close:.2f} "
                f"cloud=({cloud_bottom:.2f}-{cloud_top:.2f}), "
                f"tenkan={t:.2f}, kijun={k:.2f}"
            ),
        )
