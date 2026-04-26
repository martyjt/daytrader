"""Bollinger Bands — mean reversion at band touches.

Computes an SMA of closes with upper and lower bands defined by a
configurable number of standard deviations. When price touches the
lower band a bullish (mean-reversion-up) signal is emitted; when
price touches the upper band a bearish signal is emitted. Signal
score scales with distance beyond the band.
"""

from __future__ import annotations

import numpy as np

from ...core.context import AlgorithmContext
from ...core.types.signals import Signal
from ...core.types.visualize import (
    PlotTrace,
    VisualizeContext,
    nan_array_to_jsonable,
)
from ..base import Algorithm, AlgorithmManifest, AlgorithmParam
from ..indicators import rolling_std, sma


class BollingerBandsAlgorithm(Algorithm):
    """Mean-reversion strategy using Bollinger Bands."""

    def __init__(
        self,
        *,
        period: int = 20,
        std_dev: float = 2.0,
    ) -> None:
        self._period = period
        self._std_dev = std_dev

    @property
    def manifest(self) -> AlgorithmManifest:
        return AlgorithmManifest(
            id="bollinger_bands",
            name="Bollinger Bands",
            version="1.0.0",
            description=(
                "Mean-reversion strategy. Emits bullish signals when "
                "price touches the lower Bollinger Band and bearish "
                "signals at the upper band. Score scales with distance "
                "beyond the band."
            ),
            asset_classes=["crypto", "equities", "forex", "commodities"],
            timeframes=["5m", "15m", "30m", "1h", "4h", "1d"],
            params=[
                AlgorithmParam(
                    "period", "int", self._period,
                    min=5, max=100,
                    description="SMA / std-dev lookback period",
                ),
                AlgorithmParam(
                    "std_dev", "float", self._std_dev,
                    min=0.5, max=4.0, step=0.5,
                    description="Number of standard deviations for bands",
                ),
            ],
            author="Daytrader built-in",
            suitable_regimes=["sideways"],
        )

    def warmup_bars(self) -> int:
        return self._period

    def on_bar(self, ctx: AlgorithmContext) -> Signal | None:
        period = ctx.param("period", self._period)
        std_dev = ctx.param("std_dev", self._std_dev)

        closes = ctx.history_arrays["close"]
        if closes is None or len(closes) < period:
            return None

        mid = sma(closes, period)
        std = rolling_std(closes, period)

        # Current values
        mid_val = mid[-1]
        std_val = std[-1]
        if np.isnan(mid_val) or np.isnan(std_val) or std_val == 0:
            return None

        upper = mid_val + std_dev * std_val
        lower = mid_val - std_dev * std_val
        price = float(closes[-1])

        ctx.log(
            "bollinger_bands",
            price=price,
            sma=float(mid_val),
            upper_band=float(upper),
            lower_band=float(lower),
            std=float(std_val),
        )

        band_width = upper - lower
        if band_width <= 0:
            return None

        if price <= lower:
            # Bullish — how far below the lower band (scaled by band width)
            distance = (lower - price) / band_width
            score = min(1.0, 0.5 + distance)
            return ctx.emit(
                score=score,
                confidence=min(1.0, distance + 0.5),
                reason=(
                    f"Price {price:.4f} at/below lower Bollinger Band "
                    f"{lower:.4f} (SMA {mid_val:.4f}, {std_dev}σ)"
                ),
            )

        if price >= upper:
            # Bearish — how far above the upper band (scaled by band width)
            distance = (price - upper) / band_width
            score = max(-1.0, -(0.5 + distance))
            return ctx.emit(
                score=score,
                confidence=min(1.0, distance + 0.5),
                reason=(
                    f"Price {price:.4f} at/above upper Bollinger Band "
                    f"{upper:.4f} (SMA {mid_val:.4f}, {std_dev}σ)"
                ),
            )

        # Price inside bands — no signal
        return None

    def visualize(self, vctx: VisualizeContext) -> list[PlotTrace]:
        period = int(vctx.params.get("period", self._period))
        std_dev = float(vctx.params.get("std_dev", self._std_dev))
        mid = sma(vctx.closes, period)
        std = rolling_std(vctx.closes, period)
        upper = mid + std_dev * std
        lower = mid - std_dev * std
        return [
            PlotTrace(
                name=f"BB Upper ({period}, {std_dev}σ)",
                kind="line",
                data=nan_array_to_jsonable(upper),
                panel="price",
                color="#fa5252",
                dash="dashed",
            ),
            PlotTrace(
                name=f"BB Middle (SMA {period})",
                kind="line",
                data=nan_array_to_jsonable(mid),
                panel="price",
                color="#868e96",
            ),
            PlotTrace(
                name=f"BB Lower ({period}, {std_dev}σ)",
                kind="line",
                data=nan_array_to_jsonable(lower),
                panel="price",
                color="#40c057",
                dash="dashed",
            ),
        ]
