"""Stochastic RSI — K/D crossover in extreme zones.

Applies the Stochastic oscillator to the RSI series rather than to
raw prices.  The resulting %K and %D lines oscillate between 0 and 1
and are more responsive than plain RSI.  Signals fire when K crosses
D inside the oversold or overbought zone, indicating a probable
momentum shift.
"""

from __future__ import annotations

import numpy as np

from ...core.context import AlgorithmContext
from ...core.types.signals import Signal
from ..base import Algorithm, AlgorithmManifest, AlgorithmParam
from ..indicators import rsi as compute_rsi
from ..indicators import sma, stochastic


class StochasticRSIAlgorithm(Algorithm):
    """Stochastic RSI crossover strategy in extreme zones."""

    def __init__(
        self,
        *,
        rsi_period: int = 14,
        stoch_period: int = 14,
        smooth_k: int = 3,
        smooth_d: int = 3,
        oversold: float = 0.2,
        overbought: float = 0.8,
    ) -> None:
        self._rsi_period = rsi_period
        self._stoch_period = stoch_period
        self._smooth_k = smooth_k
        self._smooth_d = smooth_d
        self._oversold = oversold
        self._overbought = overbought

    @property
    def manifest(self) -> AlgorithmManifest:
        return AlgorithmManifest(
            id="stochastic_rsi",
            name="Stochastic RSI",
            version="1.0.0",
            description=(
                "Applies the Stochastic oscillator to the RSI series. "
                "Emits bullish signals on K/D crossover in the oversold "
                "zone and bearish signals on crossunder in the overbought "
                "zone."
            ),
            asset_classes=["crypto", "equities", "forex", "commodities"],
            timeframes=["5m", "15m", "30m", "1h", "4h", "1d"],
            params=[
                AlgorithmParam(
                    "rsi_period", "int", self._rsi_period,
                    min=2, max=50,
                    description="RSI lookback period",
                ),
                AlgorithmParam(
                    "stoch_period", "int", self._stoch_period,
                    min=2, max=50,
                    description="Stochastic lookback applied to RSI",
                ),
                AlgorithmParam(
                    "smooth_k", "int", self._smooth_k,
                    min=1, max=10,
                    description="SMA smoothing for %K line",
                ),
                AlgorithmParam(
                    "smooth_d", "int", self._smooth_d,
                    min=1, max=10,
                    description="SMA smoothing for %D line (signal line)",
                ),
                AlgorithmParam(
                    "oversold", "float", self._oversold,
                    min=0.0, max=0.5, step=0.05,
                    description="Oversold threshold (0-1 scale)",
                ),
                AlgorithmParam(
                    "overbought", "float", self._overbought,
                    min=0.5, max=1.0, step=0.05,
                    description="Overbought threshold (0-1 scale)",
                ),
            ],
            author="Daytrader built-in",
        )

    def warmup_bars(self) -> int:
        return (
            self._rsi_period
            + self._stoch_period
            + self._smooth_k
            + self._smooth_d
        )

    def on_bar(self, ctx: AlgorithmContext) -> Signal | None:
        rsi_period = ctx.param("rsi_period", self._rsi_period)
        stoch_period = ctx.param("stoch_period", self._stoch_period)
        smooth_k = ctx.param("smooth_k", self._smooth_k)
        smooth_d = ctx.param("smooth_d", self._smooth_d)
        oversold = ctx.param("oversold", self._oversold)
        overbought = ctx.param("overbought", self._overbought)

        closes = ctx.history_arrays["close"]
        min_bars = rsi_period + stoch_period + smooth_k + smooth_d
        if len(closes) < min_bars:
            return None

        # Step 1: Compute RSI series
        rsi_series = compute_rsi(closes, rsi_period)

        # Step 2: Apply stochastic() to the RSI values to get raw %K
        raw_k = stochastic(rsi_series, stoch_period)

        # Step 3: Smooth raw %K with SMA to get K line
        k_line = sma(raw_k, smooth_k)

        # Step 4: Smooth K line with SMA to get D line
        d_line = sma(k_line, smooth_d)

        # Need at least two valid K/D values for crossover detection
        k_now = k_line[-1]
        k_prev = k_line[-2]
        d_now = d_line[-1]
        d_prev = d_line[-2]

        if any(np.isnan(v) for v in (k_now, k_prev, d_now, d_prev)):
            return None

        ctx.log(
            "stochastic_rsi",
            k=float(k_now),
            d=float(d_now),
            k_prev=float(k_prev),
            d_prev=float(d_prev),
            oversold=oversold,
            overbought=overbought,
        )

        # Bullish: K crosses above D AND both below oversold threshold
        if k_prev <= d_prev and k_now > d_now and k_now < oversold and d_now < oversold:
            # Score based on how deep into oversold zone
            depth = (oversold - max(k_now, d_now)) / oversold if oversold > 0 else 0
            score = min(1.0, 0.5 + depth * 0.5)
            return ctx.emit(
                score=score,
                confidence=min(1.0, 0.6 + depth * 0.4),
                reason=(
                    f"StochRSI bullish crossover in oversold zone "
                    f"(K={k_now:.3f}, D={d_now:.3f})"
                ),
            )

        # Bearish: K crosses below D AND both above overbought threshold
        if k_prev >= d_prev and k_now < d_now and k_now > overbought and d_now > overbought:
            # Score based on how deep into overbought zone
            depth = (min(k_now, d_now) - overbought) / (1.0 - overbought) if overbought < 1 else 0
            score = max(-1.0, -(0.5 + depth * 0.5))
            return ctx.emit(
                score=score,
                confidence=min(1.0, 0.6 + depth * 0.4),
                reason=(
                    f"StochRSI bearish crossover in overbought zone "
                    f"(K={k_now:.3f}, D={d_now:.3f})"
                ),
            )

        return None
