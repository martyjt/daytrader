"""Donchian Breakout — N-period high/low channel breakout with volatility gate.

Computes a Donchian channel (N-period highest high and lowest low) and
monitors for breakouts where the close exceeds the channel boundary.
A volatility gate based on ATR expansion filters out breakouts that
occur in low-volatility environments, which are more likely to be
false signals.

Works best on trending instruments where price expansions coincide
with rising volatility.
"""

from __future__ import annotations

import numpy as np

from ...core.context import AlgorithmContext
from ...core.types.signals import Signal
from ..base import Algorithm, AlgorithmManifest, AlgorithmParam
from ..indicators import atr


class DonchianBreakoutAlgorithm(Algorithm):
    @property
    def manifest(self) -> AlgorithmManifest:
        return AlgorithmManifest(
            id="donchian_breakout",
            name="Donchian Breakout",
            version="1.0.0",
            description=(
                "N-period high/low channel breakout filtered by an ATR "
                "volatility gate. Only signals when current ATR exceeds "
                "a multiple of its rolling mean, ensuring breakouts occur "
                "during genuine volatility expansion."
            ),
            asset_classes=["crypto", "equities"],
            timeframes=["1h", "4h", "1d"],
            params=[
                AlgorithmParam(
                    name="lookback",
                    type="int",
                    default=20,
                    min=5,
                    max=100,
                    description="N-period lookback for channel high/low",
                ),
                AlgorithmParam(
                    name="atr_period",
                    type="int",
                    default=14,
                    min=5,
                    max=50,
                    description="ATR period for volatility gate",
                ),
                AlgorithmParam(
                    name="atr_vol_multiplier",
                    type="float",
                    default=1.2,
                    min=0.5,
                    max=3.0,
                    step=0.1,
                    description="ATR multiplier for volatility gate threshold",
                ),
            ],
            author="Daytrader built-in",
        )

    def warmup_bars(self) -> int:
        defaults = self.manifest.param_defaults()
        return max(int(defaults["lookback"]), int(defaults["atr_period"])) + 1

    def on_bar(self, ctx: AlgorithmContext) -> Signal | None:
        closes = ctx.history_arrays["close"]
        highs = ctx.history_arrays["high"]
        lows = ctx.history_arrays["low"]

        lookback = int(ctx.param("lookback", 20))
        atr_period = int(ctx.param("atr_period", 14))
        atr_vol_multiplier = float(ctx.param("atr_vol_multiplier", 1.2))

        min_bars = max(lookback, atr_period) + 1
        if len(closes) < min_bars:
            return None

        # N-period highest high and lowest low (excluding current bar)
        channel_high = float(np.max(highs[-lookback - 1 : -1]))
        channel_low = float(np.min(lows[-lookback - 1 : -1]))

        # Compute ATR series
        atr_series = atr(highs, lows, closes, atr_period)
        current_atr = float(atr_series[-1])
        if np.isnan(current_atr):
            return None

        # Rolling mean ATR over 2 * atr_period for baseline
        mean_window = 2 * atr_period
        valid_atr = atr_series[~np.isnan(atr_series)]
        if len(valid_atr) < mean_window:
            mean_atr = float(np.mean(valid_atr)) if len(valid_atr) > 0 else current_atr
        else:
            mean_atr = float(np.mean(valid_atr[-mean_window:]))

        # Volatility gate
        gate_open = current_atr > atr_vol_multiplier * mean_atr
        current_close = float(closes[-1])

        ctx.log(
            "donchian_breakout",
            channel_high=channel_high,
            channel_low=channel_low,
            current_atr=current_atr,
            mean_atr=mean_atr,
            gate_open=gate_open,
            atr_vol_multiplier=atr_vol_multiplier,
        )

        if not gate_open:
            return None

        # Bullish breakout: close exceeds N-period high
        if current_close > channel_high:
            return ctx.emit(
                score=0.8,
                confidence=min(1.0, current_atr / mean_atr / 2.0),
                reason=(
                    f"Donchian bullish breakout: close={current_close:.4f} > "
                    f"channel high={channel_high:.4f}, "
                    f"ATR={current_atr:.4f} (gate open, "
                    f"{current_atr / mean_atr:.2f}x mean)"
                ),
            )

        # Bearish breakout: close falls below N-period low
        if current_close < channel_low:
            return ctx.emit(
                score=-0.8,
                confidence=min(1.0, current_atr / mean_atr / 2.0),
                reason=(
                    f"Donchian bearish breakout: close={current_close:.4f} < "
                    f"channel low={channel_low:.4f}, "
                    f"ATR={current_atr:.4f} (gate open, "
                    f"{current_atr / mean_atr:.2f}x mean)"
                ),
            )

        return None
