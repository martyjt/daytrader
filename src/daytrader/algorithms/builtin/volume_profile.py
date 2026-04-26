"""Volume Profile analysis algorithm.

Analyses volume distribution across price levels (POC, VAH, VAL)
and emits signals when price reaches value area boundaries with
volume confirmation. Salvaged from CryptoTrader.
"""

from __future__ import annotations

import numpy as np

from ...core.context import AlgorithmContext
from ...core.types.signals import Signal
from ..base import Algorithm, AlgorithmManifest, AlgorithmParam


class VolumeProfileAlgorithm(Algorithm):
    """Volume Profile support/resistance strategy."""

    def __init__(
        self,
        *,
        lookback: int = 24,
        price_bins: int = 20,
        volume_spike_mult: float = 2.0,
    ) -> None:
        self._lookback = lookback
        self._price_bins = price_bins
        self._volume_spike_mult = volume_spike_mult

    @property
    def manifest(self) -> AlgorithmManifest:
        return AlgorithmManifest(
            id="volume_profile",
            name="Volume Profile",
            version="1.0.0",
            description=(
                "Analyses volume distribution across price levels "
                "(POC/VAH/VAL). Emits signals at value area boundaries "
                "with volume spike confirmation."
            ),
            asset_classes=["crypto", "equities"],
            timeframes=["1h", "4h", "1d"],
            params=[
                AlgorithmParam("lookback", "int", self._lookback, min=10, max=200, description="Bars for volume profile"),
                AlgorithmParam("price_bins", "int", self._price_bins, min=5, max=50, description="Number of price bins"),
                AlgorithmParam("volume_spike_mult", "float", self._volume_spike_mult, min=1.0, max=5.0, step=0.5, description="Volume spike multiplier"),
            ],
            author="Daytrader built-in",
        )

    def warmup_bars(self) -> int:
        return self._lookback

    def on_bar(self, ctx: AlgorithmContext) -> Signal | None:
        highs = ctx.history_arrays["high"]
        lows = ctx.history_arrays["low"]
        closes = ctx.history_arrays["close"]
        volumes = ctx.history_arrays["volume"]
        if closes is None or len(closes) < self._lookback:
            return None

        # Build volume profile over lookback window
        h = highs[-self._lookback :]
        lo = lows[-self._lookback :]
        v = volumes[-self._lookback :]

        high_price = np.max(h)
        low_price = np.min(lo)
        if high_price <= low_price:
            return None

        bin_size = (high_price - low_price) / self._price_bins
        bin_volumes = np.zeros(self._price_bins)
        bin_mids = np.array([
            low_price + bin_size * (i + 0.5) for i in range(self._price_bins)
        ])

        for j in range(self._lookback):
            candle_range = h[j] - lo[j]
            if candle_range <= 0:
                continue
            for b in range(self._price_bins):
                b_low = low_price + bin_size * b
                b_high = b_low + bin_size
                overlap_low = max(lo[j], b_low)
                overlap_high = min(h[j], b_high)
                if overlap_high > overlap_low:
                    ratio = (overlap_high - overlap_low) / candle_range
                    bin_volumes[b] += v[j] * ratio

        total_vol = bin_volumes.sum()
        if total_vol == 0:
            return None

        # Point of Control (highest volume bin)
        poc_idx = int(np.argmax(bin_volumes))
        poc = bin_mids[poc_idx]

        # Value Area (70% of total volume)
        sorted_indices = np.argsort(bin_volumes)[::-1]
        va_vol = 0.0
        va_bins = []
        for idx in sorted_indices:
            va_bins.append(idx)
            va_vol += bin_volumes[idx]
            if va_vol >= total_vol * 0.70:
                break

        vah = low_price + bin_size * (max(va_bins) + 1)
        val_ = low_price + bin_size * min(va_bins)

        close = float(ctx.bar.close)
        vol = float(ctx.bar.volume)
        avg_vol = float(np.mean(v))
        is_spike = avg_vol > 0 and vol > avg_vol * self._volume_spike_mult

        # Signal logic
        if close <= val_:
            dist = (val_ - close) / val_ if val_ > 0 else 0.0
            confidence = min(0.5 + dist * 2.0, 0.85)
            if is_spike:
                confidence = min(confidence + 0.1, 0.95)
            score = min(0.5 + dist * 3.0, 1.0)
            return ctx.emit(
                score=score,
                confidence=confidence,
                reason=f"Price {close:.2f} at/below VAL ({val_:.2f}), POC={poc:.2f}",
            )

        if close >= vah:
            dist = (close - vah) / vah if vah > 0 else 0.0
            confidence = min(0.5 + dist * 2.0, 0.85)
            if is_spike:
                confidence = min(confidence + 0.1, 0.95)
            score = -min(0.5 + dist * 3.0, 1.0)
            return ctx.emit(
                score=score,
                confidence=confidence,
                reason=f"Price {close:.2f} at/above VAH ({vah:.2f}), POC={poc:.2f}",
            )

        if is_spike:
            score = 0.55 if close > poc else -0.55
            return ctx.emit(
                score=score,
                confidence=0.55,
                reason=f"Volume spike at {close:.2f}, POC={poc:.2f}",
            )

        return None
