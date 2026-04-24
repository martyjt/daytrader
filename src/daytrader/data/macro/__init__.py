"""Macro adapters — univariate economic time-series.

Parallel to ``data.adapters`` (OHLCV) but distinct because macro series
don't fit the OHLCV schema: they're single-value readings (rates, CPI,
VIX, yield-curve slopes) at daily/weekly/monthly cadence.

Consumed by the Exploration Agent (Track 2b) to propose candidate
features, not by the backtest engine directly.
"""

from __future__ import annotations
