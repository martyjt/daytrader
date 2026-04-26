"""Alpaca data adapter — US equities historical bars via alpaca-py.

Provides OHLCV data for US stocks using Alpaca's market data API.
Supports paper and live API keys. Equities only (no crypto via this
adapter — use yfinance or ccxt for crypto).

Rate limits: 200 req/min for free tier.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime

import pandas as pd
import polars as pl

from ...core.types.bars import Timeframe
from ...core.types.common import utcnow
from ...core.types.symbols import AssetClass, Symbol
from .base import OHLCV_SCHEMA, AdapterCapabilities, AdapterHealth, DataAdapter

_ALPACA_TIMEFRAMES: dict[str, str] = {
    "1m": "1Min",
    "5m": "5Min",
    "15m": "15Min",
    "30m": "30Min",
    "1h": "1Hour",
    "1d": "1Day",
    "1w": "1Week",
}

_SUPPORTED_TIMEFRAMES = [
    Timeframe.M1,
    Timeframe.M5,
    Timeframe.M15,
    Timeframe.M30,
    Timeframe.H1,
    Timeframe.D1,
    Timeframe.W1,
]


class AlpacaAdapter(DataAdapter):
    """US equities market data via Alpaca."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        paper: bool = True,
    ) -> None:
        self._api_key = api_key
        self._api_secret = api_secret
        self._paper = paper

    @property
    def name(self) -> str:
        return "alpaca"

    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            asset_classes=[AssetClass.EQUITIES],
            timeframes=_SUPPORTED_TIMEFRAMES,
            max_history_days=3650,
            supports_streaming=False,
            rate_limit_per_minute=200,
        )

    async def fetch_ohlcv(
        self,
        symbol: Symbol,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
    ) -> pl.DataFrame:
        alpaca_tf = _ALPACA_TIMEFRAMES.get(timeframe.value)
        if alpaca_tf is None:
            raise ValueError(
                f"Alpaca does not support interval {timeframe.value!r}. "
                f"Supported: {sorted(_ALPACA_TIMEFRAMES)}"
            )

        ticker = self.to_alpaca_ticker(symbol)

        def _fetch():
            from alpaca.data.historical.stock import StockHistoricalDataClient
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

            client = StockHistoricalDataClient(self._api_key, self._api_secret)

            tf_map = {
                "1Min": TimeFrame(1, TimeFrameUnit.Minute),
                "5Min": TimeFrame(5, TimeFrameUnit.Minute),
                "15Min": TimeFrame(15, TimeFrameUnit.Minute),
                "30Min": TimeFrame(30, TimeFrameUnit.Minute),
                "1Hour": TimeFrame(1, TimeFrameUnit.Hour),
                "1Day": TimeFrame(1, TimeFrameUnit.Day),
                "1Week": TimeFrame(1, TimeFrameUnit.Week),
            }

            request = StockBarsRequest(
                symbol_or_symbols=ticker,
                timeframe=tf_map[alpaca_tf],
                start=start,
                end=end,
            )
            bars = client.get_stock_bars(request)
            # alpaca-py types this as `BarSet | dict[str, Any]`; .df only
            # exists on BarSet.
            if isinstance(bars, dict):
                return pd.DataFrame()
            return bars.df

        pdf = await asyncio.to_thread(_fetch)

        if pdf.empty:
            return pl.DataFrame(schema=OHLCV_SCHEMA)

        pdf = pdf.reset_index()
        pdf.columns = [c.lower() for c in pdf.columns]

        # Alpaca returns 'timestamp' column after reset_index
        # Rename if needed
        if "timestamp" not in pdf.columns:
            for col in ("date", "datetime"):
                if col in pdf.columns:
                    pdf = pdf.rename(columns={col: "timestamp"})
                    break

        # Keep only OHLCV columns
        keep = [
            c
            for c in ("timestamp", "open", "high", "low", "close", "volume")
            if c in pdf.columns
        ]
        pdf = pdf[keep]

        return pl.from_pandas(pdf)

    async def health(self) -> AdapterHealth:
        try:
            t0 = time.monotonic()

            def _probe():
                from alpaca.data.historical.stock import StockHistoricalDataClient

                client = StockHistoricalDataClient(self._api_key, self._api_secret)
                # Just instantiate — if keys are valid this won't raise
                return client

            await asyncio.to_thread(_probe)
            elapsed = (time.monotonic() - t0) * 1000
            return AdapterHealth(
                status="ok",
                latency_ms=round(elapsed, 1),
                last_successful_call=utcnow(),
            )
        except Exception as e:
            return AdapterHealth(status="down", error=str(e))

    @staticmethod
    def to_alpaca_ticker(symbol: Symbol) -> str:
        """Convert a ``Symbol`` to Alpaca ticker format.

        Alpaca uses plain ticker symbols for US equities: ``AAPL``, ``MSFT``.
        """
        return symbol.base
