"""yfinance data adapter — free EOD + limited intraday data.

Covers crypto (BTC-USD), equities (AAPL), forex (EURUSD=X), and
commodities (GC=F). Rate limits are lenient but unofficial; don't
hammer it.

History limits by interval:
    1m  → 7 days        5m → 60 days       15m → 60 days
    30m → 60 days       1h → 730 days       1d  → ~20 years
    1wk → ~20 years

4h is NOT natively supported by yfinance. Request 1h and aggregate
in Phase 2, or use a different adapter.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime

import polars as pl

from ...core.types.bars import Timeframe
from ...core.types.common import utcnow
from ...core.types.symbols import AssetClass, Symbol
from .base import OHLCV_SCHEMA, AdapterCapabilities, AdapterHealth, DataAdapter

_YF_INTERVALS: dict[str, str] = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "1d": "1d",
    "1w": "1wk",
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


class YFinanceAdapter(DataAdapter):
    """Free market data via ``yfinance``."""

    @property
    def name(self) -> str:
        return "yfinance"

    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            asset_classes=[
                AssetClass.CRYPTO,
                AssetClass.EQUITIES,
                AssetClass.FOREX,
                AssetClass.COMMODITIES,
            ],
            timeframes=_SUPPORTED_TIMEFRAMES,
            max_history_days=7300,
            supports_streaming=False,
            rate_limit_per_minute=100,
        )

    async def fetch_ohlcv(
        self,
        symbol: Symbol,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
    ) -> pl.DataFrame:
        yf_interval = _YF_INTERVALS.get(timeframe.value)
        if yf_interval is None:
            raise ValueError(
                f"yfinance does not support interval {timeframe.value!r}. "
                f"Supported: {sorted(_YF_INTERVALS)}"
            )

        ticker_str = self.to_yfinance_ticker(symbol)

        def _fetch():
            import yfinance as yf

            t = yf.Ticker(ticker_str)
            return t.history(
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                interval=yf_interval,
                auto_adjust=True,
            )

        pdf = await asyncio.to_thread(_fetch)

        if pdf.empty:
            return pl.DataFrame(schema=OHLCV_SCHEMA)

        pdf = pdf.reset_index()
        pdf.columns = [c.lower() for c in pdf.columns]

        # yfinance uses 'date' for daily, 'datetime' for intraday
        for col in ("date", "datetime"):
            if col in pdf.columns:
                pdf = pdf.rename(columns={col: "timestamp"})
                break

        # Keep only OHLCV columns
        keep = [c for c in ("timestamp", "open", "high", "low", "close", "volume") if c in pdf.columns]
        pdf = pdf[keep]

        return pl.from_pandas(pdf)

    async def health(self) -> AdapterHealth:
        try:
            t0 = time.monotonic()

            def _probe():
                import yfinance as yf

                return yf.Ticker("AAPL").fast_info

            await asyncio.to_thread(_probe)
            elapsed = (time.monotonic() - t0) * 1000
            return AdapterHealth(
                status="ok",
                latency_ms=round(elapsed, 1),
                last_successful_call=utcnow(),
            )
        except Exception as e:
            return AdapterHealth(status="down", error=str(e))

    # ---- ticker conversion -----------------------------------------------

    @staticmethod
    def to_yfinance_ticker(symbol: Symbol) -> str:
        """Convert a ``Symbol`` to the yfinance ticker format.

        Examples::

            crypto   BTC/USD   → BTC-USD
            equities AAPL/USD  → AAPL
            forex    EUR/USD   → EURUSD=X
            other              → symbol.base (user must provide yf-compatible)
        """
        if symbol.asset_class == AssetClass.CRYPTO:
            return f"{symbol.base}-{symbol.quote}"
        if symbol.asset_class == AssetClass.EQUITIES:
            return symbol.base
        if symbol.asset_class == AssetClass.FOREX:
            return f"{symbol.base}{symbol.quote}=X"
        return symbol.base
