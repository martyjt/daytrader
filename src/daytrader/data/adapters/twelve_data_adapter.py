"""Twelve Data adapter — free equities / FX / crypto.

Free tier: 800 requests/day, 8 req/min. Register at
``https://twelvedata.com`` and store ``TWELVE_DATA_API_KEY``.

Broader asset coverage than Alpha Vantage on the free tier (includes
crypto), slightly more generous daily cap. Useful as a third source for
cross-validation or as a fallback when yfinance or Alpha Vantage are
rate-limited.
"""

from __future__ import annotations

import time
from datetime import datetime

import httpx
import polars as pl

from ...core.types.bars import Timeframe
from ...core.types.common import utcnow
from ...core.types.symbols import AssetClass, Symbol
from .base import OHLCV_SCHEMA, AdapterCapabilities, AdapterHealth, DataAdapter

_BASE_URL = "https://api.twelvedata.com"

_INTERVALS: dict[str, str] = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "4h": "4h",
    "1d": "1day",
    "1w": "1week",
}

_SUPPORTED_TIMEFRAMES = [
    Timeframe.M1, Timeframe.M5, Timeframe.M15, Timeframe.M30,
    Timeframe.H1, Timeframe.H4, Timeframe.D1, Timeframe.W1,
]


class TwelveDataAdapter(DataAdapter):
    """Twelve Data multi-asset OHLCV adapter."""

    def __init__(self, api_key: str, timeout: float = 15.0) -> None:
        if not api_key:
            raise ValueError("TwelveDataAdapter requires an API key")
        self._api_key = api_key
        self._timeout = timeout

    @property
    def name(self) -> str:
        return "twelve_data"

    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            asset_classes=[AssetClass.EQUITIES, AssetClass.FOREX, AssetClass.CRYPTO],
            timeframes=_SUPPORTED_TIMEFRAMES,
            max_history_days=7300,
            supports_streaming=False,
            rate_limit_per_minute=8,  # free tier
        )

    async def fetch_ohlcv(
        self,
        symbol: Symbol,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
    ) -> pl.DataFrame:
        interval = _INTERVALS.get(timeframe.value)
        if interval is None:
            raise ValueError(
                f"twelve_data does not support interval {timeframe.value!r}. "
                f"Supported: {sorted(_INTERVALS)}"
            )

        td_symbol = self._to_twelve_symbol(symbol)
        params: dict[str, str | int] = {
            "symbol": td_symbol,
            "interval": interval,
            "start_date": start.strftime("%Y-%m-%d %H:%M:%S"),
            "end_date": end.strftime("%Y-%m-%d %H:%M:%S"),
            "outputsize": 5000,
            "apikey": self._api_key,
            "format": "JSON",
            "order": "ASC",
        }
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(f"{_BASE_URL}/time_series", params=params)
            resp.raise_for_status()
            payload = resp.json()

        if payload.get("status") == "error":
            msg = payload.get("message") or "unknown twelve_data error"
            raise RuntimeError(f"twelve_data: {msg}")

        values = payload.get("values", [])
        if not values:
            return pl.DataFrame(schema=OHLCV_SCHEMA)

        rows = []
        for bar in values:
            try:
                ts_str = bar["datetime"]
                ts = datetime.strptime(
                    ts_str, "%Y-%m-%d %H:%M:%S" if " " in ts_str else "%Y-%m-%d",
                )
                rows.append({
                    "timestamp": ts,
                    "open": float(bar["open"]),
                    "high": float(bar["high"]),
                    "low": float(bar["low"]),
                    "close": float(bar["close"]),
                    "volume": float(bar.get("volume", 0.0) or 0.0),
                })
            except (KeyError, ValueError):
                continue
        if not rows:
            return pl.DataFrame(schema=OHLCV_SCHEMA)
        return pl.DataFrame(rows, schema=OHLCV_SCHEMA).sort("timestamp")

    async def health(self) -> AdapterHealth:
        try:
            t0 = time.monotonic()
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(
                    f"{_BASE_URL}/api_usage", params={"apikey": self._api_key}
                )
                resp.raise_for_status()
            elapsed = (time.monotonic() - t0) * 1000
            return AdapterHealth(
                status="ok",
                latency_ms=round(elapsed, 1),
                last_successful_call=utcnow(),
            )
        except Exception as exc:
            return AdapterHealth(status="down", error=str(exc))

    # ---- symbol conversion ----------------------------------------------

    @staticmethod
    def _to_twelve_symbol(symbol: Symbol) -> str:
        """Convert a ``Symbol`` to the Twelve Data string convention.

        Examples:
            equities  AAPL/USD → ``AAPL``
            forex     EUR/USD  → ``EUR/USD``
            crypto    BTC/USD  → ``BTC/USD``
        """
        if symbol.asset_class == AssetClass.EQUITIES:
            return symbol.base.upper()
        return f"{symbol.base.upper()}/{symbol.quote.upper()}"
