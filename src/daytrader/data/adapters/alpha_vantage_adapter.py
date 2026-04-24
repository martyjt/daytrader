"""Alpha Vantage adapter — free equities / FX / commodities.

Free tier: 5 requests/min, 500/day. Register at
``https://www.alphavantage.co/support/#api-key`` and store as
``ALPHA_VANTAGE_API_KEY``. Good as a second equity source for
cross-validation against yfinance.

Covers daily + intraday (1m, 5m, 15m, 30m, 60m) for US equities and
major FX pairs.
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

_BASE_URL = "https://www.alphavantage.co/query"

# Alpha Vantage intraday intervals: 1min, 5min, 15min, 30min, 60min
_INTRADAY: dict[str, str] = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "60min",
}

_SUPPORTED_TIMEFRAMES = [
    Timeframe.M1, Timeframe.M5, Timeframe.M15, Timeframe.M30,
    Timeframe.H1, Timeframe.D1, Timeframe.W1,
]


class AlphaVantageAdapter(DataAdapter):
    """Alpha Vantage equities + FX adapter."""

    def __init__(self, api_key: str, timeout: float = 15.0) -> None:
        if not api_key:
            raise ValueError("AlphaVantageAdapter requires an API key")
        self._api_key = api_key
        self._timeout = timeout

    @property
    def name(self) -> str:
        return "alpha_vantage"

    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            asset_classes=[AssetClass.EQUITIES, AssetClass.FOREX],
            timeframes=_SUPPORTED_TIMEFRAMES,
            max_history_days=7300,
            supports_streaming=False,
            rate_limit_per_minute=5,  # free tier
        )

    async def fetch_ohlcv(
        self,
        symbol: Symbol,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
    ) -> pl.DataFrame:
        tf = timeframe.value
        if symbol.asset_class == AssetClass.EQUITIES:
            df = await self._fetch_equity(symbol, tf)
        elif symbol.asset_class == AssetClass.FOREX:
            df = await self._fetch_forex(symbol, tf)
        else:
            raise ValueError(
                f"alpha_vantage does not support asset class {symbol.asset_class}"
            )
        if df.is_empty():
            return df
        return df.filter((pl.col("timestamp") >= start) & (pl.col("timestamp") <= end))

    async def _fetch_equity(self, symbol: Symbol, tf: str) -> pl.DataFrame:
        ticker = symbol.base.upper()
        if tf in _INTRADAY:
            params = {
                "function": "TIME_SERIES_INTRADAY",
                "symbol": ticker,
                "interval": _INTRADAY[tf],
                "outputsize": "full",
                "apikey": self._api_key,
            }
            series_key_prefix = f"Time Series ({_INTRADAY[tf]})"
        elif tf == "1d":
            params = {
                "function": "TIME_SERIES_DAILY",
                "symbol": ticker,
                "outputsize": "full",
                "apikey": self._api_key,
            }
            series_key_prefix = "Time Series (Daily)"
        elif tf == "1w":
            params = {
                "function": "TIME_SERIES_WEEKLY",
                "symbol": ticker,
                "apikey": self._api_key,
            }
            series_key_prefix = "Weekly Time Series"
        else:
            raise ValueError(f"alpha_vantage equities: unsupported timeframe {tf!r}")

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(_BASE_URL, params=params)
            resp.raise_for_status()
            payload = resp.json()

        series = payload.get(series_key_prefix, {})
        if not series:
            # API returns an informational message on throttle / bad key.
            note = payload.get("Note") or payload.get("Information")
            if note:
                raise RuntimeError(f"alpha_vantage: {note}")
            return pl.DataFrame(schema=OHLCV_SCHEMA)

        rows = []
        for ts_str, bar in series.items():
            try:
                ts = datetime.strptime(
                    ts_str, "%Y-%m-%d %H:%M:%S" if " " in ts_str else "%Y-%m-%d",
                )
                rows.append({
                    "timestamp": ts,
                    "open": float(bar["1. open"]),
                    "high": float(bar["2. high"]),
                    "low": float(bar["3. low"]),
                    "close": float(bar["4. close"]),
                    "volume": float(bar.get("5. volume", 0.0) or 0.0),
                })
            except (KeyError, ValueError):
                continue
        if not rows:
            return pl.DataFrame(schema=OHLCV_SCHEMA)
        return pl.DataFrame(rows, schema=OHLCV_SCHEMA).sort("timestamp")

    async def _fetch_forex(self, symbol: Symbol, tf: str) -> pl.DataFrame:
        # FX functions: FX_INTRADAY, FX_DAILY, FX_WEEKLY.
        base, quote = symbol.base.upper(), symbol.quote.upper()
        if tf in _INTRADAY:
            params = {
                "function": "FX_INTRADAY",
                "from_symbol": base,
                "to_symbol": quote,
                "interval": _INTRADAY[tf],
                "outputsize": "full",
                "apikey": self._api_key,
            }
            series_key_prefix = f"Time Series FX ({_INTRADAY[tf]})"
        elif tf == "1d":
            params = {
                "function": "FX_DAILY",
                "from_symbol": base,
                "to_symbol": quote,
                "outputsize": "full",
                "apikey": self._api_key,
            }
            series_key_prefix = "Time Series FX (Daily)"
        elif tf == "1w":
            params = {
                "function": "FX_WEEKLY",
                "from_symbol": base,
                "to_symbol": quote,
                "apikey": self._api_key,
            }
            series_key_prefix = "Time Series FX (Weekly)"
        else:
            raise ValueError(f"alpha_vantage forex: unsupported timeframe {tf!r}")

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(_BASE_URL, params=params)
            resp.raise_for_status()
            payload = resp.json()

        series = payload.get(series_key_prefix, {})
        if not series:
            note = payload.get("Note") or payload.get("Information")
            if note:
                raise RuntimeError(f"alpha_vantage: {note}")
            return pl.DataFrame(schema=OHLCV_SCHEMA)

        rows = []
        for ts_str, bar in series.items():
            try:
                ts = datetime.strptime(
                    ts_str, "%Y-%m-%d %H:%M:%S" if " " in ts_str else "%Y-%m-%d",
                )
                rows.append({
                    "timestamp": ts,
                    "open": float(bar["1. open"]),
                    "high": float(bar["2. high"]),
                    "low": float(bar["3. low"]),
                    "close": float(bar["4. close"]),
                    "volume": 0.0,  # FX quotes don't have volume
                })
            except (KeyError, ValueError):
                continue
        if not rows:
            return pl.DataFrame(schema=OHLCV_SCHEMA)
        return pl.DataFrame(rows, schema=OHLCV_SCHEMA).sort("timestamp")

    async def health(self) -> AdapterHealth:
        try:
            t0 = time.monotonic()
            params = {
                "function": "GLOBAL_QUOTE",
                "symbol": "AAPL",
                "apikey": self._api_key,
            }
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(_BASE_URL, params=params)
                resp.raise_for_status()
            elapsed = (time.monotonic() - t0) * 1000
            return AdapterHealth(
                status="ok",
                latency_ms=round(elapsed, 1),
                last_successful_call=utcnow(),
            )
        except Exception as exc:  # noqa: BLE001
            return AdapterHealth(status="down", error=str(exc))
