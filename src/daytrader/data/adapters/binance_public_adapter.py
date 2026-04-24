"""Binance public data adapter — no API key required.

Uses the Binance public REST API (``/api/v3/klines``) for spot OHLCV.
Covers crypto pairs quoted in USDT, USDC, BUSD, BTC, ETH, etc. Rate
limit on the public endpoint is 1200 request-weight per minute — each
klines request costs 1 weight, so effectively 1000 req/min.

Use this in preference to yfinance for crypto: it's the source of
truth for Binance pair prices, supports proper intraday intervals
(including 4h which yfinance lacks), and has no unofficial-API risk.
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

_BASE_URL = "https://api.binance.com/api/v3"

_INTERVALS: dict[str, str] = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
    "1w": "1w",
}

_SUPPORTED_TIMEFRAMES = [
    Timeframe.M1, Timeframe.M5, Timeframe.M15, Timeframe.M30,
    Timeframe.H1, Timeframe.H4, Timeframe.D1, Timeframe.W1,
]

_MAX_LIMIT_PER_REQ = 1000


class BinancePublicAdapter(DataAdapter):
    """Binance public spot OHLCV adapter — no key, no auth."""

    def __init__(self, timeout: float = 10.0) -> None:
        self._timeout = timeout

    @property
    def name(self) -> str:
        return "binance_public"

    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            asset_classes=[AssetClass.CRYPTO],
            timeframes=_SUPPORTED_TIMEFRAMES,
            max_history_days=3650,  # effectively unbounded for klines
            supports_streaming=False,
            rate_limit_per_minute=1000,
        )

    async def fetch_ohlcv(
        self,
        symbol: Symbol,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
    ) -> pl.DataFrame:
        if symbol.asset_class != AssetClass.CRYPTO:
            raise ValueError(
                f"binance_public only supports crypto; got {symbol.asset_class}"
            )
        interval = _INTERVALS.get(timeframe.value)
        if interval is None:
            raise ValueError(
                f"binance_public does not support interval {timeframe.value!r}. "
                f"Supported: {sorted(_INTERVALS)}"
            )

        # Binance doesn't list a USD-quoted BTC pair — use USDT as the
        # USD-stand-in. Saves callers from having to know which stablecoin
        # Binance prefers for each pair.
        quote = symbol.quote.upper()
        if quote == "USD":
            quote = "USDT"
        pair = f"{symbol.base.upper()}{quote}"
        start_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)

        rows: list[list] = []
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            # Page through the range; each request returns up to 1000 bars.
            cursor_ms = start_ms
            while cursor_ms < end_ms:
                params = {
                    "symbol": pair,
                    "interval": interval,
                    "startTime": cursor_ms,
                    "endTime": end_ms,
                    "limit": _MAX_LIMIT_PER_REQ,
                }
                resp = await client.get(f"{_BASE_URL}/klines", params=params)
                resp.raise_for_status()
                batch = resp.json()
                if not batch:
                    break
                rows.extend(batch)
                # Advance cursor just past the last bar's open time.
                last_open = batch[-1][0]
                next_cursor = last_open + 1
                if next_cursor <= cursor_ms:
                    break  # safety — avoid infinite loop on weird data
                cursor_ms = next_cursor
                if len(batch) < _MAX_LIMIT_PER_REQ:
                    break

        if not rows:
            return pl.DataFrame(schema=OHLCV_SCHEMA)

        # Binance kline schema:
        # [open_time_ms, open, high, low, close, volume, close_time_ms, …]
        records = [
            {
                "timestamp": datetime.fromtimestamp(r[0] / 1000),
                "open": float(r[1]),
                "high": float(r[2]),
                "low": float(r[3]),
                "close": float(r[4]),
                "volume": float(r[5]),
            }
            for r in rows
        ]
        return pl.DataFrame(records, schema=OHLCV_SCHEMA)

    async def health(self) -> AdapterHealth:
        try:
            t0 = time.monotonic()
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(f"{_BASE_URL}/ping")
                resp.raise_for_status()
            elapsed = (time.monotonic() - t0) * 1000
            return AdapterHealth(
                status="ok",
                latency_ms=round(elapsed, 1),
                last_successful_call=utcnow(),
            )
        except Exception as exc:  # noqa: BLE001
            return AdapterHealth(status="down", error=str(exc))
