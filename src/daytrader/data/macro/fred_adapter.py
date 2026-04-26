"""FRED (Federal Reserve Economic Data) adapter.

Covers the macro dataset used by the Exploration Agent: rates (DGS10,
DGS2), VIX (VIXCLS), CPI (CPIAUCSL), unemployment (UNRATE), yield-curve
slopes (T10Y2Y), DXY dollar index (DTWEXBGS), and many more.

Free tier: unlimited requests with an API key from
``https://fred.stlouisfed.org/docs/api/api_key.html``. Store as
``FRED_API_KEY`` in ``.env`` or the environment.
"""

from __future__ import annotations

import time
from datetime import datetime

import httpx
import polars as pl

from ...core.types.common import utcnow
from ..adapters.base import AdapterHealth
from .base import MACRO_SCHEMA, MacroAdapter, MacroSeries

_BASE_URL = "https://api.stlouisfed.org/fred"


class FredAdapter(MacroAdapter):
    """FRED macro data adapter."""

    def __init__(self, api_key: str, timeout: float = 10.0) -> None:
        if not api_key:
            raise ValueError("FredAdapter requires an API key")
        self._api_key = api_key
        self._timeout = timeout

    @property
    def name(self) -> str:
        return "fred"

    async def fetch_series(
        self,
        series_id: str,
        start: datetime,
        end: datetime,
    ) -> pl.DataFrame:
        params: dict[str, str | int] = {
            "series_id": series_id,
            "api_key": self._api_key,
            "file_type": "json",
            "observation_start": start.strftime("%Y-%m-%d"),
            "observation_end": end.strftime("%Y-%m-%d"),
        }
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(f"{_BASE_URL}/series/observations", params=params)
            resp.raise_for_status()
            payload = resp.json()

        observations = payload.get("observations", [])
        if not observations:
            return pl.DataFrame(schema=MACRO_SCHEMA)

        rows: list[dict] = []
        for obs in observations:
            # FRED uses "." for missing values.
            val = obs.get("value")
            if val in (None, ".", ""):
                continue
            try:
                rows.append({
                    "timestamp": datetime.strptime(obs["date"], "%Y-%m-%d"),
                    "value": float(val),
                })
            except (ValueError, KeyError):
                continue
        if not rows:
            return pl.DataFrame(schema=MACRO_SCHEMA)
        return pl.DataFrame(rows, schema=MACRO_SCHEMA)

    async def search(self, query: str, limit: int = 20) -> list[MacroSeries]:
        params: dict[str, str | int] = {
            "search_text": query,
            "api_key": self._api_key,
            "file_type": "json",
            "limit": limit,
        }
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(f"{_BASE_URL}/series/search", params=params)
            resp.raise_for_status()
            payload = resp.json()

        return [
            MacroSeries(
                series_id=s["id"],
                title=s.get("title", s["id"]),
                units=s.get("units", ""),
                frequency=s.get("frequency", ""),
                source="FRED",
            )
            for s in payload.get("seriess", [])
        ]

    async def health(self) -> AdapterHealth:
        # Probe with a cheap, stable series (10-Year Treasury Rate).
        try:
            t0 = time.monotonic()
            params: dict[str, str | int] = {
                "series_id": "DGS10",
                "api_key": self._api_key,
                "file_type": "json",
                "limit": 1,
                "sort_order": "desc",
            }
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(
                    f"{_BASE_URL}/series/observations", params=params
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


# Curated list of useful series for the Exploration Agent to seed candidates
# from. Users can always pass arbitrary series IDs — this is just a default.
CURATED_SERIES: list[MacroSeries] = [
    MacroSeries("DGS10", "10-Year Treasury Constant Maturity Rate", "percent", "daily", "FRED"),
    MacroSeries("DGS2", "2-Year Treasury Constant Maturity Rate", "percent", "daily", "FRED"),
    MacroSeries("T10Y2Y", "10Y-2Y Treasury Yield Spread", "percent", "daily", "FRED"),
    MacroSeries("VIXCLS", "CBOE Volatility Index (VIX)", "index", "daily", "FRED"),
    MacroSeries("DTWEXBGS", "Trade Weighted US Dollar Index (Broad)", "index", "daily", "FRED"),
    MacroSeries("DFF", "Federal Funds Effective Rate", "percent", "daily", "FRED"),
    MacroSeries("UNRATE", "Unemployment Rate", "percent", "monthly", "FRED"),
    MacroSeries("CPIAUCSL", "Consumer Price Index (Seasonally Adjusted)", "index", "monthly", "FRED"),
    MacroSeries("HOUST", "Housing Starts", "thousands", "monthly", "FRED"),
    MacroSeries("ICSA", "Initial Jobless Claims", "claims", "weekly", "FRED"),
]
