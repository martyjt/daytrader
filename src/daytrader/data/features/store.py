"""Polars-based feature store with Parquet cache.

Stores OHLCV data and precomputed features (TA indicators, ML features)
as Parquet files. Avoids re-downloading or re-computing across backtest
runs and sessions.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl


class FeatureStore:
    """Read-through Parquet cache for OHLCV and feature DataFrames.

    Cache key format: ``{safe_symbol}_{timeframe}_{start}_{end}.parquet``

    Usage::

        store = FeatureStore("data/features")
        df = store.get("crypto:BTC/USDT", "1d", start, end)
        if df is None:
            df = await adapter.fetch_ohlcv(...)
            store.put("crypto:BTC/USDT", "1d", start, end, df)
    """

    def __init__(self, cache_dir: str | Path = "data/features") -> None:
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(
        self,
        symbol_key: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> Path:
        safe = (
            symbol_key.replace("/", "_")
            .replace(":", "_")
            .replace("@", "_")
        )
        start_s = start.strftime("%Y%m%d")
        end_s = end.strftime("%Y%m%d")
        return self._cache_dir / f"{safe}_{timeframe}_{start_s}_{end_s}.parquet"

    def get(
        self,
        symbol_key: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> pl.DataFrame | None:
        """Return cached DataFrame or ``None``."""
        path = self._cache_path(symbol_key, timeframe, start, end)
        if path.exists():
            return pl.read_parquet(path)
        return None

    def put(
        self,
        symbol_key: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        df: pl.DataFrame,
    ) -> Path:
        """Write a DataFrame to the cache. Returns the file path."""
        path = self._cache_path(symbol_key, timeframe, start, end)
        df.write_parquet(path)
        return path

    def has(
        self,
        symbol_key: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> bool:
        return self._cache_path(symbol_key, timeframe, start, end).exists()

    def invalidate(self, symbol_key: str | None = None) -> int:
        """Remove cached files. If ``symbol_key`` is None, clear everything.

        Returns the number of files removed.
        """
        count = 0
        safe_prefix = (
            symbol_key.replace("/", "_").replace(":", "_").replace("@", "_")
            if symbol_key
            else None
        )
        for f in self._cache_dir.glob("*.parquet"):
            if safe_prefix is None or f.name.startswith(safe_prefix):
                f.unlink()
                count += 1
        return count

    def list_cached(self) -> list[str]:
        """List all cache file names (without path)."""
        return sorted(f.name for f in self._cache_dir.glob("*.parquet"))
