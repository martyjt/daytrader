"""DataAdapter ABC and supporting types.

Every data source (yfinance, ccxt, alpaca, ...) implements this
interface. The rest of the system never touches vendor-specific code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime

import polars as pl

from ...core.types.bars import Timeframe
from ...core.types.symbols import AssetClass, Symbol

# Canonical Polars schema for OHLCV data returned by every adapter.
OHLCV_SCHEMA = {
    "timestamp": pl.Datetime,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
}


@dataclass(frozen=True)
class AdapterCapabilities:
    """What a data adapter can provide."""

    asset_classes: list[AssetClass]
    timeframes: list[Timeframe]
    max_history_days: int
    supports_streaming: bool = False
    rate_limit_per_minute: int = 60


@dataclass
class AdapterHealth:
    """Current health of a data adapter."""

    status: str  # "ok" | "degraded" | "down"
    latency_ms: float | None = None
    last_successful_call: datetime | None = None
    error: str | None = None


class DataAdapter(ABC):
    """Abstract base for all data source adapters.

    Subclasses must implement three methods:

    * ``capabilities()`` — declares supported asset classes and timeframes.
    * ``fetch_ohlcv()`` — returns historical bars as a Polars DataFrame.
    * ``health()`` — reports current operational status.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique adapter identifier, e.g. ``'yfinance'``, ``'ccxt_binance'``."""

    @abstractmethod
    def capabilities(self) -> AdapterCapabilities:
        """Return the adapter's declared capabilities."""

    @abstractmethod
    async def fetch_ohlcv(
        self,
        symbol: Symbol,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
    ) -> pl.DataFrame:
        """Fetch historical OHLCV data.

        Must return a Polars DataFrame matching ``OHLCV_SCHEMA``.
        """

    @abstractmethod
    async def health(self) -> AdapterHealth:
        """Probe operational health."""
