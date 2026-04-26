"""MacroAdapter ABC — univariate economic time-series.

Macro series have a single numeric value per timestamp and come in at
widely varying cadences (daily treasury rates, weekly jobless claims,
monthly CPI, quarterly GDP). That doesn't fit the OHLCV schema, so
macro adapters have their own interface.

The canonical schema returned by ``fetch_series`` is:

    timestamp (pl.Datetime), value (pl.Float64)

Missing observations are dropped server-side — the caller aligns to a
trading calendar if needed (typical approach: forward-fill daily).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar

import polars as pl

from ..adapters.base import AdapterHealth  # reuse the same health shape

MACRO_SCHEMA = {
    "timestamp": pl.Datetime,
    "value": pl.Float64,
}


@dataclass(frozen=True)
class MacroSeries:
    """Descriptor for a single macro series the adapter can fetch."""

    series_id: str
    title: str
    units: str = ""
    frequency: str = ""  # "daily" | "weekly" | "monthly" | "quarterly" | "annual"
    source: str = ""


class MacroAdapter(ABC):
    """Abstract base for macro/economic data adapters (FRED, Eurostat, …)."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique adapter identifier, e.g. ``'fred'``."""

    @abstractmethod
    async def fetch_series(
        self,
        series_id: str,
        start: datetime,
        end: datetime,
    ) -> pl.DataFrame:
        """Fetch one macro series. Returns a Polars DF matching ``MACRO_SCHEMA``."""

    @abstractmethod
    async def health(self) -> AdapterHealth:
        """Probe operational health."""

    async def search(self, query: str, limit: int = 20) -> list[MacroSeries]:
        """Optional: search for series by keyword. Default: not implemented."""
        raise NotImplementedError


class MacroAdapterRegistry:
    """Registry of available macro adapters."""

    _adapters: ClassVar[dict[str, MacroAdapter]] = {}

    @classmethod
    def register(cls, adapter: MacroAdapter) -> None:
        cls._adapters[adapter.name] = adapter

    @classmethod
    def get(cls, name: str) -> MacroAdapter:
        if name not in cls._adapters:
            raise KeyError(
                f"Macro adapter {name!r} not registered. "
                f"Available: {sorted(cls._adapters)}"
            )
        return cls._adapters[name]

    @classmethod
    def available(cls) -> list[str]:
        return sorted(cls._adapters)

    @classmethod
    def auto_register(cls) -> None:
        """Register built-in macro adapters if their API keys are present."""
        if "fred" not in cls._adapters:
            try:
                from ...core.settings import get_settings

                key = get_settings().fred_api_key.get_secret_value()
                if key:
                    from .fred_adapter import FredAdapter

                    cls.register(FredAdapter(api_key=key))
            except Exception:
                pass

    @classmethod
    def clear(cls) -> None:
        cls._adapters.clear()
