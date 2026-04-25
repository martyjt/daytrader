"""Shared market-data fan-out bus (Phase 9 — productionization).

One OHLCV fetch per ``(adapter, symbol, timeframe, time-bucket)``
regardless of how many personas trigger it within a poll cycle. The
trading loop spins up many personas in parallel; without this bus, ten
personas trading BTC-USDT on 1m would each issue an HTTP round-trip to
the adapter every 30 s. With the bus, the second through tenth callers
either piggy-back on the in-flight fetch or read a fresh cache entry.

Design points:

* **Cache key includes a quantized timestamp range** — callers compute
  ``end = utcnow()`` independently, so without quantization their
  millisecond-different ``start``/``end`` values would each be a unique
  key. The bucket size is the timeframe's own granularity, so two 1m
  callers within the same minute hit the same bucket; two 1h callers
  within the same hour likewise.
* **Single in-flight de-dupe** — concurrent callers for an in-flight key
  await the same task instead of each spawning their own RPC.
* **Cache failures don't poison** — if the adapter raises, the failing
  task is removed from the in-flight map and the cache is *not*
  populated. Subsequent callers retry.
* **TTL** — short (~30 s default). Long enough to dedupe a poll cycle,
  short enough that a freshly-closed bar shows up on the next cycle.
* **Backtests bypass the bus** — they pass historical ``data`` directly
  into the engine and never call ``adapter.fetch_ohlcv``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from ..core.types.bars import Timeframe
    from ..core.types.symbols import Symbol
    from .adapters.base import DataAdapter

logger = logging.getLogger(__name__)

DEFAULT_TTL_SECONDS = 30.0


CacheKey = tuple[str, str, str, int, int]


@dataclass
class _Entry:
    data: pl.DataFrame
    cached_at: float


@dataclass
class BusMetrics:
    """Runtime counters for the operational dashboard (added in a later phase)."""

    hits: int = 0
    misses: int = 0
    inflight_dedup: int = 0
    errors: int = 0

    def as_dict(self) -> dict[str, int | float]:
        total = self.hits + self.misses + self.inflight_dedup
        hit_ratio = (
            (self.hits + self.inflight_dedup) / total if total else 0.0
        )
        return {
            "hits": self.hits,
            "misses": self.misses,
            "inflight_dedup": self.inflight_dedup,
            "errors": self.errors,
            "hit_ratio": hit_ratio,
        }


class MarketDataBus:
    """Process-global OHLCV fan-out cache."""

    def __init__(
        self,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._ttl = ttl_seconds
        self._cache: dict[CacheKey, _Entry] = {}
        self._inflight: dict[CacheKey, asyncio.Task[pl.DataFrame]] = {}
        self._lock = asyncio.Lock()
        self._clock = clock or time.monotonic
        self.metrics = BusMetrics()

    async def fetch_ohlcv(
        self,
        adapter: DataAdapter,
        symbol: Symbol,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
    ) -> pl.DataFrame:
        """Fetch OHLCV via the cache, deduping concurrent and recent calls.

        Raises whatever the underlying adapter raises if the fetch
        fails; the failing entry is *not* cached.
        """
        key = self._make_key(adapter.name, symbol, timeframe, start, end)
        now = self._clock()

        async with self._lock:
            entry = self._cache.get(key)
            if entry is not None and now - entry.cached_at < self._ttl:
                self.metrics.hits += 1
                return entry.data

            inflight = self._inflight.get(key)
            if inflight is not None:
                self.metrics.inflight_dedup += 1
            else:
                self.metrics.misses += 1
                inflight = asyncio.create_task(
                    self._fetch_and_cache(
                        key, adapter, symbol, timeframe, start, end
                    )
                )
                self._inflight[key] = inflight

        return await inflight

    async def _fetch_and_cache(
        self,
        key: CacheKey,
        adapter: DataAdapter,
        symbol: Symbol,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
    ) -> pl.DataFrame:
        try:
            data = await adapter.fetch_ohlcv(symbol, timeframe, start, end)
            self._cache[key] = _Entry(data=data, cached_at=self._clock())
            return data
        except Exception:
            self.metrics.errors += 1
            raise
        finally:
            async with self._lock:
                self._inflight.pop(key, None)

    @staticmethod
    def _make_key(
        adapter_name: str,
        symbol: Symbol,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
    ) -> CacheKey:
        secs = timeframe.seconds
        q_start = int(start.timestamp()) // secs * secs
        q_end = int(end.timestamp()) // secs * secs
        return (adapter_name, str(symbol), str(timeframe), q_start, q_end)

    def reset(self) -> None:
        """Drop cache + counters (for tests and on shutdown)."""
        self._cache.clear()
        self._inflight.clear()
        self.metrics = BusMetrics()


# ---------------------------------------------------------------------------
# Process-global active bus
# ---------------------------------------------------------------------------

_active_bus: MarketDataBus | None = None


def set_active_bus(bus: MarketDataBus | None) -> None:
    """Install the process-global market data bus.

    Called once during ``ui/app.py:_startup``. Idempotent; passing
    ``None`` clears it (used by tests).
    """
    global _active_bus
    _active_bus = bus


def get_active_bus() -> MarketDataBus | None:
    """Return the active bus or ``None`` if startup hasn't run.

    Returning ``None`` lets the trading loop fall back to a direct
    adapter call in environments where the bus isn't installed (unit
    tests, scripts).
    """
    return _active_bus
