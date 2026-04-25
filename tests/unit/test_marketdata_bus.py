"""Tests for the Phase 9 shared market-data bus."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from datetime import UTC, datetime

import polars as pl

from daytrader.core.types.bars import Timeframe
from daytrader.core.types.symbols import Symbol
from daytrader.data.marketdata_bus import MarketDataBus


class _FakeClock:
    """Hand-driven monotonic clock so tests are deterministic on TTL."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _CountingAdapter:
    """Minimal DataAdapter stub. Counts every fetch and optionally blocks."""

    name = "fake"

    def __init__(self, delay: float = 0.0) -> None:
        self.calls = 0
        self.delay = delay

    async def fetch_ohlcv(
        self,
        symbol: Symbol,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
    ) -> pl.DataFrame:
        self.calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        return pl.DataFrame(
            {
                "timestamp": [end],
                "open": [1.0],
                "high": [1.0],
                "low": [1.0],
                "close": [1.0],
                "volume": [1.0],
            }
        )


SYMBOL = Symbol.parse("BTC-USDT")
TF = Timeframe.M1
T0 = datetime(2026, 4, 26, 12, 0, 0, tzinfo=UTC)
T1 = datetime(2026, 4, 26, 13, 0, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# In-flight de-dupe
# ---------------------------------------------------------------------------


async def test_concurrent_callers_share_one_fetch():
    """10 simultaneous identical calls → 1 adapter fetch."""
    adapter = _CountingAdapter(delay=0.05)
    bus = MarketDataBus()

    results = await asyncio.gather(
        *(bus.fetch_ohlcv(adapter, SYMBOL, TF, T0, T1) for _ in range(10))
    )

    assert adapter.calls == 1
    assert all(not r.is_empty() for r in results)
    assert bus.metrics.misses == 1
    assert bus.metrics.inflight_dedup == 9


async def test_distinct_symbols_do_not_collide():
    """Different symbols share no key → independent fetches."""
    adapter = _CountingAdapter()
    bus = MarketDataBus()

    sym_a = Symbol.parse("BTC-USDT")
    sym_b = Symbol.parse("ETH-USDT")

    await asyncio.gather(
        bus.fetch_ohlcv(adapter, sym_a, TF, T0, T1),
        bus.fetch_ohlcv(adapter, sym_b, TF, T0, T1),
    )

    assert adapter.calls == 2


async def test_distinct_timeframes_do_not_collide():
    adapter = _CountingAdapter()
    bus = MarketDataBus()

    await asyncio.gather(
        bus.fetch_ohlcv(adapter, SYMBOL, Timeframe.M1, T0, T1),
        bus.fetch_ohlcv(adapter, SYMBOL, Timeframe.H1, T0, T1),
    )

    assert adapter.calls == 2


# ---------------------------------------------------------------------------
# Cache hits / TTL
# ---------------------------------------------------------------------------


async def test_second_call_within_ttl_is_a_cache_hit():
    adapter = _CountingAdapter()
    clock = _FakeClock()
    bus = MarketDataBus(ttl_seconds=30.0, clock=clock)

    await bus.fetch_ohlcv(adapter, SYMBOL, TF, T0, T1)
    clock.advance(10)
    await bus.fetch_ohlcv(adapter, SYMBOL, TF, T0, T1)

    assert adapter.calls == 1
    assert bus.metrics.hits == 1
    assert bus.metrics.misses == 1


async def test_expired_entry_is_refetched():
    adapter = _CountingAdapter()
    clock = _FakeClock()
    bus = MarketDataBus(ttl_seconds=30.0, clock=clock)

    await bus.fetch_ohlcv(adapter, SYMBOL, TF, T0, T1)
    clock.advance(31)
    await bus.fetch_ohlcv(adapter, SYMBOL, TF, T0, T1)

    assert adapter.calls == 2


async def test_close_timestamps_in_same_bucket_dedupe():
    """Two callers with millisecond-different ``end`` values still share a key."""
    adapter = _CountingAdapter()
    bus = MarketDataBus()

    end_a = datetime(2026, 4, 26, 13, 0, 0, 123_000, tzinfo=UTC)
    end_b = datetime(2026, 4, 26, 13, 0, 0, 987_000, tzinfo=UTC)

    # Same start, different sub-minute end → should hit the same 1m bucket.
    await asyncio.gather(
        bus.fetch_ohlcv(adapter, SYMBOL, Timeframe.M1, T0, end_a),
        bus.fetch_ohlcv(adapter, SYMBOL, Timeframe.M1, T0, end_b),
    )

    assert adapter.calls == 1


# ---------------------------------------------------------------------------
# Failures don't poison the cache
# ---------------------------------------------------------------------------


async def test_failed_fetch_not_cached_and_retried():
    """Adapter raises → entry not cached, error counter ticks, retry works."""

    class _FailingThenWorking:
        name = "fake"

        def __init__(self) -> None:
            self.calls = 0

        async def fetch_ohlcv(self, symbol, timeframe, start, end):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("network blip")
            return pl.DataFrame(
                {
                    "timestamp": [end],
                    "open": [1.0],
                    "high": [1.0],
                    "low": [1.0],
                    "close": [1.0],
                    "volume": [1.0],
                }
            )

    adapter = _FailingThenWorking()
    bus = MarketDataBus()

    with suppress(RuntimeError):
        await bus.fetch_ohlcv(adapter, SYMBOL, TF, T0, T1)

    # Second call must NOT see the failed entry — it should retry.
    df = await bus.fetch_ohlcv(adapter, SYMBOL, TF, T0, T1)
    assert not df.is_empty()
    assert adapter.calls == 2
    assert bus.metrics.errors == 1


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


async def test_metrics_as_dict_reports_hit_ratio():
    adapter = _CountingAdapter(delay=0.02)
    bus = MarketDataBus()

    # 1 miss + 4 inflight dedupes → 5 calls, 1 actual fetch
    await asyncio.gather(
        *(bus.fetch_ohlcv(adapter, SYMBOL, TF, T0, T1) for _ in range(5))
    )
    # 1 cache hit on the same key
    await bus.fetch_ohlcv(adapter, SYMBOL, TF, T0, T1)

    m = bus.metrics.as_dict()
    assert m["misses"] == 1
    assert m["inflight_dedup"] == 4
    assert m["hits"] == 1
    # 5 of 6 effectively avoided an RPC.
    assert abs(m["hit_ratio"] - 5 / 6) < 1e-9


# ---------------------------------------------------------------------------
# Process-global accessor
# ---------------------------------------------------------------------------


def test_active_bus_starts_unset_and_can_be_installed_and_cleared():
    from daytrader.data.marketdata_bus import (
        get_active_bus,
        set_active_bus,
    )

    assert get_active_bus() is None or isinstance(get_active_bus(), MarketDataBus)

    bus = MarketDataBus()
    set_active_bus(bus)
    assert get_active_bus() is bus

    set_active_bus(None)
    assert get_active_bus() is None
