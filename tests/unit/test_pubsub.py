"""Tests for the in-process signal bus."""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from daytrader.core.pubsub import (
    DEFAULT_MAX_QUEUE,
    SignalBus,
    SignalEvent,
    reset_signal_bus,
    signal_bus,
)


def _event(tenant_id, **overrides):
    base = dict(
        tenant_id=tenant_id,
        persona_id=uuid4(),
        signal_id=uuid4(),
        symbol_key="BTC-USD",
        score=0.5,
        confidence=0.9,
        source="test",
        reason="",
        created_at="2026-04-25T00:00:00+00:00",
    )
    base.update(overrides)
    return SignalEvent(**base)


@pytest.fixture(autouse=True)
def _reset_bus():
    reset_signal_bus()
    yield
    reset_signal_bus()


async def test_publish_with_no_subscribers_is_noop():
    bus = SignalBus()
    bus.publish(uuid4(), _event(uuid4()))  # must not raise


async def test_subscribe_receives_published_event():
    tid = uuid4()
    bus = SignalBus()
    with bus.subscribe(tid) as q:
        ev = _event(tid)
        bus.publish(tid, ev)
        received = await asyncio.wait_for(q.get(), timeout=1.0)
    assert received is ev


async def test_subscribe_isolates_tenants():
    """A subscriber for tenant A never sees tenant B's events."""
    a, b = uuid4(), uuid4()
    bus = SignalBus()
    with bus.subscribe(a) as qa, bus.subscribe(b) as qb:
        bus.publish(a, _event(a, symbol_key="A1"))
        bus.publish(b, _event(b, symbol_key="B1"))
        ra = await asyncio.wait_for(qa.get(), timeout=1.0)
        rb = await asyncio.wait_for(qb.get(), timeout=1.0)
    assert ra.symbol_key == "A1"
    assert rb.symbol_key == "B1"
    assert qa.empty() and qb.empty()


async def test_multiple_subscribers_each_get_event():
    tid = uuid4()
    bus = SignalBus()
    with bus.subscribe(tid) as q1, bus.subscribe(tid) as q2:
        ev = _event(tid)
        bus.publish(tid, ev)
        r1 = await asyncio.wait_for(q1.get(), timeout=1.0)
        r2 = await asyncio.wait_for(q2.get(), timeout=1.0)
    assert r1 is ev and r2 is ev


async def test_unsubscribe_on_context_exit():
    tid = uuid4()
    bus = SignalBus()
    assert bus.subscriber_count(tid) == 0
    with bus.subscribe(tid):
        assert bus.subscriber_count(tid) == 1
    assert bus.subscriber_count(tid) == 0


async def test_full_queue_drops_oldest():
    tid = uuid4()
    bus = SignalBus(max_queue=3)
    with bus.subscribe(tid) as q:
        # Publish 5 events into a queue capped at 3.
        events = [_event(tid, symbol_key=f"S{i}") for i in range(5)]
        for ev in events:
            bus.publish(tid, ev)
        # Queue should hold the last 3 (S2, S3, S4).
        assert q.qsize() == 3
        received = [await q.get() for _ in range(3)]
    assert [r.symbol_key for r in received] == ["S2", "S3", "S4"]


async def test_subscribe_rejects_none_tenant():
    bus = SignalBus()
    with pytest.raises(ValueError):
        with bus.subscribe(None):
            pass


def test_signal_bus_is_singleton():
    assert signal_bus() is signal_bus()


async def test_default_max_queue_is_reasonable():
    """Sanity check: default cap is large enough not to surprise the UI."""
    assert DEFAULT_MAX_QUEUE >= 50
