"""In-process pub/sub for live UI streaming, keyed by ``tenant_id``.

A publisher (e.g. the trading loop, after it has persisted a signal) calls
``signal_bus().publish(tenant_id, payload)``. Each subscriber holds an
``asyncio.Queue`` registered under its tenant; ``publish`` fans the
payload out to every queue belonging to that tenant.

Cross-tenant isolation is structural: a subscriber registered for
tenant A literally never sees a payload published for tenant B because
the dispatch table is keyed on ``tenant_id`` and we only iterate the
matching bucket. Callers cannot subscribe with ``None``.

Backpressure policy: each subscriber queue has a small fixed size
(``DEFAULT_MAX_QUEUE``). When full, the oldest message is dropped to
make room. We prefer "lose the oldest tick" over blocking the
publisher, since this bus drives a UI feed where staleness is worse
than gaps.

Out of scope (Phase 4): cross-process delivery (Redis), reconnect
replay, durable history. A page that re-mounts after a disconnect
reloads its initial state from the database.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from typing import Any
from uuid import UUID

logger = logging.getLogger(__name__)


DEFAULT_MAX_QUEUE = 100


@dataclass(frozen=True)
class SignalEvent:
    """A signal emission published over the bus."""

    tenant_id: UUID
    persona_id: UUID
    signal_id: UUID
    symbol_key: str
    score: float
    confidence: float
    source: str
    reason: str
    created_at: str  # ISO-8601 string — JSON-friendly for UI consumers


class SignalBus:
    """In-process fan-out keyed by tenant_id."""

    def __init__(self, *, max_queue: int = DEFAULT_MAX_QUEUE) -> None:
        self._max_queue = max_queue
        self._subscribers: dict[UUID, set[asyncio.Queue]] = {}

    @contextmanager
    def subscribe(self, tenant_id: UUID) -> Iterator[asyncio.Queue]:
        """Register a queue for ``tenant_id``; auto-deregister on exit.

        Usage::

            with signal_bus().subscribe(tid) as q:
                while True:
                    event = await q.get()
                    ...
        """
        if tenant_id is None:
            raise ValueError("subscribe() requires a non-None tenant_id")
        q: asyncio.Queue = asyncio.Queue(maxsize=self._max_queue)
        self._subscribers.setdefault(tenant_id, set()).add(q)
        try:
            yield q
        finally:
            bucket = self._subscribers.get(tenant_id)
            if bucket is not None:
                bucket.discard(q)
                if not bucket:
                    self._subscribers.pop(tenant_id, None)

    def publish(self, tenant_id: UUID, event: Any) -> None:
        """Fan out ``event`` to every subscriber for ``tenant_id``.

        On full queues, drops the oldest item to make room — never
        blocks the caller. Safe to call from any coroutine on the
        running event loop.
        """
        bucket = self._subscribers.get(tenant_id)
        if not bucket:
            return
        for q in bucket:
            _put_drop_oldest(q, event)

    def subscriber_count(self, tenant_id: UUID) -> int:
        """How many live subscribers are registered for ``tenant_id`` (test hook)."""
        bucket = self._subscribers.get(tenant_id)
        return len(bucket) if bucket else 0


def _put_drop_oldest(q: asyncio.Queue, item: Any) -> None:
    """Best-effort enqueue. On full, drop the oldest entry first."""
    try:
        q.put_nowait(item)
        return
    except asyncio.QueueFull:
        pass
    with suppress(asyncio.QueueEmpty):
        q.get_nowait()
    try:
        q.put_nowait(item)
    except asyncio.QueueFull:
        # Another consumer beat us to it — silently drop this item.
        logger.debug("pubsub: dropped event after eviction race")


_bus: SignalBus | None = None


def signal_bus() -> SignalBus:
    """Return the process-wide ``SignalBus`` (lazy singleton)."""
    global _bus
    if _bus is None:
        _bus = SignalBus()
    return _bus


def reset_signal_bus() -> None:
    """Test hook — drop the singleton so each test gets a fresh bus."""
    global _bus
    _bus = None
