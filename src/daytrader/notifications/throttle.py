"""ThrottledNotifier — collapse repeated alerts inside a sliding window."""

from __future__ import annotations

import time
from collections.abc import Callable
from uuid import UUID

from .base import Notifier

DEFAULT_THROTTLE_SECONDS = 300.0  # 5 minutes


class ThrottledNotifier(Notifier):
    """Suppress duplicate ``(tenant_id, dedupe_key)`` notifications.

    Wraps any underlying :class:`Notifier`. Calls without a ``dedupe_key``
    pass straight through — the caller is asserting "always send this
    one". With a key, the wrapper records the wall-clock time of the
    last delivery; identical keys arriving inside ``window_seconds`` are
    dropped. The cache grows unbounded in principle but pays only one
    timestamp per distinct key, so a friend-tier deployment with a
    handful of plugins per tenant tops out at a few KB.
    """

    def __init__(
        self,
        inner: Notifier,
        *,
        window_seconds: float = DEFAULT_THROTTLE_SECONDS,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._inner = inner
        self._window = window_seconds
        self._clock = clock or time.monotonic
        self._last_sent: dict[tuple[UUID, str], float] = {}

    async def notify(
        self,
        tenant_id: UUID,
        message: str,
        *,
        dedupe_key: str | None = None,
    ) -> None:
        if dedupe_key is not None and self._suppressed(tenant_id, dedupe_key):
            return
        await self._inner.notify(tenant_id, message, dedupe_key=dedupe_key)

    def _suppressed(self, tenant_id: UUID, dedupe_key: str) -> bool:
        now = self._clock()
        key = (tenant_id, dedupe_key)
        last = self._last_sent.get(key)
        if last is not None and (now - last) < self._window:
            return True
        self._last_sent[key] = now
        return False

    def reset(self) -> None:
        """Drop the dedupe map (used by tests)."""
        self._last_sent.clear()
