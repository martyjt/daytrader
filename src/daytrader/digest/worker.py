"""Per-tenant daily digest worker.

Sleeps until the next 08:00 local, builds the digest, fires it at the
active notifier, then loops. Backed by an asyncio task so it shuts down
cleanly with the rest of the supervisor stack.

We deliberately avoid pulling in APScheduler here — the cadence is
"once per day", and a single sleep + wall-clock target keeps shutdown
behaviour trivial (cancel the task, move on). APScheduler's value lives
on the Exploration / Shadow workers where multiple jobs share a process.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable
from datetime import datetime, time, timedelta
from uuid import UUID

from ..notifications import notify_active
from .service import build_digest, format_digest

logger = logging.getLogger(__name__)

DEFAULT_HOUR_LOCAL = 8
DEFAULT_MINUTE_LOCAL = 0


class DailyDigestWorker:
    """One async loop per tenant — sleep, send, repeat."""

    def __init__(
        self,
        tenant_id: UUID,
        *,
        hour_local: int = DEFAULT_HOUR_LOCAL,
        minute_local: int = DEFAULT_MINUTE_LOCAL,
        clock: Callable[[], datetime] | None = None,
        sleeper: Callable[[float], asyncio.Future[None] | asyncio.Task[None]] | None = None,
        notify: Callable[..., asyncio.Future[None]] | None = None,
    ) -> None:
        self._tenant_id = tenant_id
        self._hour = int(hour_local)
        self._minute = int(minute_local)
        # ``datetime.now()`` (local tz) by default — APScheduler does the
        # same for the Exploration/Shadow scans so behaviour stays
        # consistent across the production cadence.
        self._clock = clock or datetime.now
        self._sleep = sleeper or asyncio.sleep
        self._notify = notify or notify_active
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()

    @property
    def tenant_id(self) -> UUID:
        return self._tenant_id

    async def start(self) -> None:
        if self._task is not None:
            return
        self._stop.clear()
        self._task = asyncio.create_task(
            self._run(), name=f"digest-{self._tenant_id}"
        )

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    def seconds_until_next_run(self, now: datetime | None = None) -> float:
        """Return seconds from ``now`` to the next ``hour:minute`` local boundary.

        If we're already past today's boundary, schedule for tomorrow.
        """
        current = now or self._clock()
        target = current.replace(
            hour=self._hour, minute=self._minute, second=0, microsecond=0
        )
        if target <= current:
            target = target + timedelta(days=1)
        return max(0.0, (target - current).total_seconds())

    async def run_once(self, now: datetime | None = None) -> str:
        """Build + send a single digest. Returns the formatted text.

        Exposed so the supervisor's first refresh can fire one immediately
        after start (test hook), and so admin pages can preview without
        waiting for 08:00.
        """
        anchor = now or self._clock()
        summary = await build_digest(self._tenant_id, window_end=anchor)
        message = format_digest(summary)
        await self._notify(
            self._tenant_id, message, dedupe_key=f"digest:{anchor.date()}"
        )
        return message

    async def _run(self) -> None:
        while not self._stop.is_set():
            try:
                wait_seconds = self.seconds_until_next_run()
                logger.info(
                    "digest worker for tenant %s sleeping %.0fs",
                    self._tenant_id, wait_seconds,
                )
                await self._sleep(wait_seconds)
                if self._stop.is_set():
                    return
                await self.run_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                # Logged + swallowed: a flaky DB query at 08:00 must not
                # bring down the supervisor. The next day's run will retry.
                logger.exception(
                    "digest worker failed for tenant %s", self._tenant_id
                )
                # Avoid a hot loop if the failure is immediate; sleep an
                # hour and try again rather than spinning.
                with contextlib.suppress(asyncio.CancelledError):
                    await self._sleep(3600.0)


def parse_local_time(value: str | None) -> tuple[int, int]:
    """Parse ``HH:MM`` env values; fall back to the 08:00 default."""
    if not value:
        return DEFAULT_HOUR_LOCAL, DEFAULT_MINUTE_LOCAL
    try:
        parsed = time.fromisoformat(value)
    except ValueError:
        logger.warning(
            "Invalid DIGEST_LOCAL_TIME %r — using default %02d:%02d",
            value, DEFAULT_HOUR_LOCAL, DEFAULT_MINUTE_LOCAL,
        )
        return DEFAULT_HOUR_LOCAL, DEFAULT_MINUTE_LOCAL
    return parsed.hour, parsed.minute
