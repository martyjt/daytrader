"""Background regime watcher.

Periodically re-runs ``get_current_regime()`` so the Regime Badge reflects
the current broad-market state without a user having to click anything,
and the alert system fires automatically when the regime flips.

Scheduled via APScheduler; interval is configurable in AppSettings
(``regime_refresh_minutes``). Default 30 minutes.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)


class RegimeWatcher:
    """Keep the regime cache warm + trigger regime-change alerts."""

    def __init__(
        self,
        *,
        interval_minutes: float = 30.0,
        pulse_symbol: str = "BTC-USD",
        pulse_timeframe: str = "1d",
    ) -> None:
        self._interval = max(1.0, float(interval_minutes))
        self._symbol = pulse_symbol
        self._timeframe = pulse_timeframe
        self._scheduler: Any = None
        self._last_run: datetime | None = None
        self._last_regime: str | None = None

    @property
    def last_run(self) -> datetime | None:
        return self._last_run

    @property
    def last_regime(self) -> str | None:
        return self._last_regime

    async def start(self) -> None:
        try:
            from apscheduler.schedulers.asyncio import AsyncIOScheduler
            from apscheduler.triggers.interval import IntervalTrigger
        except ImportError:
            logger.warning("apscheduler not installed — regime watcher disabled")
            return

        # Use the scheduler's own "now" so the next-run-time respects the
        # scheduler's timezone (APScheduler defaults to local time; passing
        # a bare utcnow() can land 12+ hours in the past on non-UTC hosts).
        scheduler = AsyncIOScheduler()
        scheduler.add_job(
            self._tick,
            trigger=IntervalTrigger(minutes=self._interval),
            id="regime_watcher",
            next_run_time=datetime.now() + timedelta(seconds=5),
            replace_existing=True,
        )
        scheduler.start()
        self._scheduler = scheduler
        logger.info(
            "Regime watcher started: every %.1fm on %s %s",
            self._interval, self._symbol, self._timeframe,
        )

    async def stop(self) -> None:
        if self._scheduler is not None:
            try:
                self._scheduler.shutdown(wait=False)
            except Exception:  # noqa: BLE001
                pass
            self._scheduler = None

    async def _tick(self) -> None:
        """One refresh. Forces a cache miss so the HMM re-fits on fresh data."""
        try:
            from .services_regime import get_current_regime

            snapshot = await get_current_regime(
                symbol_str=self._symbol,
                timeframe_str=self._timeframe,
                force_refresh=True,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("regime watcher tick failed: %s", exc)
            return

        self._last_run = datetime.utcnow()
        if snapshot.status == "ok":
            self._last_regime = snapshot.regime
        # The regime service itself fires the alert on state change.
