"""Background scheduler for Shadow Tournaments.

Periodically runs one tournament per configured primary/candidates pair
so winners surface without user intervention. Settings:

    SHADOW_SCHEDULE_HOURS=24                     # 0 = disabled
    SHADOW_SCHEDULE_PRIMARY=ema_crossover
    SHADOW_SCHEDULE_CANDIDATES=macd_signal,rsi_mean_reversion,ichimoku_cloud
    SHADOW_SCHEDULE_SYMBOL=BTC-USD
    SHADOW_SCHEDULE_TIMEFRAME=1d
    SHADOW_SCHEDULE_LOOKBACK_DAYS=180

Disabled by default — heavy compute, so we never fire it without the
user turning it on.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

logger = logging.getLogger(__name__)


class ShadowScheduler:
    """Periodic background runner for Shadow Tournaments."""

    def __init__(
        self,
        *,
        tenant_id: UUID,
        interval_hours: float,
        primary_algo_id: str,
        candidate_algo_ids: list[str],
        symbol: str,
        timeframe: str = "1d",
        lookback_days: int = 180,
        n_folds: int = 5,
    ) -> None:
        self._tenant_id = tenant_id
        self._interval_hours = max(0.0, float(interval_hours))
        self._primary = primary_algo_id
        self._candidates = [c for c in candidate_algo_ids if c and c != primary_algo_id]
        self._symbol = symbol
        self._timeframe = timeframe
        self._lookback_days = lookback_days
        self._n_folds = n_folds
        self._scheduler: Any = None
        self._last_run: datetime | None = None

    @property
    def is_enabled(self) -> bool:
        return (
            self._interval_hours > 0
            and bool(self._primary)
            and bool(self._candidates)
            and bool(self._symbol)
        )

    @property
    def last_run(self) -> datetime | None:
        return self._last_run

    async def start(self) -> None:
        if not self.is_enabled:
            logger.info("Shadow scheduler disabled")
            return
        try:
            from apscheduler.schedulers.asyncio import AsyncIOScheduler
            from apscheduler.triggers.interval import IntervalTrigger
        except ImportError:
            logger.warning("apscheduler not installed — shadow scheduler disabled")
            return

        scheduler = AsyncIOScheduler()
        scheduler.add_job(
            self._tick,
            trigger=IntervalTrigger(hours=self._interval_hours),
            id="shadow_tournament",
            next_run_time=datetime.now() + timedelta(minutes=2),
            replace_existing=True,
        )
        scheduler.start()
        self._scheduler = scheduler
        logger.info(
            "Shadow scheduler started: every %.2fh, primary=%s, "
            "candidates=%s, symbol=%s",
            self._interval_hours, self._primary, self._candidates, self._symbol,
        )

    async def stop(self) -> None:
        if self._scheduler is not None:
            try:
                self._scheduler.shutdown(wait=False)
            except Exception:  # noqa: BLE001
                pass
            self._scheduler = None

    async def run_now(self) -> Any:
        return await self._tick()

    async def _tick(self) -> Any:
        from .shadow_trading import run_shadow_tournament

        end = datetime.utcnow()
        start = end - timedelta(days=self._lookback_days)
        try:
            report = await run_shadow_tournament(
                tenant_id=self._tenant_id,
                primary_algo_id=self._primary,
                candidate_algo_ids=self._candidates,
                symbol_str=self._symbol,
                timeframe_str=self._timeframe,
                start=start,
                end=end,
                n_folds=self._n_folds,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Shadow tick failed: %s", exc)
            return None
        self._last_run = datetime.utcnow()
        return report
