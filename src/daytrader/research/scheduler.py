"""Background scheduler for the Exploration Agent.

Wraps ``apscheduler``'s AsyncIOScheduler. When enabled, runs one full
``ExplorationAgent.scan()`` every ``interval_hours`` for each configured
target. Results land in the ``discoveries`` table and surface in the
Research Lab → Discoveries tab without user intervention.

Disabled by default (``EXPLORATION_SCHEDULE_HOURS=0``). Enable via:

    EXPLORATION_SCHEDULE_HOURS=6
    EXPLORATION_SCHEDULE_SYMBOLS=BTC-USD,ETH-USD
    EXPLORATION_SCHEDULE_TIMEFRAME=1d
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

logger = logging.getLogger(__name__)


class ExplorationScheduler:
    """Periodic background runner for the Exploration Agent."""

    def __init__(
        self,
        *,
        tenant_id: UUID,
        interval_hours: float,
        symbols: list[str],
        timeframe: str = "1d",
        lookback_days: int = 365,
        include_fred: bool = True,
        sentiment_queries: list[str] | None = None,
        task: str = "classification",
    ) -> None:
        self._tenant_id = tenant_id
        self._interval_hours = max(0.0, float(interval_hours))
        self._symbols = [s.strip() for s in symbols if s.strip()]
        self._timeframe = timeframe
        self._lookback_days = lookback_days
        self._include_fred = include_fred
        self._sentiment_queries = sentiment_queries or []
        self._task = task
        self._scheduler: Any = None
        self._last_run: datetime | None = None
        self._last_summary: dict[str, Any] = {}

    @property
    def is_enabled(self) -> bool:
        return self._interval_hours > 0 and bool(self._symbols)

    @property
    def last_run(self) -> datetime | None:
        return self._last_run

    @property
    def last_summary(self) -> dict[str, Any]:
        return dict(self._last_summary)

    async def start(self) -> None:
        """Start the scheduler. No-op if disabled."""
        if not self.is_enabled:
            logger.info("Exploration scheduler disabled (interval_hours=0 or no symbols)")
            return
        try:
            from apscheduler.schedulers.asyncio import AsyncIOScheduler
            from apscheduler.triggers.interval import IntervalTrigger
        except ImportError:
            logger.warning("apscheduler not installed — scheduler disabled")
            return

        # Local time — APScheduler defaults to local tz, so utcnow() here
        # would schedule the first run 12+ hours in the past on non-UTC hosts.
        scheduler = AsyncIOScheduler()
        scheduler.add_job(
            self._run_once,
            trigger=IntervalTrigger(hours=self._interval_hours),
            id="exploration_scan",
            next_run_time=datetime.now() + timedelta(minutes=1),
            replace_existing=True,
        )
        scheduler.start()
        self._scheduler = scheduler
        logger.info(
            "Exploration scheduler started: every %.2fh, symbols=%s",
            self._interval_hours, self._symbols,
        )

    async def stop(self) -> None:
        if self._scheduler is not None:
            try:
                self._scheduler.shutdown(wait=False)
            except Exception:  # noqa: BLE001
                pass
            self._scheduler = None

    async def run_now(self) -> dict[str, Any]:
        """Trigger a scan immediately (outside the schedule). Returns summary."""
        return await self._run_once()

    async def _run_once(self) -> dict[str, Any]:
        """Run one scan per configured symbol and summarize."""
        from .exploration_agent import ExplorationAgent, ExplorationConfig

        agent = ExplorationAgent(ExplorationConfig(
            include_fred=self._include_fred,
            sentiment_queries=self._sentiment_queries,
            task=self._task,
        ))
        end = datetime.utcnow()
        start = end - timedelta(days=self._lookback_days)

        summary: dict[str, Any] = {
            "ran_at": end.isoformat(),
            "symbols": [],
            "total_candidates": 0,
            "total_significant": 0,
            "errors": [],
        }
        for symbol in self._symbols:
            try:
                result = await agent.scan(
                    tenant_id=self._tenant_id,
                    symbol_str=symbol,
                    timeframe_str=self._timeframe,
                    start=start,
                    end=end,
                )
            except Exception as exc:  # noqa: BLE001
                summary["errors"].append({"symbol": symbol, "error": str(exc)})
                logger.exception("Exploration scan failed for %s", symbol)
                continue
            summary["symbols"].append({
                "symbol": symbol,
                "n_bars": result.n_bars,
                "n_candidates": result.n_candidates,
                "n_significant": result.n_significant,
            })
            summary["total_candidates"] += result.n_candidates
            summary["total_significant"] += result.n_significant

        self._last_run = end
        self._last_summary = summary
        logger.info(
            "Exploration scheduler run complete: %d symbols, %d significant",
            len(summary["symbols"]), summary["total_significant"],
        )
        return summary
