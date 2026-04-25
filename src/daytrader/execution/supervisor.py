"""TradingLoopSupervisor — one ``TradingLoop`` per active tenant.

A tenant is "active" when it has the background-workers flag set AND owns
at least one persona in paper or live mode. New personas spin up a loop
within a poll cycle (default 60s); the last persona being deleted/paused
spins it back down.
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from sqlalchemy import select

from ..core.supervisor import BackgroundSupervisor
from ..storage.database import get_session
from ..storage.models import PersonaModel, TenantModel
from .loop import TradingLoop

logger = logging.getLogger(__name__)


class TradingLoopSupervisor(BackgroundSupervisor):
    """One ``TradingLoop`` per tenant with active personas."""

    name = "trading-loop-supervisor"

    def __init__(
        self,
        *,
        journal: Any,
        kill_switch: Any,
        global_risk: Any,
        poll_seconds: float = 30.0,
        supervisor_poll_seconds: float = 60.0,
    ) -> None:
        super().__init__(supervisor_poll_seconds=supervisor_poll_seconds)
        self._journal = journal
        self._kill_switch = kill_switch
        self._global_risk = global_risk
        self._loop_poll_seconds = poll_seconds

    async def _active_tenants(self) -> set[UUID]:
        async with get_session() as session:
            stmt = (
                select(PersonaModel.tenant_id)
                .join(TenantModel, TenantModel.id == PersonaModel.tenant_id)
                .where(PersonaModel.mode.in_(("paper", "live")))
                .where(TenantModel.background_workers_enabled.is_(True))
                .distinct()
            )
            rows = (await session.execute(stmt)).scalars().all()
            return set(rows)

    def _make_worker(self, tenant_id: UUID) -> TradingLoop:
        return TradingLoop(
            journal=self._journal,
            kill_switch=self._kill_switch,
            global_risk=self._global_risk,
            poll_seconds=self._loop_poll_seconds,
            tenant_id=tenant_id,
        )
