"""Per-tenant supervisors for research schedulers.

The Exploration Agent and Shadow Tournament schedulers were originally
constructed as singletons bound to ``settings.default_tenant_id``. With
multi-tenant auth, each tenant should get its own scheduler — but ONLY
when the global feature is enabled and the tenant has the
``background_workers_enabled`` flag set. The cadence and parameters
remain global env-driven (per-tenant overrides are out of scope).
"""

from __future__ import annotations

import logging
from uuid import UUID

from sqlalchemy import select

from ..core.settings import get_settings
from ..core.supervisor import BackgroundSupervisor
from ..storage.database import get_session
from ..storage.models import TenantModel
from .scheduler import ExplorationScheduler
from .shadow_scheduler import ShadowScheduler

logger = logging.getLogger(__name__)


async def _enabled_tenants() -> set[UUID]:
    """All tenants with background_workers_enabled=True."""
    async with get_session() as session:
        rows = (
            await session.execute(
                select(TenantModel.id).where(
                    TenantModel.background_workers_enabled.is_(True)
                )
            )
        ).scalars().all()
        return set(rows)


class ExplorationSupervisor(BackgroundSupervisor):
    """One ``ExplorationScheduler`` per tenant when the global cadence is on."""

    name = "exploration-supervisor"

    async def _active_tenants(self) -> set[UUID]:
        s = get_settings()
        if s.exploration_schedule_hours <= 0:
            return set()
        return await _enabled_tenants()

    def _make_worker(self, tenant_id: UUID) -> ExplorationScheduler:
        s = get_settings()
        symbols = [
            sym.strip() for sym in s.exploration_schedule_symbols.split(",")
            if sym.strip()
        ]
        return ExplorationScheduler(
            tenant_id=tenant_id,
            interval_hours=s.exploration_schedule_hours,
            symbols=symbols,
            timeframe=s.exploration_schedule_timeframe,
            lookback_days=s.exploration_schedule_lookback_days,
        )


class ShadowSupervisor(BackgroundSupervisor):
    """One ``ShadowScheduler`` per tenant when the global cadence is on."""

    name = "shadow-supervisor"

    async def _active_tenants(self) -> set[UUID]:
        s = get_settings()
        if s.shadow_schedule_hours <= 0:
            return set()
        return await _enabled_tenants()

    def _make_worker(self, tenant_id: UUID) -> ShadowScheduler:
        s = get_settings()
        candidates = [
            c.strip() for c in s.shadow_schedule_candidates.split(",") if c.strip()
        ]
        return ShadowScheduler(
            tenant_id=tenant_id,
            interval_hours=s.shadow_schedule_hours,
            primary_algo_id=s.shadow_schedule_primary,
            candidate_algo_ids=candidates,
            symbol=s.shadow_schedule_symbol,
            timeframe=s.shadow_schedule_timeframe,
            lookback_days=s.shadow_schedule_lookback_days,
        )
