"""Per-tenant supervisor for the daily digest worker.

Mirrors the Exploration / Shadow supervisors: one
:class:`DailyDigestWorker` per tenant with
``background_workers_enabled=True``. Off by default at the env layer —
the worker only runs once per day, but a deployment that explicitly
disables digests via ``DAILY_DIGEST_ENABLED=0`` shouldn't spawn idle
tasks.
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import select

from ..core.settings import get_settings
from ..core.supervisor import BackgroundSupervisor
from ..storage.database import get_session
from ..storage.models import TenantModel
from .worker import DailyDigestWorker, parse_local_time


class DailyDigestSupervisor(BackgroundSupervisor):
    """One :class:`DailyDigestWorker` per tenant when digests are enabled."""

    name = "digest-supervisor"

    async def _active_tenants(self) -> set[UUID]:
        s = get_settings()
        if not s.daily_digest_enabled:
            return set()
        async with get_session() as session:
            rows = (
                await session.execute(
                    select(TenantModel.id).where(
                        TenantModel.background_workers_enabled.is_(True)
                    )
                )
            ).scalars().all()
            return set(rows)

    def _make_worker(self, tenant_id: UUID) -> DailyDigestWorker:
        s = get_settings()
        hour, minute = parse_local_time(s.daily_digest_local_time)
        return DailyDigestWorker(
            tenant_id=tenant_id,
            hour_local=hour,
            minute_local=minute,
        )
