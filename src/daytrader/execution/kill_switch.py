"""Global emergency kill switch.

When activated, all live and paper personas are set to PAUSED, the
trading loop stops processing, and the tenant's plugin worker
subprocesses are torn down. The event is recorded in the journal and
audit log.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING
from uuid import UUID

from ..core import audit
from ..core.context import tenant_scope
from ..storage.database import get_session
from ..storage.models import PersonaModel
from ..storage.repository import TenantRepository

if TYPE_CHECKING:
    from ..algorithms.sandbox import PluginWorkerManager
    from ..journal.writer import JournalWriter

logger = logging.getLogger(__name__)


class KillSwitch:
    """Global trading halt.

    Usage::

        ks = KillSwitch()
        count = await ks.activate(tenant_id, reason="manual")
        assert ks.is_activated
        ks.reset()
    """

    def __init__(
        self,
        journal: JournalWriter | None = None,
        plugin_manager: PluginWorkerManager | None = None,
    ) -> None:
        self._activated = asyncio.Event()
        self._journal = journal
        self._plugin_manager = plugin_manager

    @property
    def is_activated(self) -> bool:
        return self._activated.is_set()

    async def activate(self, tenant_id: UUID, reason: str = "manual") -> int:
        """Activate the kill switch.

        Sets all live/paper personas to ``paused``, tears down the
        tenant's plugin worker subprocess (if any), and journals the
        event. Returns the number of personas paused.
        """
        self._activated.set()
        count = await self._pause_all_personas(tenant_id)
        plugins_killed = await self._shutdown_plugins(tenant_id)

        logger.warning(
            "Kill switch activated (reason=%s): %d personas paused, plugins_killed=%s",
            reason,
            count,
            plugins_killed,
        )

        if self._journal:
            await self._journal.log_kill_switch(tenant_id, reason)

        await audit.record(
            "kill_switch.activate",
            tenant_id=tenant_id,
            extra={
                "reason": reason,
                "personas_paused": count,
                "plugins_killed": plugins_killed,
            },
        )

        return count

    async def kill_plugins(self, tenant_id: UUID, reason: str = "manual") -> bool:
        """Tear down a single tenant's plugin worker without touching personas.

        Returns ``True`` if a worker was running and got killed.
        Used by the per-tenant ``Kill plugins`` button on /admin/tenants
        so a super admin can stop a misbehaving plugin without halting
        that tenant's live/paper trading.
        """
        killed = await self._shutdown_plugins(tenant_id)
        logger.warning(
            "Plugins killed for tenant %s (reason=%s, had_worker=%s)",
            tenant_id,
            reason,
            killed,
        )
        await audit.record(
            "kill_switch.plugins",
            tenant_id=tenant_id,
            extra={"reason": reason, "had_worker": killed},
        )
        return killed

    async def _pause_all_personas(self, tenant_id: UUID) -> int:
        """Set all active personas to paused."""
        paused = 0
        async with get_session() as session:
            with tenant_scope(tenant_id):
                repo = TenantRepository(session, PersonaModel)
                for mode in ("live", "paper"):
                    personas = await repo.get_all(mode=mode)
                    for persona in personas:
                        persona.mode = "paused"
                        paused += 1
                await session.commit()
        return paused

    async def _shutdown_plugins(self, tenant_id: UUID) -> bool:
        """Tear down the tenant's plugin worker subprocess if one is running."""
        manager = self._plugin_manager
        if manager is None or not manager.has_handle(tenant_id):
            return False
        try:
            await manager.shutdown_tenant(tenant_id)
        except Exception as exc:
            logger.warning(
                "Plugin worker shutdown failed for tenant %s: %s", tenant_id, exc
            )
            return False
        return True

    def reset(self) -> None:
        """Clear the kill switch (allow trading to resume)."""
        self._activated.clear()
        logger.info("Kill switch reset")


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_kill_switch: KillSwitch | None = None


def init_kill_switch(
    journal: JournalWriter | None = None,
    plugin_manager: PluginWorkerManager | None = None,
) -> KillSwitch:
    """Create the global kill switch singleton (called at app startup)."""
    global _kill_switch
    _kill_switch = KillSwitch(journal=journal, plugin_manager=plugin_manager)
    return _kill_switch


def get_kill_switch() -> KillSwitch:
    """Return the global kill switch. Raises if not initialised."""
    if _kill_switch is None:
        raise RuntimeError("Kill switch not initialised. Call init_kill_switch() first.")
    return _kill_switch
