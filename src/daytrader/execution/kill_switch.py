"""Global emergency kill switch.

When activated, all live and paper personas are set to PAUSED and the
trading loop stops processing.  The event is recorded in the journal.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING
from uuid import UUID

from ..core.context import tenant_scope
from ..storage.database import get_session
from ..storage.models import PersonaModel
from ..storage.repository import TenantRepository

if TYPE_CHECKING:
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

    def __init__(self, journal: JournalWriter | None = None) -> None:
        self._activated = asyncio.Event()
        self._journal = journal

    @property
    def is_activated(self) -> bool:
        return self._activated.is_set()

    async def activate(self, tenant_id: UUID, reason: str = "manual") -> int:
        """Activate the kill switch.

        Sets all live/paper personas to ``paused`` and journals the event.
        Returns the number of personas paused.
        """
        self._activated.set()
        count = await self._pause_all_personas(tenant_id)

        logger.warning(
            "Kill switch activated (reason=%s): %d personas paused",
            reason,
            count,
        )

        if self._journal:
            await self._journal.log_kill_switch(tenant_id, reason)

        return count

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

    def reset(self) -> None:
        """Clear the kill switch (allow trading to resume)."""
        self._activated.clear()
        logger.info("Kill switch reset")


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_kill_switch: KillSwitch | None = None


def init_kill_switch(journal: JournalWriter | None = None) -> KillSwitch:
    """Create the global kill switch singleton (called at app startup)."""
    global _kill_switch
    _kill_switch = KillSwitch(journal=journal)
    return _kill_switch


def get_kill_switch() -> KillSwitch:
    """Return the global kill switch. Raises if not initialised."""
    if _kill_switch is None:
        raise RuntimeError("Kill switch not initialised. Call init_kill_switch() first.")
    return _kill_switch
