"""Notifier ABC — the only contract callers should depend on."""

from __future__ import annotations

from abc import ABC, abstractmethod
from uuid import UUID


class Notifier(ABC):
    """Send a human-readable alert to a tenant.

    Implementations must be safe to call from any async context and must
    never raise — operational visibility is nice to have, but a failed
    notification cannot abort the caller's primary work.
    """

    @abstractmethod
    async def notify(
        self,
        tenant_id: UUID,
        message: str,
        *,
        dedupe_key: str | None = None,
    ) -> None:
        """Deliver ``message`` to the tenant's configured channel.

        ``dedupe_key`` is a hint to throttling layers; identical keys
        within a short window collapse to a single delivery. Pass a
        stable identifier (e.g. ``f"plugin_error:{algo_id}"``) so a
        flapping plugin does not flood the channel.
        """
