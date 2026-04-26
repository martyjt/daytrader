"""NoopNotifier — drops every notification on the floor."""

from __future__ import annotations

from uuid import UUID

from .base import Notifier


class NoopNotifier(Notifier):
    """Discards notifications. Used in tests and as a safe default."""

    async def notify(
        self,
        tenant_id: UUID,
        message: str,
        *,
        dedupe_key: str | None = None,
    ) -> None:
        return None
