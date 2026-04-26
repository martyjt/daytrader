"""In-memory alert center for regime changes, drift, discoveries, etc.

Surfaces a small badge in the shell header showing the unread count and a
popover listing recent alerts. Alerts are in-process only — intentionally
lightweight so background monitors (regime, drift, correlation) can fire
as often as they like without DB churn.

If an alert needs permanent history, the producer also writes a
``JournalEntryModel`` with a matching ``event_type``. The header badge
reflects the in-process stream; the Journal page reflects the durable
record.
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal
from uuid import UUID, uuid4

AlertLevel = Literal["info", "warning", "critical"]


@dataclass(frozen=True)
class Alert:
    """One alert card for the dropdown."""

    id: UUID
    created_at: datetime
    level: AlertLevel
    title: str
    body: str = ""
    source: str = ""  # "regime" | "drift" | "discovery" | "correlation" | "kill_switch"
    data: dict = field(default_factory=dict)


class AlertCenter:
    """Process-global ring buffer of alerts + per-client read state."""

    _instance: AlertCenter | None = None
    _lock = asyncio.Lock()

    def __init__(self, capacity: int = 100) -> None:
        self._capacity = capacity
        self._alerts: deque[Alert] = deque(maxlen=capacity)
        self._unread_ids: set[UUID] = set()
        self._subscribers: list[asyncio.Event] = []

    @classmethod
    def instance(cls) -> AlertCenter:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def add(
        self,
        *,
        level: AlertLevel,
        title: str,
        body: str = "",
        source: str = "",
        data: dict | None = None,
    ) -> Alert:
        """Record a new alert. Safe to call from any thread / task."""
        alert = Alert(
            id=uuid4(),
            created_at=datetime.utcnow(),
            level=level,
            title=title,
            body=body,
            source=source,
            data=dict(data or {}),
        )
        self._alerts.appendleft(alert)
        self._unread_ids.add(alert.id)
        # Notify any UI listeners (they re-poll `unread_count` on their timer).
        for ev in list(self._subscribers):
            ev.set()
        return alert

    def list(self, limit: int = 50) -> list[Alert]:
        return list(self._alerts)[:limit]

    def unread_count(self) -> int:
        return len(self._unread_ids)

    def mark_all_read(self) -> int:
        n = len(self._unread_ids)
        self._unread_ids.clear()
        return n

    def mark_read(self, alert_id: UUID) -> None:
        self._unread_ids.discard(alert_id)

    def clear(self) -> None:
        self._alerts.clear()
        self._unread_ids.clear()


def alerts() -> AlertCenter:
    """Module-level shortcut: ``from .alerts import alerts; alerts().add(...)``."""
    return AlertCenter.instance()
