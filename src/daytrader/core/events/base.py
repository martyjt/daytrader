"""Base event type and registry enum."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from uuid import UUID, uuid4

from ..types.common import utcnow


class EventType(StrEnum):
    PRICE_TICK = "price_tick"
    BAR_CLOSED = "bar_closed"
    SIGNAL_EMITTED = "signal_emitted"
    ORDER_SUBMITTED = "order_submitted"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    RISK_BREACH = "risk_breach"
    SYSTEM = "system"


def _new_id() -> UUID:
    return uuid4()


@dataclass(frozen=True, slots=True, kw_only=True)
class Event:
    """Base event. Concrete events subclass this with ``kw_only=True``."""

    type: EventType
    id: UUID = field(default_factory=_new_id)
    timestamp: datetime = field(default_factory=utcnow)
    tenant_id: UUID | None = None
