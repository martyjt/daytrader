"""Order lifecycle events."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from uuid import UUID

from ..types.orders import Order
from .base import Event, EventType


@dataclass(frozen=True, slots=True, kw_only=True)
class OrderSubmittedEvent(Event):
    type: EventType = EventType.ORDER_SUBMITTED
    persona_id: UUID
    order: Order


@dataclass(frozen=True, slots=True, kw_only=True)
class OrderFilledEvent(Event):
    type: EventType = EventType.ORDER_FILLED
    persona_id: UUID
    order_id: UUID
    fill_price: Decimal
    fill_quantity: Decimal
