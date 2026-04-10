"""Market data events."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from ..types.bars import Bar
from .base import Event, EventType


@dataclass(frozen=True, slots=True, kw_only=True)
class PriceTickEvent(Event):
    type: EventType = EventType.PRICE_TICK
    symbol_key: str
    price: Decimal


@dataclass(frozen=True, slots=True, kw_only=True)
class BarClosedEvent(Event):
    type: EventType = EventType.BAR_CLOSED
    symbol_key: str
    timeframe: str
    bar: Bar
