"""Signal emission events."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from ..types.signals import Signal
from .base import Event, EventType


@dataclass(frozen=True, slots=True, kw_only=True)
class SignalEmittedEvent(Event):
    type: EventType = EventType.SIGNAL_EMITTED
    persona_id: UUID
    signal: Signal
