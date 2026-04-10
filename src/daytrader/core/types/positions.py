"""Position value type."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import UUID


@dataclass(frozen=True, slots=True)
class Position:
    """An open or closed position in a symbol.

    ``quantity`` is signed: positive = long, negative = short, zero = flat.
    """

    id: UUID
    persona_id: UUID
    symbol_key: str
    quantity: Decimal
    avg_entry_price: Decimal
    opened_at: datetime
    realized_pnl: Decimal = Decimal(0)
    unrealized_pnl: Decimal = Decimal(0)
    stop_loss: Decimal | None = None
    take_profit: Decimal | None = None
    closed_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_long(self) -> bool:
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        return self.quantity < 0

    @property
    def is_flat(self) -> bool:
        return self.quantity == 0

    @property
    def is_closed(self) -> bool:
        return self.closed_at is not None
