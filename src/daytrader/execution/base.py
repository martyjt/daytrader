"""ExecutionAdapter ABC — same interface for paper and live trading.

A persona can be promoted from paper to live by swapping the adapter
instance. The rest of the system doesn't change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from decimal import Decimal
from uuid import UUID

from ..core.types.orders import Order


class ExecutionAdapter(ABC):
    """Abstract base for execution venues (paper, Binance, Alpaca, ...)."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Venue identifier, e.g. ``'paper'``, ``'binance'``."""

    @abstractmethod
    async def submit_order(self, order: Order) -> Order:
        """Submit an order. Returns the order with updated fill info/status."""

    @abstractmethod
    async def cancel_order(self, order_id: UUID) -> bool:
        """Cancel a pending order. Returns True if cancelled."""

    @abstractmethod
    async def get_positions(self, persona_id: UUID) -> dict[str, Decimal]:
        """Return current positions: ``{symbol_key: quantity}``."""

    @abstractmethod
    async def get_balance(self, persona_id: UUID) -> Decimal:
        """Return available cash balance."""
