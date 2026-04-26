"""Paper trading executor — in-memory, instant fills, no real money.

Orders fill immediately at the requested price (no slippage in Phase 0).
Positions and cash balances are tracked in-memory per persona.

Phase 1 will add: slippage model, persistent state (DB-backed),
scheduled bar processing, and real-time P&L updates to the UI.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from decimal import Decimal
from uuid import UUID

from ..core.types.orders import Order, OrderSide, OrderStatus
from .base import ExecutionAdapter


@dataclass
class _PaperState:
    """Per-persona paper trading state."""

    cash: Decimal
    positions: dict[str, Decimal] = field(default_factory=dict)
    filled_orders: list[Order] = field(default_factory=list)


class PaperExecutor(ExecutionAdapter):
    """In-memory paper trading. Fills instantly at the requested price."""

    def __init__(self) -> None:
        self._states: dict[UUID, _PaperState] = {}

    @property
    def name(self) -> str:
        return "paper"

    def initialize_persona(self, persona_id: UUID, capital: Decimal) -> None:
        """Set up a persona's paper trading state. Call before submitting orders."""
        self._states[persona_id] = _PaperState(cash=capital)

    async def submit_order(self, order: Order) -> Order:
        state = self._states.get(order.persona_id)
        if state is None:
            return replace(
                order,
                status=OrderStatus.REJECTED,
                reason="Persona not initialized for paper trading",
            )

        price = order.price or Decimal(0)
        if price <= 0:
            return replace(
                order, status=OrderStatus.REJECTED, reason="Price must be > 0"
            )

        if order.side == OrderSide.BUY:
            cost = order.quantity * price
            if cost > state.cash:
                return replace(
                    order,
                    status=OrderStatus.REJECTED,
                    reason=f"Insufficient funds: need {cost}, have {state.cash}",
                )
            state.cash -= cost
            current = state.positions.get(order.symbol_key, Decimal(0))
            state.positions[order.symbol_key] = current + order.quantity

        elif order.side == OrderSide.SELL:
            current = state.positions.get(order.symbol_key, Decimal(0))
            if order.quantity > current:
                return replace(
                    order,
                    status=OrderStatus.REJECTED,
                    reason=f"Insufficient position: need {order.quantity}, have {current}",
                )
            revenue = order.quantity * price
            state.cash += revenue
            state.positions[order.symbol_key] = current - order.quantity

        filled = replace(
            order,
            status=OrderStatus.FILLED,
            filled_quantity=order.quantity,
            avg_fill_price=price,
        )
        state.filled_orders.append(filled)
        return filled

    async def cancel_order(self, order_id: UUID) -> bool:
        return False  # Instant fills — nothing to cancel.

    async def get_positions(self, persona_id: UUID) -> dict[str, Decimal]:
        state = self._states.get(persona_id)
        if state is None:
            return {}
        return {k: v for k, v in state.positions.items() if v != 0}

    async def get_balance(self, persona_id: UUID) -> Decimal:
        state = self._states.get(persona_id)
        return state.cash if state else Decimal(0)

    def get_order_history(self, persona_id: UUID) -> list[Order]:
        state = self._states.get(persona_id)
        return list(state.filled_orders) if state else []
