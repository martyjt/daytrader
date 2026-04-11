"""JournalWriter — async service that persists trading events.

Usage::

    writer = JournalWriter()
    await writer.log_order_filled(tenant_id, persona_id, order)
    await writer.log_kill_switch(tenant_id, "manual")
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from ..core.context import tenant_scope
from ..core.events.base import EventType
from ..core.types.orders import Order
from ..core.types.signals import Signal
from ..storage.database import get_session
from ..storage.models import JournalEntryModel
from ..storage.repository import TenantRepository

logger = logging.getLogger(__name__)


class JournalWriter:
    """Persists structured trading events to the journal_entries table."""

    async def log(
        self,
        tenant_id: UUID,
        event_type: str,
        summary: str,
        *,
        persona_id: UUID | None = None,
        severity: str = "info",
        detail: dict[str, Any] | None = None,
    ) -> None:
        """Write a single journal entry."""
        try:
            async with get_session() as session:
                with tenant_scope(tenant_id):
                    repo = TenantRepository(session, JournalEntryModel)
                    await repo.create(
                        persona_id=persona_id,
                        event_type=event_type,
                        severity=severity,
                        summary=summary,
                        detail=detail or {},
                    )
                    await session.commit()
        except Exception:
            logger.exception("Failed to write journal entry: %s", summary)

    # ---- Convenience methods ------------------------------------------------

    async def log_order_submitted(
        self, tenant_id: UUID, persona_id: UUID, order: Order
    ) -> None:
        await self.log(
            tenant_id,
            EventType.ORDER_SUBMITTED,
            f"{order.side.value.upper()} {order.quantity} {order.symbol_key} @ {order.price}",
            persona_id=persona_id,
            detail=_order_detail(order),
        )

    async def log_order_filled(
        self, tenant_id: UUID, persona_id: UUID, order: Order
    ) -> None:
        await self.log(
            tenant_id,
            EventType.ORDER_FILLED,
            f"Filled {order.side.value.upper()} {order.filled_quantity} "
            f"{order.symbol_key} @ {order.avg_fill_price}",
            persona_id=persona_id,
            detail=_order_detail(order),
        )

    async def log_order_cancelled(
        self, tenant_id: UUID, persona_id: UUID, order: Order
    ) -> None:
        await self.log(
            tenant_id,
            EventType.ORDER_CANCELLED,
            f"Cancelled {order.side.value.upper()} {order.symbol_key}",
            persona_id=persona_id,
            detail=_order_detail(order),
        )

    async def log_signal_emitted(
        self,
        tenant_id: UUID,
        persona_id: UUID,
        signal: Signal,
    ) -> None:
        direction = "LONG" if signal.score > 0 else "SHORT" if signal.score < 0 else "FLAT"
        await self.log(
            tenant_id,
            EventType.SIGNAL_EMITTED,
            f"{direction} signal on {signal.symbol_key} "
            f"(score={signal.score:.2f}, conf={signal.confidence:.2f})",
            persona_id=persona_id,
            detail={
                "signal_id": str(signal.id),
                "symbol_key": signal.symbol_key,
                "score": signal.score,
                "confidence": signal.confidence,
                "source": signal.source,
                "reason": signal.reason,
            },
        )

    async def log_risk_breach(
        self,
        tenant_id: UUID,
        persona_id: UUID | None,
        breach_type: str,
        details: dict[str, Any],
    ) -> None:
        await self.log(
            tenant_id,
            EventType.RISK_BREACH,
            f"Risk breach: {breach_type}",
            persona_id=persona_id,
            severity="critical",
            detail={"breach_type": breach_type, **details},
        )

    async def log_kill_switch(
        self, tenant_id: UUID, reason: str = "manual"
    ) -> None:
        await self.log(
            tenant_id,
            EventType.KILL_SWITCH,
            f"Kill switch activated: {reason}",
            severity="critical",
            detail={"reason": reason},
        )

    async def log_mode_change(
        self,
        tenant_id: UUID,
        persona_id: UUID,
        old_mode: str,
        new_mode: str,
    ) -> None:
        await self.log(
            tenant_id,
            EventType.MODE_CHANGE,
            f"Mode changed: {old_mode} → {new_mode}",
            persona_id=persona_id,
            detail={"old_mode": old_mode, "new_mode": new_mode},
        )


def _order_detail(order: Order) -> dict[str, Any]:
    """Extract a JSON-safe detail dict from an Order."""
    return {
        "order_id": str(order.id),
        "symbol_key": order.symbol_key,
        "side": order.side.value,
        "type": order.type.value,
        "quantity": str(order.quantity),
        "price": str(order.price) if order.price else None,
        "status": order.status.value,
        "filled_quantity": str(order.filled_quantity),
        "avg_fill_price": str(order.avg_fill_price) if order.avg_fill_price else None,
        "reason": order.reason,
    }
