"""Build a tenant's "what happened yesterday" digest.

Reads from the journal + signal + persona tables that the trading loop
already populates. Pure SQL, no scheduling — the worker calls
:func:`build_digest` and pipes the result into the active notifier.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from uuid import UUID

from sqlalchemy import func, select

from ..core.events.base import EventType
from ..storage.database import get_session
from ..storage.models import (
    JournalEntryModel,
    PersonaModel,
    SignalModel,
)


@dataclass(frozen=True, slots=True)
class DigestSummary:
    """Counts that fit into a one-line Slack post."""

    tenant_id: UUID
    window_start: datetime
    window_end: datetime
    persona_count: int
    active_persona_count: int
    signal_count: int
    fill_count: int
    risk_breach_count: int
    total_equity: Decimal


async def build_digest(
    tenant_id: UUID,
    *,
    window_end: datetime,
    window_hours: int = 24,
) -> DigestSummary:
    """Compute counts for the 24h window ending at ``window_end``.

    The window is exclusive on the left (``> window_start``) and
    inclusive on the right (``<= window_end``) so back-to-back daily
    digests don't double-count a journal entry that lands exactly on
    the boundary.
    """
    window_start = window_end - timedelta(hours=window_hours)

    async with get_session() as session:
        signal_count = await session.scalar(
            select(func.count())
            .select_from(SignalModel)
            .where(
                SignalModel.tenant_id == tenant_id,
                SignalModel.created_at > window_start,
                SignalModel.created_at <= window_end,
            )
        ) or 0

        fill_count = await session.scalar(
            select(func.count())
            .select_from(JournalEntryModel)
            .where(
                JournalEntryModel.tenant_id == tenant_id,
                JournalEntryModel.event_type == EventType.ORDER_FILLED.value,
                JournalEntryModel.created_at > window_start,
                JournalEntryModel.created_at <= window_end,
            )
        ) or 0

        risk_breach_count = await session.scalar(
            select(func.count())
            .select_from(JournalEntryModel)
            .where(
                JournalEntryModel.tenant_id == tenant_id,
                JournalEntryModel.event_type == EventType.RISK_BREACH.value,
                JournalEntryModel.created_at > window_start,
                JournalEntryModel.created_at <= window_end,
            )
        ) or 0

        personas = (
            await session.execute(
                select(PersonaModel).where(
                    PersonaModel.tenant_id == tenant_id
                )
            )
        ).scalars().all()

    persona_count = len(personas)
    active_persona_count = sum(1 for p in personas if p.mode in ("paper", "live"))
    total_equity = sum(
        (Decimal(p.current_equity or 0) for p in personas), Decimal(0)
    )

    return DigestSummary(
        tenant_id=tenant_id,
        window_start=window_start,
        window_end=window_end,
        persona_count=persona_count,
        active_persona_count=active_persona_count,
        signal_count=int(signal_count),
        fill_count=int(fill_count),
        risk_breach_count=int(risk_breach_count),
        total_equity=total_equity,
    )


def format_digest(summary: DigestSummary) -> str:
    """Render a one-paragraph Slack-ready digest from a summary."""
    date_label = summary.window_end.strftime("%Y-%m-%d")
    if summary.persona_count == 0:
        return (
            f"Daytrader digest for {date_label}: no personas yet — "
            f"head to /personas to create one."
        )

    head = (
        f"Daytrader digest for {date_label}: "
        f"{summary.active_persona_count}/{summary.persona_count} personas active, "
        f"equity ${summary.total_equity:,.2f}."
    )
    activity = (
        f" Yesterday: {summary.signal_count} signal(s), "
        f"{summary.fill_count} fill(s)."
    )
    if summary.risk_breach_count:
        activity += f" {summary.risk_breach_count} risk breach(es)."
    if (
        summary.signal_count == 0
        and summary.fill_count == 0
        and summary.risk_breach_count == 0
    ):
        activity = " Quiet day — no trades or signals."
    return head + activity
