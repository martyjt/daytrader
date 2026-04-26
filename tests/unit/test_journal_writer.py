"""Tests for the journal writer service."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from uuid import UUID, uuid4

import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from daytrader.core.events.base import EventType
from daytrader.core.types.orders import Order, OrderSide, OrderStatus, OrderType
from daytrader.storage.models import JournalEntryModel, PersonaModel, TenantModel

TENANT_ID = UUID("00000000-0000-0000-0000-000000000001")
PERSONA_ID = UUID("00000000-0000-0000-0000-000000000002")


@pytest_asyncio.fixture
async def session(engine):
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as s:
        # Seed tenant first so the persona FK insert sees it on Postgres
        # (SQLite's lax FK enforcement was hiding an ordering issue here).
        s.add(TenantModel(id=TENANT_ID, name="test"))
        await s.commit()
        s.add(
            PersonaModel(
                id=PERSONA_ID,
                tenant_id=TENANT_ID,
                name="Test Bot",
                mode="paper",
                initial_capital=Decimal("10000"),
                current_equity=Decimal("10000"),
            )
        )
        await s.commit()
        yield s


@pytest_asyncio.fixture
async def _patch_session(engine, monkeypatch):
    """Patch get_session to use the test engine."""
    factory = async_sessionmaker(engine, expire_on_commit=False)

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _get_session():
        async with factory() as s:
            try:
                yield s
            except Exception:
                await s.rollback()
                raise

    monkeypatch.setattr("daytrader.journal.writer.get_session", _get_session)


async def test_log_writes_entry(session, _patch_session):
    from daytrader.journal.writer import JournalWriter

    writer = JournalWriter()
    await writer.log(
        TENANT_ID,
        EventType.SYSTEM,
        "Test entry",
        severity="info",
        detail={"key": "value"},
    )

    result = await session.execute(
        select(JournalEntryModel).where(
            JournalEntryModel.tenant_id == TENANT_ID
        )
    )
    entries = result.scalars().all()
    assert len(entries) == 1
    assert entries[0].event_type == "system"
    assert entries[0].summary == "Test entry"
    assert entries[0].detail["key"] == "value"


async def test_log_kill_switch(session, _patch_session):
    from daytrader.journal.writer import JournalWriter

    writer = JournalWriter()
    await writer.log_kill_switch(TENANT_ID, "manual")

    result = await session.execute(
        select(JournalEntryModel).where(
            JournalEntryModel.event_type == "kill_switch"
        )
    )
    entry = result.scalars().first()
    assert entry is not None
    assert entry.severity == "critical"
    assert "manual" in entry.summary


async def test_log_mode_change(session, _patch_session):
    from daytrader.journal.writer import JournalWriter

    writer = JournalWriter()
    await writer.log_mode_change(TENANT_ID, PERSONA_ID, "paper", "live")

    result = await session.execute(
        select(JournalEntryModel).where(
            JournalEntryModel.event_type == "mode_change"
        )
    )
    entry = result.scalars().first()
    assert entry is not None
    assert entry.persona_id == PERSONA_ID
    assert entry.detail["old_mode"] == "paper"
    assert entry.detail["new_mode"] == "live"


async def test_log_order_filled(session, _patch_session):
    from daytrader.journal.writer import JournalWriter

    writer = JournalWriter()
    order = Order(
        id=uuid4(),
        persona_id=PERSONA_ID,
        symbol_key="crypto:BTC/USDT",
        side=OrderSide.BUY,
        type=OrderType.MARKET,
        quantity=Decimal("0.5"),
        status=OrderStatus.FILLED,
        created_at=datetime.now(UTC),
        price=Decimal("50000"),
        filled_quantity=Decimal("0.5"),
        avg_fill_price=Decimal("49950"),
    )
    await writer.log_order_filled(TENANT_ID, PERSONA_ID, order)

    result = await session.execute(
        select(JournalEntryModel).where(
            JournalEntryModel.event_type == "order_filled"
        )
    )
    entry = result.scalars().first()
    assert entry is not None
    assert "BUY" in entry.summary
    assert entry.detail["quantity"] == "0.5"


async def test_log_with_null_persona(session, _patch_session):
    """Global events (like kill switch) have no persona_id."""
    from daytrader.journal.writer import JournalWriter

    writer = JournalWriter()
    await writer.log(
        TENANT_ID,
        EventType.SYSTEM,
        "Global event",
        persona_id=None,
    )

    result = await session.execute(
        select(JournalEntryModel).where(
            JournalEntryModel.summary == "Global event"
        )
    )
    entry = result.scalars().first()
    assert entry is not None
    assert entry.persona_id is None
