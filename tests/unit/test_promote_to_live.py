"""Tests for the promote-to-live service and journal list."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker

from daytrader.core.context import tenant_scope
from daytrader.storage.models import (
    JournalEntryModel,
    PersonaModel,
    TenantModel,
)
from daytrader.storage.repository import TenantRepository

TENANT_ID = UUID("00000000-0000-0000-0000-000000000001")


@pytest_asyncio.fixture
async def session(engine):
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as s:
        s.add(TenantModel(id=TENANT_ID, name="test"))
        await s.commit()
        yield s


@pytest_asyncio.fixture
async def _patch_deps(engine, monkeypatch):
    """Patch get_session and _tenant_id for services."""
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

    monkeypatch.setattr("daytrader.ui.services.get_session", _get_session)
    monkeypatch.setattr("daytrader.journal.writer.get_session", _get_session)
    monkeypatch.setattr(
        "daytrader.ui.services._tenant_id", lambda: TENANT_ID
    )


async def _create_persona(session, mode="paper", days_ago=30) -> UUID:
    pid = uuid4()
    created = datetime.now(UTC) - timedelta(days=days_ago)
    with tenant_scope(TENANT_ID):
        repo = TenantRepository(session, PersonaModel)
        await repo.create(
            id=pid,
            name="Test Bot",
            mode=mode,
            asset_class="crypto",
            base_currency="USDT",
            initial_capital=Decimal("10000"),
            current_equity=Decimal("11000"),
            created_at=created,
        )
        await session.commit()
    return pid


async def _seed_journal_trades(session, persona_id: UUID, count: int) -> None:
    """Seed fake order_filled entries in the journal."""
    with tenant_scope(TENANT_ID):
        repo = TenantRepository(session, JournalEntryModel)
        for _ in range(count):
            await repo.create(
                persona_id=persona_id,
                event_type="order_filled",
                severity="info",
                summary="Test fill",
                detail={},
            )
        await session.commit()


# ---------------------------------------------------------------------------
# promote_to_live
# ---------------------------------------------------------------------------


async def test_promote_passes_gates(session, _patch_deps):
    pid = await _create_persona(session, days_ago=30)
    await _seed_journal_trades(session, pid, count=15)

    from daytrader.ui.services import promote_to_live

    persona, gate_result = await promote_to_live(pid, venue="binance")

    assert gate_result.overall_pass is True
    assert persona.mode == "live"
    meta = persona.meta or {}
    assert meta.get("venue") == "binance"


async def test_promote_fails_gates_insufficient_trades(session, _patch_deps):
    pid = await _create_persona(session, days_ago=30)
    await _seed_journal_trades(session, pid, count=3)  # Below min_trades=10

    from daytrader.ui.services import promote_to_live

    persona, gate_result = await promote_to_live(pid, venue="binance")

    assert gate_result.overall_pass is False
    assert persona.mode == "paper"  # Not promoted


async def test_promote_fails_gates_insufficient_days(session, _patch_deps):
    pid = await _create_persona(session, days_ago=2)  # Below min_days=7
    await _seed_journal_trades(session, pid, count=20)

    from daytrader.ui.services import promote_to_live

    _persona, gate_result = await promote_to_live(pid, venue="binance")

    assert gate_result.overall_pass is False


async def test_promote_wrong_mode_raises(session, _patch_deps):
    pid = await _create_persona(session, mode="backtest")

    from daytrader.ui.services import promote_to_live

    with pytest.raises(ValueError, match="paper mode"):
        await promote_to_live(pid, venue="binance")


# ---------------------------------------------------------------------------
# list_journal_entries
# ---------------------------------------------------------------------------


async def test_list_journal_entries(session, _patch_deps):
    pid = await _create_persona(session)
    await _seed_journal_trades(session, pid, count=5)

    from daytrader.ui.services import list_journal_entries

    entries = await list_journal_entries()
    assert len(entries) == 5
    assert all(e.event_type == "order_filled" for e in entries)


async def test_list_journal_entries_filter_by_type(session, _patch_deps):
    pid = await _create_persona(session)
    await _seed_journal_trades(session, pid, count=3)

    # Add a kill_switch entry
    with tenant_scope(TENANT_ID):
        repo = TenantRepository(session, JournalEntryModel)
        await repo.create(
            event_type="kill_switch",
            severity="critical",
            summary="Kill switch test",
            detail={},
        )
        await session.commit()

    from daytrader.ui.services import list_journal_entries

    entries = await list_journal_entries(event_type="kill_switch")
    assert len(entries) == 1
    assert entries[0].severity == "critical"


async def test_list_journal_entries_filter_by_persona(session, _patch_deps):
    pid1 = await _create_persona(session)
    await _seed_journal_trades(session, pid1, count=3)

    from daytrader.ui.services import list_journal_entries

    entries = await list_journal_entries(persona_id=pid1)
    assert len(entries) == 3

    # Filter by non-existent persona
    entries = await list_journal_entries(persona_id=uuid4())
    assert len(entries) == 0
