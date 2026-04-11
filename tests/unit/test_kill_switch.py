"""Tests for the kill switch."""

from __future__ import annotations

from decimal import Decimal
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from daytrader.core.context import tenant_scope
from daytrader.execution.kill_switch import KillSwitch
from daytrader.storage.database import Base
from daytrader.storage.models import PersonaModel, TenantModel
from daytrader.storage.repository import TenantRepository


TENANT_ID = UUID("00000000-0000-0000-0000-000000000001")


@pytest_asyncio.fixture
async def engine():
    e = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with e.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield e
    await e.dispose()


@pytest_asyncio.fixture
async def session(engine):
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as s:
        s.add(TenantModel(id=TENANT_ID, name="test"))
        await s.commit()
        yield s


async def _add_persona(session, name: str, mode: str) -> UUID:
    """Helper to add a persona in the test DB."""
    pid = uuid4()
    with tenant_scope(TENANT_ID):
        repo = TenantRepository(session, PersonaModel)
        await repo.create(
            id=pid,
            name=name,
            mode=mode,
            asset_class="crypto",
            base_currency="USDT",
            initial_capital=Decimal("10000"),
            current_equity=Decimal("10000"),
        )
        await session.commit()
    return pid


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

    monkeypatch.setattr("daytrader.execution.kill_switch.get_session", _get_session)


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------


def test_initial_state():
    ks = KillSwitch()
    assert ks.is_activated is False


def test_reset_clears():
    ks = KillSwitch()
    ks._activated.set()
    assert ks.is_activated is True
    ks.reset()
    assert ks.is_activated is False


# ---------------------------------------------------------------------------
# Activate — pauses personas
# ---------------------------------------------------------------------------


async def test_activate_pauses_live_and_paper(session, _patch_session):
    await _add_persona(session, "Live Bot", "live")
    await _add_persona(session, "Paper Bot", "paper")
    await _add_persona(session, "Backtest Bot", "backtest")

    ks = KillSwitch()
    count = await ks.activate(TENANT_ID, reason="test")

    assert count == 2
    assert ks.is_activated is True

    # Verify DB state
    with tenant_scope(TENANT_ID):
        repo = TenantRepository(session, PersonaModel)
        all_personas = await repo.get_all()
        modes = {p.name: p.mode for p in all_personas}

    assert modes["Live Bot"] == "paused"
    assert modes["Paper Bot"] == "paused"
    assert modes["Backtest Bot"] == "backtest"  # Unchanged


async def test_activate_with_no_active_personas(session, _patch_session):
    await _add_persona(session, "Backtest Only", "backtest")

    ks = KillSwitch()
    count = await ks.activate(TENANT_ID)

    assert count == 0
    assert ks.is_activated is True


async def test_activate_with_journal(session, _patch_session):
    """Kill switch logs to journal when journal writer is provided."""
    from unittest.mock import AsyncMock

    mock_journal = AsyncMock()
    ks = KillSwitch(journal=mock_journal)

    await ks.activate(TENANT_ID, reason="drawdown")

    mock_journal.log_kill_switch.assert_called_once_with(TENANT_ID, "drawdown")
