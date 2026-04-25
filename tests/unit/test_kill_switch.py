"""Tests for the kill switch."""

from __future__ import annotations

from decimal import Decimal
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker

from daytrader.core.context import tenant_scope
from daytrader.execution.kill_switch import KillSwitch
from daytrader.storage.models import PersonaModel, TenantModel
from daytrader.storage.repository import TenantRepository


TENANT_ID = UUID("00000000-0000-0000-0000-000000000001")


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


# ---------------------------------------------------------------------------
# Plugin worker shutdown (Phase 8)
# ---------------------------------------------------------------------------


class _FakePluginManager:
    """Stand-in for PluginWorkerManager — records shutdown_tenant calls."""

    def __init__(self, tenants_with_workers: set[UUID] | None = None) -> None:
        self._tenants = set(tenants_with_workers or set())
        self.shutdown_calls: list[UUID] = []

    def has_handle(self, tenant_id: UUID) -> bool:
        return tenant_id in self._tenants

    async def shutdown_tenant(self, tenant_id: UUID) -> None:
        self.shutdown_calls.append(tenant_id)
        self._tenants.discard(tenant_id)


async def test_activate_kills_plugin_worker_when_one_runs(session, _patch_session):
    """activate() should also tear down the tenant's plugin worker."""
    pm = _FakePluginManager(tenants_with_workers={TENANT_ID})
    ks = KillSwitch(plugin_manager=pm)

    await ks.activate(TENANT_ID, reason="manual")

    assert pm.shutdown_calls == [TENANT_ID]


async def test_activate_skips_plugin_shutdown_when_no_worker(session, _patch_session):
    """No worker for that tenant → don't call shutdown_tenant."""
    pm = _FakePluginManager(tenants_with_workers=set())
    ks = KillSwitch(plugin_manager=pm)

    await ks.activate(TENANT_ID, reason="manual")

    assert pm.shutdown_calls == []


async def test_activate_works_without_plugin_manager(session, _patch_session):
    """Backwards-compat: KillSwitch with no plugin manager still pauses personas."""
    await _add_persona(session, "Live Bot", "live")
    ks = KillSwitch()  # no plugin_manager

    count = await ks.activate(TENANT_ID, reason="manual")

    assert count == 1


async def test_activate_continues_when_plugin_shutdown_raises(
    session, _patch_session
):
    """A plugin worker crash during shutdown must not abort the kill."""
    await _add_persona(session, "Live Bot", "live")

    class Boom(_FakePluginManager):
        async def shutdown_tenant(self, tenant_id):  # type: ignore[override]
            raise RuntimeError("worker stuck")

    pm = Boom(tenants_with_workers={TENANT_ID})
    ks = KillSwitch(plugin_manager=pm)

    count = await ks.activate(TENANT_ID, reason="manual")

    assert count == 1  # personas still paused
    assert ks.is_activated is True


async def test_kill_plugins_targets_only_workers(session, _patch_session):
    """kill_plugins() tears down the worker but leaves personas alone."""
    await _add_persona(session, "Live Bot", "live")
    pm = _FakePluginManager(tenants_with_workers={TENANT_ID})
    ks = KillSwitch(plugin_manager=pm)

    killed = await ks.kill_plugins(TENANT_ID, reason="admin")

    assert killed is True
    assert pm.shutdown_calls == [TENANT_ID]

    # Persona stays in live mode — kill_plugins must not pause trading.
    with tenant_scope(TENANT_ID):
        repo = TenantRepository(session, PersonaModel)
        personas = await repo.get_all()
        assert all(p.mode == "live" for p in personas)

    # Doesn't flip the global activation flag either.
    assert ks.is_activated is False


async def test_kill_plugins_returns_false_when_no_worker(session, _patch_session):
    pm = _FakePluginManager(tenants_with_workers=set())
    ks = KillSwitch(plugin_manager=pm)

    killed = await ks.kill_plugins(TENANT_ID, reason="admin")

    assert killed is False
    assert pm.shutdown_calls == []
