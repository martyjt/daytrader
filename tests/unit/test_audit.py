"""Tests for the Phase 7 audit log."""

from __future__ import annotations

from contextlib import asynccontextmanager
from decimal import Decimal
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from daytrader.core import audit
from daytrader.core.context import tenant_scope
from daytrader.execution.kill_switch import KillSwitch
from daytrader.storage.database import Base
from daytrader.storage.models import (
    AuditLogModel,
    PersonaModel,
    TenantModel,
    UserModel,
)
from daytrader.storage.repository import TenantRepository


TENANT_A = UUID("00000000-0000-0000-0000-000000000001")
TENANT_B = UUID("00000000-0000-0000-0000-000000000002")
USER_A = UUID("00000000-0000-0000-0000-0000000000aa")


@pytest_asyncio.fixture
async def engine():
    e = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with e.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield e
    await e.dispose()


@pytest_asyncio.fixture
async def session_factory(engine):
    return async_sessionmaker(engine, expire_on_commit=False)


@pytest_asyncio.fixture
async def seeded(session_factory):
    """Seed tenants + a user so FK constraints are satisfied."""
    async with session_factory() as s:
        s.add(TenantModel(id=TENANT_A, name="alpha"))
        s.add(TenantModel(id=TENANT_B, name="bravo"))
        s.add(
            UserModel(
                id=USER_A,
                tenant_id=TENANT_A,
                email="alice@alpha.test",
                password_hash="x",
                role="member",
            )
        )
        await s.commit()
    return session_factory


def _patch_get_session(monkeypatch, factory, *modules: str) -> None:
    @asynccontextmanager
    async def _get():
        async with factory() as s:
            try:
                yield s
            except Exception:
                await s.rollback()
                raise

    for mod in modules:
        monkeypatch.setattr(f"{mod}.get_session", _get)


# ---------------------------------------------------------------------------
# record() — basic writes
# ---------------------------------------------------------------------------


async def test_record_writes_row_with_explicit_ids(seeded, monkeypatch):
    _patch_get_session(monkeypatch, seeded, "daytrader.core.audit")
    monkeypatch.setattr(audit, "current_tenant_id", lambda: None)
    monkeypatch.setattr(audit, "current_user_id", lambda: None)

    rid = uuid4()
    await audit.record(
        "test.action",
        resource_type="thing",
        resource_id=rid,
        tenant_id=TENANT_A,
        user_id=USER_A,
        extra={"k": "v"},
    )

    async with seeded() as s:
        row = (await s.execute(select(AuditLogModel))).scalar_one()

    assert row.action == "test.action"
    assert row.resource_type == "thing"
    assert row.resource_id == str(rid)
    assert row.tenant_id == TENANT_A
    assert row.user_id == USER_A
    assert row.extra == {"k": "v"}
    assert row.created_at is not None


async def test_record_falls_back_to_session_context(seeded, monkeypatch):
    _patch_get_session(monkeypatch, seeded, "daytrader.core.audit")
    monkeypatch.setattr(audit, "current_tenant_id", lambda: TENANT_A)
    monkeypatch.setattr(audit, "current_user_id", lambda: USER_A)

    await audit.record("session.action")

    async with seeded() as s:
        row = (await s.execute(select(AuditLogModel))).scalar_one()
    assert row.tenant_id == TENANT_A
    assert row.user_id == USER_A


async def test_record_explicit_overrides_session(seeded, monkeypatch):
    """Non-None explicit args override the session fallback."""
    _patch_get_session(monkeypatch, seeded, "daytrader.core.audit")
    monkeypatch.setattr(audit, "current_tenant_id", lambda: TENANT_A)
    monkeypatch.setattr(audit, "current_user_id", lambda: USER_A)

    other_user = uuid4()
    # Pre-seed the override user to satisfy FK
    async with seeded() as s:
        s.add(
            UserModel(
                id=other_user,
                tenant_id=TENANT_B,
                email="bob@bravo.test",
                password_hash="x",
                role="member",
            )
        )
        await s.commit()

    await audit.record(
        "override.action", tenant_id=TENANT_B, user_id=other_user
    )

    async with seeded() as s:
        row = (await s.execute(select(AuditLogModel))).scalar_one()
    assert row.tenant_id == TENANT_B
    assert row.user_id == other_user


async def test_record_swallows_db_errors(monkeypatch, caplog):
    """A failed audit write logs a warning but does not raise."""

    @asynccontextmanager
    async def _bad_session():
        raise RuntimeError("simulated DB outage")
        yield  # pragma: no cover

    monkeypatch.setattr("daytrader.core.audit.get_session", _bad_session)
    monkeypatch.setattr(audit, "current_tenant_id", lambda: None)
    monkeypatch.setattr(audit, "current_user_id", lambda: None)

    import logging

    with caplog.at_level(logging.WARNING, logger="daytrader.core.audit"):
        await audit.record("any.action")  # must not raise

    assert any("Failed to write audit row" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Tenant scoping (audit log queries by tenant)
# ---------------------------------------------------------------------------


async def test_tenant_scoping_query(seeded, monkeypatch):
    _patch_get_session(monkeypatch, seeded, "daytrader.core.audit")
    monkeypatch.setattr(audit, "current_tenant_id", lambda: None)
    monkeypatch.setattr(audit, "current_user_id", lambda: None)

    await audit.record("a.action", tenant_id=TENANT_A)
    await audit.record("a.action", tenant_id=TENANT_A)
    await audit.record("b.action", tenant_id=TENANT_B)

    async with seeded() as s:
        rows_a = (
            await s.execute(
                select(AuditLogModel).where(AuditLogModel.tenant_id == TENANT_A)
            )
        ).scalars().all()
        rows_b = (
            await s.execute(
                select(AuditLogModel).where(AuditLogModel.tenant_id == TENANT_B)
            )
        ).scalars().all()

    assert len(rows_a) == 2
    assert len(rows_b) == 1
    assert rows_b[0].action == "b.action"


# ---------------------------------------------------------------------------
# Per-action wiring (representative action sites)
# ---------------------------------------------------------------------------


async def test_kill_switch_activate_records_audit(seeded, monkeypatch):
    """KillSwitch.activate() writes a kill_switch.activate audit row."""
    _patch_get_session(
        monkeypatch,
        seeded,
        "daytrader.core.audit",
        "daytrader.execution.kill_switch",
    )
    monkeypatch.setattr(audit, "current_tenant_id", lambda: None)
    monkeypatch.setattr(audit, "current_user_id", lambda: None)

    # Seed an active persona so activate() has work to do.
    async with seeded() as s:
        with tenant_scope(TENANT_A):
            repo = TenantRepository(s, PersonaModel)
            await repo.create(
                id=uuid4(),
                name="bot",
                mode="paper",
                asset_class="crypto",
                base_currency="USDT",
                initial_capital=Decimal("1000"),
                current_equity=Decimal("1000"),
            )
            await s.commit()

    ks = KillSwitch()
    paused = await ks.activate(TENANT_A, reason="test")
    assert paused == 1

    async with seeded() as s:
        rows = (
            await s.execute(
                select(AuditLogModel).where(
                    AuditLogModel.action == "kill_switch.activate"
                )
            )
        ).scalars().all()

    assert len(rows) == 1
    row = rows[0]
    assert row.tenant_id == TENANT_A
    assert row.extra["reason"] == "test"
    assert row.extra["personas_paused"] == 1
    # Phase 8: plugins_killed reported even when no worker was running.
    assert row.extra["plugins_killed"] is False


async def test_kill_switch_activate_records_plugin_kill(seeded, monkeypatch):
    """activate() with a plugin_manager records plugins_killed=True."""
    _patch_get_session(
        monkeypatch,
        seeded,
        "daytrader.core.audit",
        "daytrader.execution.kill_switch",
    )
    monkeypatch.setattr(audit, "current_tenant_id", lambda: None)
    monkeypatch.setattr(audit, "current_user_id", lambda: None)

    class _Manager:
        def __init__(self):
            self._tenants = {TENANT_A}
            self.calls: list = []

        def has_handle(self, t):
            return t in self._tenants

        async def shutdown_tenant(self, t):
            self.calls.append(t)
            self._tenants.discard(t)

    pm = _Manager()
    ks = KillSwitch(plugin_manager=pm)
    await ks.activate(TENANT_A, reason="manual")

    assert pm.calls == [TENANT_A]

    async with seeded() as s:
        row = (
            await s.execute(
                select(AuditLogModel).where(
                    AuditLogModel.action == "kill_switch.activate"
                )
            )
        ).scalar_one()
    assert row.extra["plugins_killed"] is True


async def test_kill_plugins_records_audit(seeded, monkeypatch):
    """kill_plugins() writes a kill_switch.plugins audit row."""
    _patch_get_session(
        monkeypatch,
        seeded,
        "daytrader.core.audit",
        "daytrader.execution.kill_switch",
    )
    monkeypatch.setattr(audit, "current_tenant_id", lambda: None)
    monkeypatch.setattr(audit, "current_user_id", lambda: None)

    class _Manager:
        def __init__(self):
            self._tenants = {TENANT_A}

        def has_handle(self, t):
            return t in self._tenants

        async def shutdown_tenant(self, t):
            self._tenants.discard(t)

    ks = KillSwitch(plugin_manager=_Manager())
    killed = await ks.kill_plugins(TENANT_A, reason="admin")
    assert killed is True

    async with seeded() as s:
        row = (
            await s.execute(
                select(AuditLogModel).where(
                    AuditLogModel.action == "kill_switch.plugins"
                )
            )
        ).scalar_one()
    assert row.tenant_id == TENANT_A
    assert row.extra["reason"] == "admin"
    assert row.extra["had_worker"] is True


async def test_login_success_records_audit(seeded, monkeypatch):
    """authenticate() with valid creds writes a login.success row."""
    from daytrader.auth import service as auth_service
    from daytrader.auth.password import hash_password

    _patch_get_session(
        monkeypatch,
        seeded,
        "daytrader.core.audit",
        "daytrader.auth.service",
    )
    monkeypatch.setattr(audit, "current_tenant_id", lambda: None)
    monkeypatch.setattr(audit, "current_user_id", lambda: None)

    # Set the user's password hash so verify_password succeeds.
    async with seeded() as s:
        user = (
            await s.execute(select(UserModel).where(UserModel.id == USER_A))
        ).scalar_one()
        user.password_hash = hash_password("hunter22")
        await s.commit()

    sess_user = await auth_service.authenticate("alice@alpha.test", "hunter22")
    assert sess_user.user_id == USER_A

    async with seeded() as s:
        rows = (
            await s.execute(
                select(AuditLogModel).where(
                    AuditLogModel.action == "login.success"
                )
            )
        ).scalars().all()

    assert len(rows) == 1
    assert rows[0].user_id == USER_A
    assert rows[0].tenant_id == TENANT_A


async def test_login_failure_records_audit(seeded, monkeypatch):
    """authenticate() with bad creds writes a login.failure row and raises."""
    from daytrader.auth import service as auth_service
    from daytrader.auth.password import hash_password

    _patch_get_session(
        monkeypatch,
        seeded,
        "daytrader.core.audit",
        "daytrader.auth.service",
    )
    monkeypatch.setattr(audit, "current_tenant_id", lambda: None)
    monkeypatch.setattr(audit, "current_user_id", lambda: None)

    async with seeded() as s:
        user = (
            await s.execute(select(UserModel).where(UserModel.id == USER_A))
        ).scalar_one()
        user.password_hash = hash_password("right-password")
        await s.commit()

    with pytest.raises(auth_service.AuthError):
        await auth_service.authenticate("alice@alpha.test", "wrong")
    with pytest.raises(auth_service.AuthError):
        await auth_service.authenticate("nobody@nowhere.test", "x")

    async with seeded() as s:
        rows = (
            await s.execute(
                select(AuditLogModel)
                .where(AuditLogModel.action == "login.failure")
                .order_by(AuditLogModel.created_at)
            )
        ).scalars().all()

    assert len(rows) == 2
    reasons = {r.extra.get("reason") for r in rows}
    assert reasons == {"bad_password", "no_user"}
