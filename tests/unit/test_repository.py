"""Tests for tenant-scoped repository using in-memory SQLite.

Proves that:
- CRUD operations work
- Tenant isolation is enforced (tenant A cannot see tenant B's data)
- Operations outside a tenant scope raise
"""

from decimal import Decimal
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker

from daytrader.core.context import current_tenant, tenant_scope
from daytrader.storage.models import PersonaModel, TenantModel
from daytrader.storage.repository import TenantRepository


@pytest_asyncio.fixture
async def session(engine):
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as s:
        yield s


@pytest_asyncio.fixture
async def tenant_a(session):
    tid = uuid4()
    session.add(TenantModel(id=tid, name="tenant_a"))
    await session.commit()
    return tid


@pytest_asyncio.fixture
async def tenant_b(session):
    tid = uuid4()
    session.add(TenantModel(id=tid, name="tenant_b"))
    await session.commit()
    return tid


def _persona_kwargs(**overrides):
    defaults = dict(
        name="TestBot",
        mode="paper",
        asset_class="crypto",
        base_currency="USDT",
        initial_capital=Decimal("10000"),
        current_equity=Decimal("10000"),
        risk_profile="balanced",
    )
    defaults.update(overrides)
    return defaults


async def test_create_and_get(session, tenant_a):
    repo = TenantRepository(session, PersonaModel)
    with tenant_scope(tenant_a):
        p = await repo.create(**_persona_kwargs())
        assert p.name == "TestBot"
        assert p.tenant_id == tenant_a

        fetched = await repo.get(p.id)
        assert fetched is not None
        assert fetched.id == p.id


async def test_get_all(session, tenant_a):
    repo = TenantRepository(session, PersonaModel)
    with tenant_scope(tenant_a):
        await repo.create(**_persona_kwargs(name="Bot1"))
        await repo.create(**_persona_kwargs(name="Bot2"))
        all_p = await repo.get_all()
        assert len(all_p) == 2


async def test_update(session, tenant_a):
    repo = TenantRepository(session, PersonaModel)
    with tenant_scope(tenant_a):
        p = await repo.create(**_persona_kwargs())
        updated = await repo.update(p.id, name="RenamedBot")
        assert updated.name == "RenamedBot"


async def test_delete(session, tenant_a):
    repo = TenantRepository(session, PersonaModel)
    with tenant_scope(tenant_a):
        p = await repo.create(**_persona_kwargs())
        deleted = await repo.delete(p.id)
        assert deleted is True
        assert await repo.get(p.id) is None


async def test_count(session, tenant_a):
    repo = TenantRepository(session, PersonaModel)
    with tenant_scope(tenant_a):
        await repo.create(**_persona_kwargs(name="A"))
        await repo.create(**_persona_kwargs(name="B"))
        assert await repo.count() == 2


async def test_tenant_isolation(session, tenant_a, tenant_b):
    """Tenant A's personas are invisible to tenant B and vice versa."""
    repo = TenantRepository(session, PersonaModel)

    with tenant_scope(tenant_a):
        await repo.create(**_persona_kwargs(name="A-Bot"))

    with tenant_scope(tenant_b):
        await repo.create(**_persona_kwargs(name="B-Bot"))

    with tenant_scope(tenant_a):
        mine = await repo.get_all()
        assert len(mine) == 1
        assert mine[0].name == "A-Bot"

    with tenant_scope(tenant_b):
        theirs = await repo.get_all()
        assert len(theirs) == 1
        assert theirs[0].name == "B-Bot"


async def test_cross_tenant_get_returns_none(session, tenant_a, tenant_b):
    """Getting by ID from the wrong tenant returns None, not the record."""
    repo = TenantRepository(session, PersonaModel)

    with tenant_scope(tenant_a):
        p = await repo.create(**_persona_kwargs(name="A-Bot"))

    with tenant_scope(tenant_b):
        assert await repo.get(p.id) is None


async def test_no_tenant_scope_raises(session):
    """Operations without a tenant scope fail immediately."""
    repo = TenantRepository(session, PersonaModel)
    with pytest.raises(RuntimeError):
        await repo.get_all()
