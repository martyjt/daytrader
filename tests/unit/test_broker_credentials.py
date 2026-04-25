"""Tests for the broker credentials service."""

from __future__ import annotations

from contextlib import asynccontextmanager
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from cryptography.fernet import Fernet
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from daytrader.core.crypto import (
    assert_encryption_key_for_existing_secrets,
    reset_codec_cache,
)
from daytrader.execution.credentials import (
    CredentialError,
    delete_credential,
    get_decrypted,
    list_credentials,
    save_credential,
)
from daytrader.execution.registry import ExecutionRegistry
from daytrader.storage.database import Base
from daytrader.storage.models import TenantModel

TENANT_A = UUID("00000000-0000-0000-0000-000000000001")
TENANT_B = UUID("00000000-0000-0000-0000-000000000002")


@pytest_asyncio.fixture
async def engine():
    e = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with e.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield e
    await e.dispose()


@pytest_asyncio.fixture
async def _patch_session(engine, monkeypatch):
    factory = async_sessionmaker(engine, expire_on_commit=False)

    @asynccontextmanager
    async def _get_session():
        async with factory() as s:
            try:
                yield s
            except Exception:
                await s.rollback()
                raise

    # Patch the module attribute so ``from ...storage.database import get_session``
    # picks up the test factory at call time.
    monkeypatch.setattr("daytrader.storage.database.get_session", _get_session)
    monkeypatch.setattr("daytrader.execution.credentials.get_session", _get_session)

    async with factory() as s:
        s.add(TenantModel(id=TENANT_A, name="a"))
        s.add(TenantModel(id=TENANT_B, name="b"))
        await s.commit()
    return factory


@pytest.fixture(autouse=True)
def _force_codec(monkeypatch):
    """Use a deterministic Fernet key for the test session.

    ``get_settings`` and ``get_codec`` are both ``functools.lru_cache``-d.
    Clearing them lets ``SecretCodec()`` re-read ``APP_ENCRYPTION_KEY``.
    """
    key = Fernet.generate_key().decode()
    monkeypatch.setenv("APP_ENCRYPTION_KEY", key)

    from daytrader.core import settings as settings_mod

    settings_mod.get_settings.cache_clear()
    reset_codec_cache()
    yield
    settings_mod.get_settings.cache_clear()
    reset_codec_cache()
    ExecutionRegistry.clear()


# ---- save / list / delete -----------------------------------------------


@pytest.mark.asyncio
async def test_save_and_list(_patch_session):
    cid = await save_credential(
        tenant_id=TENANT_A,
        broker_name="binance",
        fields={"api_key": "pubkey", "api_secret": "secret"},
        is_testnet=True,
    )
    assert isinstance(cid, UUID)

    items = await list_credentials(TENANT_A)
    assert len(items) == 1
    assert items[0].broker_name == "binance"
    assert items[0].is_testnet is True
    # api_key is masked, never plaintext
    assert "pubkey" not in items[0].api_key_masked


@pytest.mark.asyncio
async def test_save_replaces_existing(_patch_session):
    await save_credential(
        tenant_id=TENANT_A,
        broker_name="binance",
        fields={"api_key": "old", "api_secret": "old"},
        is_testnet=True,
    )
    await save_credential(
        tenant_id=TENANT_A,
        broker_name="binance",
        fields={"api_key": "new", "api_secret": "new"},
        is_testnet=False,
    )
    items = await list_credentials(TENANT_A)
    assert len(items) == 1
    assert items[0].is_testnet is False

    payload, is_testnet = await get_decrypted(TENANT_A, "binance")
    assert payload == {"api_key": "new", "api_secret": "new"}
    assert is_testnet is False


@pytest.mark.asyncio
async def test_list_is_tenant_scoped(_patch_session):
    await save_credential(
        tenant_id=TENANT_A,
        broker_name="binance",
        fields={"api_key": "a", "api_secret": "a"},
        is_testnet=True,
    )
    await save_credential(
        tenant_id=TENANT_B,
        broker_name="alpaca",
        fields={"api_key": "b", "api_secret": "b"},
        is_testnet=True,
    )
    a = await list_credentials(TENANT_A)
    b = await list_credentials(TENANT_B)
    assert {r.broker_name for r in a} == {"binance"}
    assert {r.broker_name for r in b} == {"alpaca"}


@pytest.mark.asyncio
async def test_delete_credential(_patch_session):
    cid = await save_credential(
        tenant_id=TENANT_A,
        broker_name="binance",
        fields={"api_key": "k", "api_secret": "s"},
        is_testnet=True,
    )
    ok = await delete_credential(tenant_id=TENANT_A, credential_id=cid)
    assert ok is True
    assert await list_credentials(TENANT_A) == []


@pytest.mark.asyncio
async def test_delete_unknown_returns_false(_patch_session):
    ok = await delete_credential(tenant_id=TENANT_A, credential_id=uuid4())
    assert ok is False


@pytest.mark.asyncio
async def test_delete_other_tenant_returns_false(_patch_session):
    cid = await save_credential(
        tenant_id=TENANT_A,
        broker_name="binance",
        fields={"api_key": "k", "api_secret": "s"},
        is_testnet=True,
    )
    # TENANT_B should not be able to delete TENANT_A's credential.
    ok = await delete_credential(tenant_id=TENANT_B, credential_id=cid)
    assert ok is False
    assert len(await list_credentials(TENANT_A)) == 1


# ---- validation ----------------------------------------------------------


@pytest.mark.asyncio
async def test_save_rejects_unknown_broker(_patch_session):
    with pytest.raises(CredentialError, match="Unsupported"):
        await save_credential(
            tenant_id=TENANT_A,
            broker_name="bogus",
            fields={"api_key": "k", "api_secret": "s"},
            is_testnet=True,
        )


@pytest.mark.asyncio
async def test_save_rejects_empty_fields(_patch_session):
    with pytest.raises(CredentialError, match="required"):
        await save_credential(
            tenant_id=TENANT_A,
            broker_name="binance",
            fields={"api_key": "", "api_secret": "s"},
            is_testnet=True,
        )


# ---- registry integration ------------------------------------------------


@pytest.mark.asyncio
async def test_save_invalidates_cached_adapter(_patch_session, monkeypatch):
    # Pre-populate the cache as if an adapter were already built.
    fake_adapter = object()
    ExecutionRegistry._tenant_adapters[(TENANT_A, "binance")] = fake_adapter  # type: ignore[assignment]

    await save_credential(
        tenant_id=TENANT_A,
        broker_name="binance",
        fields={"api_key": "k", "api_secret": "s"},
        is_testnet=True,
    )
    assert (TENANT_A, "binance") not in ExecutionRegistry._tenant_adapters


# ---- startup validation --------------------------------------------------


@pytest.mark.asyncio
async def test_startup_assertion_passes_when_key_matches(_patch_session):
    await save_credential(
        tenant_id=TENANT_A,
        broker_name="binance",
        fields={"api_key": "k", "api_secret": "s"},
        is_testnet=True,
    )
    await assert_encryption_key_for_existing_secrets()  # should not raise


@pytest.mark.asyncio
async def test_startup_assertion_passes_with_no_secrets(_patch_session, monkeypatch):
    monkeypatch.setenv("APP_ENCRYPTION_KEY", "")
    await assert_encryption_key_for_existing_secrets()
