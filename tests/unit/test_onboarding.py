"""Tests for the Phase 13 /welcome onboarding-status helper."""

from __future__ import annotations

from contextlib import asynccontextmanager
from decimal import Decimal
from uuid import UUID

import pytest_asyncio
from cryptography.fernet import Fernet
from sqlalchemy.ext.asyncio import async_sessionmaker

from daytrader.core.crypto import get_codec, reset_codec_cache
from daytrader.storage.models import (
    BrokerCredentialModel,
    PersonaModel,
    TenantModel,
)
from daytrader.ui.services_onboarding import get_onboarding_status

TENANT = UUID("00000000-0000-0000-0000-0000000000b1")


@pytest_asyncio.fixture
async def session_factory(engine, monkeypatch):
    key = Fernet.generate_key().decode()
    monkeypatch.setenv("APP_ENCRYPTION_KEY", key)
    from daytrader.core import settings as settings_mod

    settings_mod.get_settings.cache_clear()
    reset_codec_cache()
    yield async_sessionmaker(engine, expire_on_commit=False)
    settings_mod.get_settings.cache_clear()
    reset_codec_cache()


@pytest_asyncio.fixture
async def seeded(session_factory):
    async with session_factory() as s:
        s.add(TenantModel(id=TENANT, name="alpha"))
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


async def test_onboarding_status_fresh(seeded, monkeypatch):
    _patch_get_session(
        monkeypatch,
        seeded,
        "daytrader.ui.services_onboarding",
        "daytrader.notifications.service",
    )
    status = await get_onboarding_status(TENANT)
    assert status.has_persona is False
    assert status.has_broker_creds is False
    assert status.has_webhook is False
    assert status.is_fresh is True
    assert status.is_complete is False


async def test_onboarding_status_persona_only(seeded, monkeypatch):
    _patch_get_session(
        monkeypatch,
        seeded,
        "daytrader.ui.services_onboarding",
        "daytrader.notifications.service",
    )
    async with seeded() as s:
        s.add(
            PersonaModel(
                tenant_id=TENANT,
                name="p",
                mode="paper",
                asset_class="crypto",
                base_currency="USDT",
                initial_capital=Decimal(100),
                current_equity=Decimal(100),
                risk_profile="balanced",
            )
        )
        await s.commit()

    status = await get_onboarding_status(TENANT)
    assert status.has_persona is True
    assert status.has_broker_creds is False
    assert status.is_fresh is False
    assert status.is_complete is False


async def test_onboarding_status_complete(seeded, monkeypatch):
    _patch_get_session(
        monkeypatch,
        seeded,
        "daytrader.ui.services_onboarding",
        "daytrader.notifications.service",
    )
    async with seeded() as s:
        s.add(
            PersonaModel(
                tenant_id=TENANT,
                name="p",
                mode="paper",
                asset_class="crypto",
                base_currency="USDT",
                initial_capital=Decimal(100),
                current_equity=Decimal(100),
                risk_profile="balanced",
            )
        )
        s.add(
            BrokerCredentialModel(
                tenant_id=TENANT,
                broker_name="binance",
                credential_data="x",
                is_testnet=True,
            )
        )
        row = await s.get(TenantModel, TENANT)
        assert row is not None
        row.notification_webhook_url = get_codec().encrypt(
            "https://hooks.example/abc"
        )
        await s.commit()

    status = await get_onboarding_status(TENANT)
    assert status.has_persona is True
    assert status.has_broker_creds is True
    assert status.has_webhook is True
    assert status.is_fresh is False
    assert status.is_complete is True
