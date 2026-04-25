"""Tests for the execution adapter registry (paper + per-tenant brokers)."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from daytrader.execution.base import ExecutionAdapter
from daytrader.execution.paper import PaperExecutor
from daytrader.execution.registry import ExecutionRegistry


@pytest.fixture(autouse=True)
def _clean_registry():
    """Ensure each test starts with a fresh registry."""
    ExecutionRegistry.clear()
    yield
    ExecutionRegistry.clear()


# ---- global (paper) registry --------------------------------------------


def test_register_and_get():
    adapter = PaperExecutor()
    ExecutionRegistry.register(adapter)
    assert ExecutionRegistry.get("paper") is adapter


def test_get_unknown_raises():
    with pytest.raises(KeyError, match="not registered"):
        ExecutionRegistry.get("nonexistent")


def test_available_returns_sorted_names():
    ExecutionRegistry.register(PaperExecutor())
    assert ExecutionRegistry.available() == ["paper"]


def test_clear_removes_all():
    ExecutionRegistry.register(PaperExecutor())
    ExecutionRegistry.clear()
    assert ExecutionRegistry.available() == []


def test_auto_register_only_adds_paper():
    """auto_register registers paper but NOT live brokers (those are per-tenant)."""
    ExecutionRegistry.auto_register()
    assert ExecutionRegistry.available() == ["paper"]


def test_auto_register_idempotent():
    ExecutionRegistry.auto_register()
    first = ExecutionRegistry.get("paper")
    ExecutionRegistry.auto_register()
    assert ExecutionRegistry.get("paper") is first


# ---- per-tenant adapters -------------------------------------------------


class _StubExecutor(ExecutionAdapter):
    @property
    def name(self) -> str:
        return "binance"

    async def submit_order(self, order):  # pragma: no cover - unused
        return order

    async def cancel_order(self, order_id):  # pragma: no cover
        return True

    async def get_positions(self, persona_id):  # pragma: no cover
        return {}

    async def get_balance(self, persona_id) -> Decimal:
        return Decimal("123.45")


@pytest.mark.asyncio
async def test_get_for_tenant_paper_returns_global():
    paper = PaperExecutor()
    ExecutionRegistry.register(paper)
    out = await ExecutionRegistry.get_for_tenant(uuid4(), "paper")
    assert out is paper


@pytest.mark.asyncio
async def test_get_for_tenant_returns_none_when_no_credentials():
    with patch(
        "daytrader.execution.credentials.get_decrypted",
        new=AsyncMock(return_value=None),
    ):
        result = await ExecutionRegistry.get_for_tenant(uuid4(), "binance")
    assert result is None


@pytest.mark.asyncio
async def test_get_for_tenant_caches_per_tenant_broker_pair():
    tenant = uuid4()
    payload = {"api_key": "k", "api_secret": "s"}
    stub = _StubExecutor()
    with patch(
        "daytrader.execution.credentials.get_decrypted",
        new=AsyncMock(return_value=(payload, True)),
    ), patch(
        "daytrader.execution.credentials.build_executor",
        return_value=stub,
    ) as build_mock:
        first = await ExecutionRegistry.get_for_tenant(tenant, "binance")
        second = await ExecutionRegistry.get_for_tenant(tenant, "binance")
    assert first is stub
    assert second is stub
    # build_executor should only be called once thanks to the cache
    assert build_mock.call_count == 1


@pytest.mark.asyncio
async def test_invalidate_tenant_drops_cached_adapter():
    tenant = uuid4()
    payload = {"api_key": "k", "api_secret": "s"}
    with patch(
        "daytrader.execution.credentials.get_decrypted",
        new=AsyncMock(return_value=(payload, True)),
    ), patch(
        "daytrader.execution.credentials.build_executor",
        return_value=_StubExecutor(),
    ) as build_mock:
        await ExecutionRegistry.get_for_tenant(tenant, "binance")
        ExecutionRegistry.invalidate_tenant(tenant)
        await ExecutionRegistry.get_for_tenant(tenant, "binance")
    assert build_mock.call_count == 2


@pytest.mark.asyncio
async def test_invalidate_tenant_only_drops_named_broker():
    tenant = uuid4()
    payload = {"api_key": "k", "api_secret": "s"}
    with patch(
        "daytrader.execution.credentials.get_decrypted",
        new=AsyncMock(return_value=(payload, True)),
    ), patch(
        "daytrader.execution.credentials.build_executor",
        return_value=_StubExecutor(),
    ):
        await ExecutionRegistry.get_for_tenant(tenant, "binance")
        await ExecutionRegistry.get_for_tenant(tenant, "alpaca")
        ExecutionRegistry.invalidate_tenant(tenant, "binance")
        cached = ExecutionRegistry.cached_tenant_adapters()
    keys = {k[1] for k in cached if k[0] == tenant}
    assert keys == {"alpaca"}
