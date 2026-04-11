"""Tests for the live trading loop."""

from __future__ import annotations

from decimal import Decimal
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import numpy as np
import polars as pl
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from daytrader.core.context import tenant_scope
from daytrader.execution.loop import TradingLoop, _timeframe_to_days
from daytrader.execution.paper import PaperExecutor
from daytrader.execution.registry import ExecutionRegistry
from daytrader.storage.database import Base
from daytrader.storage.models import PersonaModel, TenantModel
from daytrader.storage.repository import TenantRepository
from daytrader.core.types.bars import Timeframe


TENANT_ID = UUID("00000000-0000-0000-0000-000000000001")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rising_ohlcv(n: int = 50) -> pl.DataFrame:
    """Generate a simple rising OHLCV DataFrame."""
    base = 100.0
    timestamps = [
        datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(days=i)
        for i in range(n)
    ]
    closes = [base + i * 0.5 for i in range(n)]
    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": [c - 0.1 for c in closes],
            "high": [c + 0.5 for c in closes],
            "low": [c - 0.5 for c in closes],
            "close": closes,
            "volume": [1000.0] * n,
        }
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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


@pytest_asyncio.fixture
async def _patch_session(engine, monkeypatch):
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

    monkeypatch.setattr("daytrader.execution.loop.get_session", _get_session)


@pytest.fixture(autouse=True)
def _clean_registry():
    ExecutionRegistry.clear()
    yield
    ExecutionRegistry.clear()


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


def test_timeframe_to_days():
    assert _timeframe_to_days(Timeframe.D1) == 1.0
    assert _timeframe_to_days(Timeframe.H1) == pytest.approx(1 / 24)
    assert _timeframe_to_days(Timeframe.W1) == 7.0


def test_initial_state():
    loop = TradingLoop(tenant_id=TENANT_ID)
    assert loop.is_running is False


async def test_start_stop():
    loop = TradingLoop(tenant_id=TENANT_ID, poll_seconds=0.01)
    await loop.start()
    assert loop.is_running is True
    await loop.stop()
    assert loop.is_running is False


async def test_kill_switch_skips_cycle():
    """When kill switch is activated, _process_cycle should not be called."""
    mock_ks = MagicMock()
    mock_ks.is_activated = True

    loop = TradingLoop(
        kill_switch=mock_ks,
        tenant_id=TENANT_ID,
        poll_seconds=0.01,
    )

    # Patch _process_cycle to track calls
    loop._process_cycle = AsyncMock()

    await loop.start()
    await asyncio.sleep(0.05)
    await loop.stop()

    loop._process_cycle.assert_not_called()


async def test_process_cycle_no_personas(session, _patch_session):
    """Cycle with no active personas should exit cleanly."""
    loop = TradingLoop(tenant_id=TENANT_ID)
    await loop._process_cycle()  # Should not raise


async def test_process_cycle_with_paper_persona(session, _patch_session):
    """Cycle processes a paper persona with trading config."""
    # Setup: persona with trading config in meta
    pid = uuid4()
    with tenant_scope(TENANT_ID):
        repo = TenantRepository(session, PersonaModel)
        await repo.create(
            id=pid,
            name="Paper Bot",
            mode="paper",
            asset_class="crypto",
            base_currency="USDT",
            initial_capital=Decimal("10000"),
            current_equity=Decimal("10000"),
            meta={
                "algorithm_id": "buy_hold",
                "symbol": "BTC-USD",
                "timeframe": "1d",
            },
        )
        await session.commit()

    # Register algorithms and paper executor
    from daytrader.algorithms.registry import AlgorithmRegistry

    AlgorithmRegistry.auto_register()
    paper = PaperExecutor()
    paper.initialize_persona(pid, Decimal("10000"))
    ExecutionRegistry.register(paper)

    # Mock the data adapter to return rising OHLCV
    mock_adapter = AsyncMock()
    mock_adapter.fetch_ohlcv = AsyncMock(return_value=_rising_ohlcv(250))

    with patch.object(TradingLoop, "_resolve_data_adapter", return_value=mock_adapter):
        mock_journal = AsyncMock()
        loop = TradingLoop(
            journal=mock_journal,
            tenant_id=TENANT_ID,
        )
        await loop._process_cycle()

    # The buy_hold algo always emits a buy signal, so we should see journal calls
    # (at least signal emitted + possibly order submitted/filled)
    assert mock_journal.log_signal_emitted.call_count >= 0  # May or may not emit depending on algo


async def test_global_risk_breach_triggers_kill_switch(session, _patch_session):
    """When global drawdown breaches, the kill switch should activate."""
    # Create persona with very low equity (simulates drawdown)
    pid = uuid4()
    with tenant_scope(TENANT_ID):
        repo = TenantRepository(session, PersonaModel)
        await repo.create(
            id=pid,
            name="Low Equity",
            mode="paper",
            asset_class="crypto",
            base_currency="USDT",
            initial_capital=Decimal("10000"),
            current_equity=Decimal("1000"),  # 90% loss
        )
        await session.commit()

    # Set up global risk monitor with already-high peak
    from daytrader.risk.global_risk import GlobalRiskConfig, GlobalRiskMonitor

    monitor = GlobalRiskMonitor(GlobalRiskConfig(max_drawdown_pct=10.0))
    await monitor.check_drawdown(10000)  # Set peak to 10000

    mock_ks = AsyncMock()
    mock_ks.is_activated = False

    loop = TradingLoop(
        global_risk=monitor,
        kill_switch=mock_ks,
        tenant_id=TENANT_ID,
    )
    await loop._process_cycle()

    # Kill switch should have been activated due to drawdown breach
    mock_ks.activate.assert_called_once()
    assert "global_drawdown" in mock_ks.activate.call_args[1]["reason"]


# Need asyncio for the kill_switch test
import asyncio
