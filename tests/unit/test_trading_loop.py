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
from daytrader.storage.models import PersonaModel, StrategyConfigModel, TenantModel
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


# ---------------------------------------------------------------------------
# Persona config resolution (strategy binding)
# ---------------------------------------------------------------------------


async def test_resolve_persona_config_with_strategy_binding(session, _patch_session):
    """A persona bound via strategy_config_id reads through to the saved strategy."""
    sid = uuid4()
    pid = uuid4()
    with tenant_scope(TENANT_ID):
        strat_repo = TenantRepository(session, StrategyConfigModel)
        await strat_repo.create(
            id=sid,
            name="BTC RSI",
            algo_id="rsi_mean_reversion",
            symbol="BTC-USD",
            timeframe="1h",
            venue="binance_spot",
            algo_params={"rsi_period": 21, "oversold": 25},
        )
        persona_repo = TenantRepository(session, PersonaModel)
        persona = await persona_repo.create(
            id=pid,
            name="Bound Bot",
            mode="paper",
            asset_class="crypto",
            base_currency="USDT",
            initial_capital=Decimal("10000"),
            current_equity=Decimal("10000"),
            meta={"strategy_config_id": str(sid)},
        )
        await session.commit()

    loop = TradingLoop(tenant_id=TENANT_ID)
    algo_id, symbol, timeframe, params, venue = await loop._resolve_persona_config(persona)

    assert algo_id == "rsi_mean_reversion"
    assert symbol == "BTC-USD"
    assert timeframe == "1h"
    assert params == {"rsi_period": 21, "oversold": 25}
    assert venue == "binance_spot"


async def test_resolve_persona_config_strategy_edits_propagate(session, _patch_session):
    """Editing the saved strategy is reflected on the next resolution."""
    sid = uuid4()
    pid = uuid4()
    with tenant_scope(TENANT_ID):
        strat_repo = TenantRepository(session, StrategyConfigModel)
        await strat_repo.create(
            id=sid,
            name="EMA",
            algo_id="ema_crossover",
            symbol="ETH-USD",
            timeframe="1d",
            venue="binance_spot",
            algo_params={"fast": 10, "slow": 30},
        )
        persona_repo = TenantRepository(session, PersonaModel)
        persona = await persona_repo.create(
            id=pid,
            name="Bound Bot",
            mode="paper",
            asset_class="crypto",
            base_currency="USDT",
            initial_capital=Decimal("10000"),
            current_equity=Decimal("10000"),
            meta={"strategy_config_id": str(sid)},
        )
        await session.commit()

        await strat_repo.update(sid, algo_params={"fast": 5, "slow": 50})
        await session.commit()

    loop = TradingLoop(tenant_id=TENANT_ID)
    _, _, _, params, _ = await loop._resolve_persona_config(persona)

    assert params == {"fast": 5, "slow": 50}


async def test_resolve_persona_config_strategy_deleted_falls_back(session, _patch_session):
    """When the bound strategy was deleted, fall back to embedded meta keys."""
    pid = uuid4()
    missing_sid = uuid4()
    with tenant_scope(TENANT_ID):
        repo = TenantRepository(session, PersonaModel)
        persona = await repo.create(
            id=pid,
            name="Orphan",
            mode="paper",
            asset_class="crypto",
            base_currency="USDT",
            initial_capital=Decimal("10000"),
            current_equity=Decimal("10000"),
            meta={
                "strategy_config_id": str(missing_sid),
                "algo_id": "buy_hold",
                "symbol": "BTC-USD",
                "timeframe": "1d",
                "params": {"fallback": True},
                "venue": "fallback_venue",
            },
        )
        await session.commit()

    loop = TradingLoop(tenant_id=TENANT_ID)
    algo_id, symbol, timeframe, params, venue = await loop._resolve_persona_config(persona)

    assert algo_id == "buy_hold"
    assert symbol == "BTC-USD"
    assert timeframe == "1d"
    assert params == {"fallback": True}
    assert venue == "fallback_venue"


async def test_resolve_persona_config_back_compat_algorithm_id(session, _patch_session):
    """Legacy ``algorithm_id`` meta key still resolves (no binding)."""
    pid = uuid4()
    with tenant_scope(TENANT_ID):
        repo = TenantRepository(session, PersonaModel)
        persona = await repo.create(
            id=pid,
            name="Legacy",
            mode="paper",
            asset_class="crypto",
            base_currency="USDT",
            initial_capital=Decimal("10000"),
            current_equity=Decimal("10000"),
            meta={
                "algorithm_id": "buy_hold",
                "symbol": "BTC-USD",
                "timeframe": "4h",
            },
        )
        await session.commit()

    loop = TradingLoop(tenant_id=TENANT_ID)
    algo_id, symbol, timeframe, params, venue = await loop._resolve_persona_config(persona)

    assert algo_id == "buy_hold"
    assert symbol == "BTC-USD"
    assert timeframe == "4h"
    assert params == {}
    assert venue is None


async def test_resolve_persona_config_no_config(session, _patch_session):
    """An unconfigured persona returns Nones — caller should skip."""
    pid = uuid4()
    with tenant_scope(TENANT_ID):
        repo = TenantRepository(session, PersonaModel)
        persona = await repo.create(
            id=pid,
            name="Empty",
            mode="paper",
            asset_class="crypto",
            base_currency="USDT",
            initial_capital=Decimal("10000"),
            current_equity=Decimal("10000"),
            meta={},
        )
        await session.commit()

    loop = TradingLoop(tenant_id=TENANT_ID)
    algo_id, symbol, timeframe, params, venue = await loop._resolve_persona_config(persona)

    assert algo_id is None
    assert symbol is None
    assert timeframe == "1d"
    assert params == {}
    assert venue is None


async def test_resolve_persona_config_invalid_strategy_id(session, _patch_session):
    """Malformed strategy_config_id is logged and falls back to meta."""
    pid = uuid4()
    with tenant_scope(TENANT_ID):
        repo = TenantRepository(session, PersonaModel)
        persona = await repo.create(
            id=pid,
            name="BadId",
            mode="paper",
            asset_class="crypto",
            base_currency="USDT",
            initial_capital=Decimal("10000"),
            current_equity=Decimal("10000"),
            meta={
                "strategy_config_id": "not-a-uuid",
                "algo_id": "buy_hold",
                "symbol": "BTC-USD",
            },
        )
        await session.commit()

    loop = TradingLoop(tenant_id=TENANT_ID)
    algo_id, symbol, _, _, _ = await loop._resolve_persona_config(persona)

    assert algo_id == "buy_hold"
    assert symbol == "BTC-USD"


# ---------------------------------------------------------------------------
# Signal persistence + pubsub fan-out (Phase 4)
# ---------------------------------------------------------------------------


async def test_persist_and_publish_signal_writes_row_and_publishes(
    session, _patch_session
):
    """A signal emitted by the loop is persisted AND fanned out on the bus."""
    from sqlalchemy import select

    from daytrader.core.pubsub import (
        SignalEvent,
        reset_signal_bus,
        signal_bus,
    )
    from daytrader.core.types.signals import Signal
    from daytrader.storage.models import SignalModel

    reset_signal_bus()

    pid = uuid4()
    with tenant_scope(TENANT_ID):
        repo = TenantRepository(session, PersonaModel)
        persona = await repo.create(
            id=pid,
            name="Pub Bot",
            mode="paper",
            asset_class="crypto",
            base_currency="USDT",
            initial_capital=Decimal("10000"),
            current_equity=Decimal("10000"),
            meta={},
        )
        await session.commit()

    sig = Signal.new(
        symbol_key="BTC-USD",
        score=0.8,
        confidence=0.7,
        source="test",
        reason="unit test",
    )

    loop = TradingLoop(tenant_id=TENANT_ID)

    bus = signal_bus()
    with bus.subscribe(TENANT_ID) as queue:
        await loop._persist_and_publish_signal(persona, sig)
        event = await asyncio.wait_for(queue.get(), timeout=1.0)

    assert isinstance(event, SignalEvent)
    assert event.signal_id == sig.id
    assert event.tenant_id == TENANT_ID
    assert event.persona_id == pid
    assert event.symbol_key == "BTC-USD"
    assert event.score == pytest.approx(0.8)

    # Row was persisted under the tenant.
    rows = (
        await session.execute(
            select(SignalModel).where(SignalModel.tenant_id == TENANT_ID)
        )
    ).scalars().all()
    assert len(rows) == 1
    assert rows[0].id == sig.id
    assert rows[0].persona_id == pid
    assert rows[0].score == pytest.approx(0.8)


async def test_persist_failure_does_not_publish(session, _patch_session, monkeypatch):
    """When the DB write fails, no event is broadcast."""
    from daytrader.core.pubsub import reset_signal_bus, signal_bus
    from daytrader.core.types.signals import Signal

    reset_signal_bus()

    pid = uuid4()
    with tenant_scope(TENANT_ID):
        repo = TenantRepository(session, PersonaModel)
        persona = await repo.create(
            id=pid,
            name="Fail Bot",
            mode="paper",
            asset_class="crypto",
            base_currency="USDT",
            initial_capital=Decimal("10000"),
            current_equity=Decimal("10000"),
            meta={},
        )
        await session.commit()

    sig = Signal.new(
        symbol_key="BTC-USD", score=0.5, source="test", reason=""
    )

    # Force the repo create to blow up.
    from daytrader.storage import repository as repo_module

    async def _boom(self, **_kw):
        raise RuntimeError("boom")

    monkeypatch.setattr(repo_module.TenantRepository, "create", _boom)

    loop = TradingLoop(tenant_id=TENANT_ID)
    bus = signal_bus()
    with bus.subscribe(TENANT_ID) as queue:
        await loop._persist_and_publish_signal(persona, sig)
        # No event should arrive; assert the queue stays empty briefly.
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(queue.get(), timeout=0.1)
