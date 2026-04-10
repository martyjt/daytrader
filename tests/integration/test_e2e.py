"""End-to-end acceptance test for Phase 0.

Proves the complete Ritual flow without network or Docker:
    Create persona → Backtest with real engine → Verify KPIs → Promote to paper

Uses SQLite in-memory and synthetic OHLCV data.
"""

from datetime import datetime, timedelta
from decimal import Decimal
from uuid import uuid4

import polars as pl
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from daytrader.algorithms.registry import AlgorithmRegistry
from daytrader.backtest.engine import BacktestEngine
from daytrader.core.context import tenant_scope
from daytrader.core.types.bars import Timeframe
from daytrader.core.types.symbols import AssetClass, Symbol
from daytrader.storage.database import Base
from daytrader.storage.models import PersonaModel, TenantModel
from daytrader.storage.repository import TenantRepository


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
        yield s


@pytest_asyncio.fixture
async def tenant_id(session):
    tid = uuid4()
    session.add(TenantModel(id=tid, name="e2e-test"))
    await session.commit()
    return tid


def _rising_market(n: int = 30) -> pl.DataFrame:
    """30 bars of steadily rising prices."""
    return pl.DataFrame(
        {
            "timestamp": [
                datetime(2024, 1, 1) + timedelta(days=i) for i in range(n)
            ],
            "open": [100.0 + i * 0.5 for i in range(n)],
            "high": [101.0 + i * 0.5 for i in range(n)],
            "low": [99.0 + i * 0.5 for i in range(n)],
            "close": [100.0 + i * 0.5 for i in range(n)],
            "volume": [1000.0] * n,
        }
    )


async def test_full_ritual_flow(session, tenant_id):
    """End-to-end: create persona → backtest → verify → promote to paper."""

    # 1. Register algorithms
    AlgorithmRegistry.clear()
    AlgorithmRegistry.auto_register()
    assert "buy_hold" in AlgorithmRegistry.available()

    # 2. Create a persona in backtest mode
    repo = TenantRepository(session, PersonaModel)
    with tenant_scope(tenant_id):
        persona = await repo.create(
            name="E2E Test Bot",
            mode="backtest",
            asset_class="crypto",
            base_currency="USD",
            initial_capital=Decimal("10000"),
            current_equity=Decimal("10000"),
            risk_profile="balanced",
        )
        await session.commit()
        persona_id = persona.id

    # 3. Run a backtest with the buy_hold algorithm on synthetic data
    bt_engine = BacktestEngine()
    result = await bt_engine.run(
        algorithm=AlgorithmRegistry.get("buy_hold"),
        symbol=Symbol("BTC", "USD", AssetClass.CRYPTO),
        timeframe=Timeframe.D1,
        start=datetime(2024, 1, 1),
        end=datetime(2024, 1, 30),
        initial_capital=10_000.0,
        data=_rising_market(30),
    )

    # 4. Verify backtest produced meaningful results
    assert len(result.equity_curve) == 30
    assert result.final_equity > result.initial_capital
    assert result.kpis["total_return_pct"] > 0
    assert result.kpis["num_trades"] >= 1
    assert len(result.signals) == 30  # buy_hold emits on every bar

    # 5. Promote to paper — update mode and equity
    with tenant_scope(tenant_id):
        promoted = await repo.update(
            persona_id,
            mode="paper",
            current_equity=Decimal(str(round(result.final_equity, 2))),
        )
        await session.commit()

    # 6. Verify the persona is now in paper mode with updated equity
    with tenant_scope(tenant_id):
        final = await repo.get(persona_id)
        assert final is not None
        assert final.mode == "paper"
        assert final.name == "E2E Test Bot"
        assert float(final.current_equity) > 10_000.0


async def test_backtest_with_declining_market(session, tenant_id):
    """Buy & hold on a declining market should lose money."""
    AlgorithmRegistry.clear()
    AlgorithmRegistry.auto_register()

    declining = pl.DataFrame(
        {
            "timestamp": [
                datetime(2024, 1, 1) + timedelta(days=i) for i in range(20)
            ],
            "open": [100.0 - i * 0.5 for i in range(20)],
            "high": [101.0 - i * 0.5 for i in range(20)],
            "low": [99.0 - i * 0.5 for i in range(20)],
            "close": [100.0 - i * 0.5 for i in range(20)],
            "volume": [1000.0] * 20,
        }
    )

    engine = BacktestEngine()
    result = await engine.run(
        algorithm=AlgorithmRegistry.get("buy_hold"),
        symbol=Symbol("BTC", "USD", AssetClass.CRYPTO),
        timeframe=Timeframe.D1,
        start=datetime(2024, 1, 1),
        end=datetime(2024, 1, 20),
        data=declining,
    )

    assert result.final_equity < result.initial_capital
    assert result.kpis["total_return_pct"] < 0
    assert result.kpis["max_drawdown_pct"] < 0


async def test_multiple_personas_isolated(session, tenant_id):
    """Two personas in the same tenant are independent."""
    repo = TenantRepository(session, PersonaModel)

    with tenant_scope(tenant_id):
        p1 = await repo.create(
            name="Bot A",
            mode="paper",
            asset_class="crypto",
            base_currency="USD",
            initial_capital=Decimal("5000"),
            current_equity=Decimal("5000"),
            risk_profile="conservative",
        )
        p2 = await repo.create(
            name="Bot B",
            mode="backtest",
            asset_class="equities",
            base_currency="USD",
            initial_capital=Decimal("20000"),
            current_equity=Decimal("20000"),
            risk_profile="aggressive",
        )
        await session.commit()

        all_personas = await repo.get_all()
        assert len(all_personas) == 2

        # Each has independent state
        names = {p.name for p in all_personas}
        assert names == {"Bot A", "Bot B"}
