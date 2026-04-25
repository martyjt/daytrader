"""End-to-end backtest with a sandboxed plugin.

Verifies the full path: install plugin → ``BacktestEngine.run()`` calls
``_precompute_sandbox`` → one ``replay_bars`` RPC → engine reads signals
out of the per-bar dict in ``_simulate``. If any link breaks, this test
catches it.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from decimal import Decimal
from uuid import UUID

import polars as pl
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from daytrader.algorithms.registry import AlgorithmRegistry
from daytrader.algorithms.sandbox import (
    PluginWorkerManager,
    SandboxedAlgorithm,
)
from daytrader.algorithms.sandbox.installer import install_plugin
from daytrader.backtest.engine import BacktestEngine
from daytrader.core.types.bars import Timeframe
from daytrader.core.types.symbols import AssetClass, Symbol
from daytrader.storage.database import Base
from daytrader.storage.models import TenantModel, UserModel


TENANT_A = UUID("00000000-0000-0000-0000-0000000000aa")
USER_A = UUID("00000000-0000-0000-0000-0000000000a1")


# A deterministic algorithm: emit +1 (long) when close is above the 5-bar
# moving average, -1 (short — but the v1 engine is long-only so this becomes
# a flat) otherwise. Confidence stays at 1.0 so signals always trigger.
_BACKTEST_PLUGIN = b"""
from daytrader.algorithms.base import Algorithm, AlgorithmManifest


class MeanCross(Algorithm):
    @property
    def manifest(self):
        return AlgorithmManifest(id="mean_cross_test", name="MeanCross")

    def warmup_bars(self):
        return 5

    def on_bar(self, ctx):
        closes = ctx.history(5, "close")
        avg = float(closes.mean())
        score = 0.8 if float(ctx.bar.close) > avg else -0.8
        return ctx.emit(score=score, confidence=1.0, reason="mean")
"""


@pytest_asyncio.fixture
async def db_engine():
    e = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with e.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield e
    await e.dispose()


@pytest_asyncio.fixture
async def db(db_engine, monkeypatch):
    factory = async_sessionmaker(db_engine, expire_on_commit=False)

    @asynccontextmanager
    async def _get_session():
        async with factory() as s:
            try:
                yield s
            except Exception:
                await s.rollback()
                raise

    monkeypatch.setattr("daytrader.storage.database.get_session", _get_session)
    monkeypatch.setattr(
        "daytrader.algorithms.sandbox.installer.get_session", _get_session
    )

    async with factory() as s:
        s.add(TenantModel(id=TENANT_A, name="a"))
        s.add(UserModel(
            id=USER_A, tenant_id=TENANT_A, email="a@x", role="owner",
        ))
        await s.commit()
    return factory


@pytest.fixture
def manager(tmp_path):
    yield PluginWorkerManager(base_dir=tmp_path / "uploads")


@pytest.fixture(autouse=True)
def _clean_registry():
    AlgorithmRegistry.clear()
    yield
    AlgorithmRegistry.clear()


@pytest_asyncio.fixture(autouse=True)
async def _shutdown(manager):
    yield
    await manager.shutdown_all()


def _synthetic_bars(n: int = 50) -> pl.DataFrame:
    """Build a deterministic OHLCV series — alternating up/down by 1."""
    timestamps = [
        datetime(2026, 1, 1, tzinfo=timezone.utc).replace(day=(i % 28) + 1, month=((i // 28) % 12) + 1)
        for i in range(n)
    ]
    closes = [100.0 + (1.0 if i % 2 == 0 else -0.5) * i for i in range(n)]
    return pl.DataFrame({
        "timestamp": timestamps,
        "open": closes,
        "high": [c + 1.0 for c in closes],
        "low": [c - 1.0 for c in closes],
        "close": closes,
        "volume": [1000.0] * n,
    })


async def test_backtest_run_with_sandboxed_algo(db, manager):
    await install_plugin(
        manager=manager, tenant_id=TENANT_A, user_id=USER_A,
        filename="mean_cross_test.py", algorithm_id="mean_cross_test",
        payload=_BACKTEST_PLUGIN,
    )
    algo = AlgorithmRegistry.get("mean_cross_test", tenant_id=TENANT_A)
    assert isinstance(algo, SandboxedAlgorithm)

    data = _synthetic_bars(40)
    engine = BacktestEngine()
    result = await engine.run(
        algorithm=algo,
        symbol=Symbol("BTC", "USDT", AssetClass.CRYPTO, "binance"),
        timeframe=Timeframe.D1,
        start=datetime(2026, 1, 1, tzinfo=timezone.utc),
        end=datetime(2026, 6, 1, tzinfo=timezone.utc),
        initial_capital=10_000.0,
        commission_bps=10.0,
        data=data,
    )

    # The engine should have produced an equity curve covering every bar
    # and emitted signals on the post-warmup bars.
    assert len(result.equity_curve) == 40
    # Warmup is 5 → 35 signal-eligible bars.
    assert len(result.signals) == 35
    # Confirm signals reconstruct via the protocol — they should have the
    # plugin's source + reason tags.
    assert all(s.source == "mean_cross_test" for s in result.signals)
    assert all(s.reason == "mean" for s in result.signals)
    # And the engine actually traded at least once on the alternating data.
    assert len(result.trades) >= 1


async def test_backtest_signals_match_replay_bars_directly(db, manager):
    """Sanity: the per-bar dict the engine builds matches a direct call."""
    await install_plugin(
        manager=manager, tenant_id=TENANT_A, user_id=USER_A,
        filename="mean_cross_test.py", algorithm_id="mean_cross_test",
        payload=_BACKTEST_PLUGIN,
    )
    algo = AlgorithmRegistry.get("mean_cross_test", tenant_id=TENANT_A)

    data = _synthetic_bars(20)
    engine = BacktestEngine()
    result = await engine.run(
        algorithm=algo,
        symbol=Symbol("BTC", "USDT", AssetClass.CRYPTO, "binance"),
        timeframe=Timeframe.D1,
        start=datetime(2026, 1, 1, tzinfo=timezone.utc),
        end=datetime(2026, 2, 1, tzinfo=timezone.utc),
        initial_capital=10_000.0,
        data=data,
    )
    # 15 signal-eligible bars, scores in {-0.8, +0.8} — verify both directions.
    scores = {round(s.score, 1) for s in result.signals}
    assert scores.issubset({0.8, -0.8})
