"""Phase 5 — promotion + feature hydration + feature_threshold algo."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

import polars as pl
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker

from daytrader.algorithms.builtin.feature_threshold import (
    FeatureThresholdAlgorithm,
)
from daytrader.core.context import AlgorithmContext
from daytrader.core.types.bars import Bar, Timeframe
from daytrader.core.types.symbols import Symbol
from daytrader.research.feature_hydration import (
    FeatureHydrator,
    reset_feature_hydrator,
)
from daytrader.research.promotion import (
    PromotionError,
    promote_discovery,
)
from daytrader.storage.models import (
    DiscoveryModel,
    StrategyConfigModel,
    TenantModel,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def session_factory(engine):
    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield factory


@pytest_asyncio.fixture
async def patched_get_session(session_factory, monkeypatch):
    """Patch get_session() so promote_discovery uses our in-memory engine.

    Modules grab ``get_session`` via ``from ..storage.database import
    get_session`` so we have to patch the *imported name* on each
    consumer, not the source module.
    """
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _fake_get_session():
        async with session_factory() as s:
            yield s

    monkeypatch.setattr(
        "daytrader.research.promotion.get_session", _fake_get_session,
    )
    yield


@pytest_asyncio.fixture
async def tenant_a(session_factory):
    tid = uuid4()
    async with session_factory() as s:
        s.add(TenantModel(id=tid, name="tenant_a"))
        await s.commit()
    return tid


@pytest_asyncio.fixture
async def tenant_b(session_factory):
    tid = uuid4()
    async with session_factory() as s:
        s.add(TenantModel(id=tid, name="tenant_b"))
        await s.commit()
    return tid


async def _make_discovery(
    session_factory,
    tenant_id: UUID,
    *,
    candidate_name: str = "fred:DGS10",
    candidate_source: str = "fred",
    target_symbol: str = "BTC/USDT",
    target_timeframe: str = "1d",
    significant: bool = True,
    status: str = "new",
    meta: dict | None = None,
) -> UUID:
    did = uuid4()
    async with session_factory() as s:
        s.add(DiscoveryModel(
            id=did,
            tenant_id=tenant_id,
            candidate_name=candidate_name,
            candidate_source=candidate_source,
            target_symbol=target_symbol,
            target_timeframe=target_timeframe,
            baseline_metric=0.5,
            candidate_metric=0.6,
            lift=0.1,
            p_value=0.01,
            q_value=0.05,
            significant=significant,
            n_folds=5,
            status=status,
            meta=meta or {"series_id": "DGS10"},
        ))
        await s.commit()
    return did


# ---------------------------------------------------------------------------
# promote_discovery
# ---------------------------------------------------------------------------


async def test_promote_creates_strategy_config(
    patched_get_session, session_factory, tenant_a
):
    did = await _make_discovery(session_factory, tenant_a)

    result = await promote_discovery(tenant_id=tenant_a, discovery_id=did)

    assert result.discovery_id == did
    assert result.algo_id == "feature_threshold"
    assert result.feature_name == "fred:DGS10"

    # Strategy was actually persisted under the tenant.
    async with session_factory() as s:
        from sqlalchemy import select

        rows = (await s.execute(
            select(StrategyConfigModel).where(
                StrategyConfigModel.id == result.strategy_config_id,
            )
        )).scalars().all()
    assert len(rows) == 1
    sc = rows[0]
    assert sc.tenant_id == tenant_a
    assert sc.algo_id == "feature_threshold"
    assert sc.symbol == "BTC/USDT"
    assert sc.timeframe == "1d"
    assert sc.algo_params["feature_name"] == "fred:DGS10"
    assert sc.source_discovery_id == did


async def test_promote_flips_discovery_status_and_records_strategy_id(
    patched_get_session, session_factory, tenant_a
):
    did = await _make_discovery(session_factory, tenant_a)

    result = await promote_discovery(tenant_id=tenant_a, discovery_id=did)

    async with session_factory() as s:
        from sqlalchemy import select

        disc = (await s.execute(
            select(DiscoveryModel).where(DiscoveryModel.id == did)
        )).scalar_one()
    assert disc.status == "promoted"
    assert disc.meta["strategy_config_id"] == str(result.strategy_config_id)


async def test_promote_rejects_already_promoted(
    patched_get_session, session_factory, tenant_a
):
    did = await _make_discovery(
        session_factory, tenant_a, status="promoted"
    )

    with pytest.raises(PromotionError, match="already promoted"):
        await promote_discovery(tenant_id=tenant_a, discovery_id=did)


async def test_promote_rejects_unknown_id(
    patched_get_session, session_factory, tenant_a
):
    with pytest.raises(PromotionError, match="not found"):
        await promote_discovery(tenant_id=tenant_a, discovery_id=uuid4())


async def test_promote_refuses_cross_tenant(
    patched_get_session, session_factory, tenant_a, tenant_b
):
    """Tenant A cannot promote tenant B's discovery."""
    did = await _make_discovery(session_factory, tenant_b)

    with pytest.raises(PromotionError, match="not found"):
        await promote_discovery(tenant_id=tenant_a, discovery_id=did)


async def test_promote_cross_asset_uses_both_directions(
    patched_get_session, session_factory, tenant_a
):
    """Cross-asset discoveries default to ``direction=both`` since
    another asset's price can be informative in either direction."""
    did = await _make_discovery(
        session_factory, tenant_a,
        candidate_name="cross:SPY",
        candidate_source="cross_asset",
        meta={"symbol": "SPY"},
    )

    result = await promote_discovery(tenant_id=tenant_a, discovery_id=did)

    async with session_factory() as s:
        from sqlalchemy import select

        sc = (await s.execute(
            select(StrategyConfigModel).where(
                StrategyConfigModel.id == result.strategy_config_id,
            )
        )).scalar_one()
    assert sc.algo_params["direction"] == "both"


# ---------------------------------------------------------------------------
# feature_threshold algorithm
# ---------------------------------------------------------------------------


def _ctx_with_features(features: dict[str, float], params: dict) -> AlgorithmContext:
    emitted: list = []
    return AlgorithmContext(
        tenant_id=uuid4(),
        persona_id=uuid4(),
        algorithm_id="feature_threshold",
        symbol=Symbol.parse("BTC/USDT"),
        timeframe=Timeframe.D1,
        now=datetime(2026, 4, 25, tzinfo=UTC),
        bar=Bar(
            timestamp=datetime(2026, 4, 25, tzinfo=UTC),
            open=1, high=1, low=1, close=1, volume=1,
        ),
        history_arrays={},
        features=features,
        params=params,
        emit_fn=emitted.append,
        log_fn=lambda *a, **k: None,
    ), emitted


def test_feature_threshold_emits_long_above_upper():
    algo = FeatureThresholdAlgorithm()
    ctx, _emitted = _ctx_with_features(
        features={"fred:DGS10": 0.05},
        params={"feature_name": "fred:DGS10", "upper_threshold": 0.0,
                "lower_threshold": -0.05, "direction": "both"},
    )
    sig = algo.on_bar(ctx)
    assert sig is not None
    assert sig.score == 1.0
    assert "fred:DGS10" in sig.reason


def test_feature_threshold_emits_short_below_lower():
    algo = FeatureThresholdAlgorithm()
    ctx, _emitted = _ctx_with_features(
        features={"x": -0.5},
        params={"feature_name": "x", "upper_threshold": 0.1,
                "lower_threshold": -0.1, "direction": "both"},
    )
    sig = algo.on_bar(ctx)
    assert sig is not None
    assert sig.score == -1.0


def test_feature_threshold_long_only_blocks_short():
    algo = FeatureThresholdAlgorithm()
    ctx, _emitted = _ctx_with_features(
        features={"x": -0.5},
        params={"feature_name": "x", "upper_threshold": 0.1,
                "lower_threshold": -0.1, "direction": "long_only"},
    )
    assert algo.on_bar(ctx) is None


def test_feature_threshold_returns_none_when_feature_missing():
    algo = FeatureThresholdAlgorithm()
    ctx, _emitted = _ctx_with_features(
        features={},  # hydration failed → empty
        params={"feature_name": "fred:DGS10", "direction": "long_only"},
    )
    assert algo.on_bar(ctx) is None


def test_feature_threshold_returns_none_when_feature_name_blank():
    algo = FeatureThresholdAlgorithm()
    ctx, _emitted = _ctx_with_features(
        features={"x": 1.0},
        params={"feature_name": ""},  # misconfigured config
    )
    assert algo.on_bar(ctx) is None


def test_feature_threshold_registered_in_auto_register():
    from daytrader.algorithms.registry import AlgorithmRegistry

    AlgorithmRegistry.clear()
    AlgorithmRegistry.auto_register()
    assert "feature_threshold" in AlgorithmRegistry.available()


# ---------------------------------------------------------------------------
# FeatureHydrator
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_hydrator():
    reset_feature_hydrator()
    yield
    reset_feature_hydrator()


class _FakeDiscovery:
    """Minimal duck-typed Discovery for hydrator unit tests."""

    def __init__(self, source: str, meta: dict, name: str = "test") -> None:
        self.id = uuid4()
        self.candidate_source = source
        self.candidate_name = name
        self.meta = meta


async def test_hydrator_returns_none_for_unknown_source():
    hydrator = FeatureHydrator()
    disc = _FakeDiscovery(source="unknown_thing", meta={})
    assert await hydrator.hydrate(disc) is None


async def test_hydrator_returns_none_when_meta_missing_key():
    hydrator = FeatureHydrator()
    # FRED with no series_id
    assert await hydrator.hydrate(
        _FakeDiscovery(source="fred", meta={})
    ) is None
    # sentiment with no query
    assert await hydrator.hydrate(
        _FakeDiscovery(source="sentiment", meta={})
    ) is None
    # cross_asset with no symbol
    assert await hydrator.hydrate(
        _FakeDiscovery(source="cross_asset", meta={})
    ) is None


async def test_hydrator_returns_none_when_adapter_unavailable():
    """If FRED adapter not registered, hydration returns None gracefully."""
    from daytrader.data.macro.base import MacroAdapterRegistry

    MacroAdapterRegistry.clear()  # ensure FRED not registered
    hydrator = FeatureHydrator()
    disc = _FakeDiscovery(source="fred", meta={"series_id": "DGS10"})
    assert await hydrator.hydrate(disc) is None


async def test_hydrator_swallows_adapter_errors():
    """Upstream API failure → None, not propagated exception."""
    hydrator = FeatureHydrator()
    fake_adapter = AsyncMock()
    fake_adapter.fetch_series.side_effect = RuntimeError("FRED is down")

    with patch(
        "daytrader.data.macro.base.MacroAdapterRegistry.available",
        return_value=["fred"],
    ), patch(
        "daytrader.data.macro.base.MacroAdapterRegistry.get",
        return_value=fake_adapter,
    ):
        disc = _FakeDiscovery(source="fred", meta={"series_id": "DGS10"})
        assert await hydrator.hydrate(disc) is None


async def test_hydrator_returns_latest_fred_value():
    hydrator = FeatureHydrator()
    fake_adapter = AsyncMock()
    fake_adapter.fetch_series.return_value = pl.DataFrame({
        "timestamp": [
            datetime(2026, 4, 23, tzinfo=UTC),
            datetime(2026, 4, 24, tzinfo=UTC),
        ],
        "value": [4.10, 4.25],
    })

    with patch(
        "daytrader.data.macro.base.MacroAdapterRegistry.available",
        return_value=["fred"],
    ), patch(
        "daytrader.data.macro.base.MacroAdapterRegistry.get",
        return_value=fake_adapter,
    ):
        disc = _FakeDiscovery(source="fred", meta={"series_id": "DGS10"})
        value = await hydrator.hydrate(disc)
    assert value == 4.25


async def test_hydrator_caches_repeat_calls():
    """Second call within TTL must not re-hit upstream."""
    hydrator = FeatureHydrator(ttl_seconds=60.0)
    fake_adapter = AsyncMock()
    fake_adapter.fetch_series.return_value = pl.DataFrame({
        "timestamp": [datetime(2026, 4, 24, tzinfo=UTC)],
        "value": [3.5],
    })

    with patch(
        "daytrader.data.macro.base.MacroAdapterRegistry.available",
        return_value=["fred"],
    ), patch(
        "daytrader.data.macro.base.MacroAdapterRegistry.get",
        return_value=fake_adapter,
    ):
        disc = _FakeDiscovery(source="fred", meta={"series_id": "DGS10"})
        v1 = await hydrator.hydrate(disc)
        v2 = await hydrator.hydrate(disc)

    assert v1 == v2 == 3.5
    assert fake_adapter.fetch_series.call_count == 1


async def test_hydrator_sentiment_averages_event_scores():
    from daytrader.data.sentiment.base import NewsEvent

    hydrator = FeatureHydrator()
    events = [
        NewsEvent(
            timestamp=datetime(2026, 4, 24, tzinfo=UTC),
            source="x", title="t1", sentiment_score=0.4,
        ),
        NewsEvent(
            timestamp=datetime(2026, 4, 24, tzinfo=UTC),
            source="x", title="t2", sentiment_score=0.6,
        ),
        NewsEvent(
            timestamp=datetime(2026, 4, 24, tzinfo=UTC),
            source="x", title="t3", sentiment_score=None,  # ignored
        ),
    ]
    fake_adapter = AsyncMock()
    fake_adapter.fetch_news.return_value = events

    with patch(
        "daytrader.data.sentiment.base.SentimentAdapterRegistry.available",
        return_value=["newsapi"],
    ), patch(
        "daytrader.data.sentiment.base.SentimentAdapterRegistry.get",
        return_value=fake_adapter,
    ):
        disc = _FakeDiscovery(
            source="sentiment", meta={"query": "bitcoin"}
        )
        value = await hydrator.hydrate(disc)
    assert value == pytest.approx(0.5)


async def test_hydrator_sentiment_returns_none_when_no_scored_events():
    from daytrader.data.sentiment.base import NewsEvent

    hydrator = FeatureHydrator()
    fake_adapter = AsyncMock()
    fake_adapter.fetch_news.return_value = [
        NewsEvent(
            timestamp=datetime(2026, 4, 24, tzinfo=UTC),
            source="x", title="t", sentiment_score=None,
        ),
    ]

    with patch(
        "daytrader.data.sentiment.base.SentimentAdapterRegistry.available",
        return_value=["newsapi"],
    ), patch(
        "daytrader.data.sentiment.base.SentimentAdapterRegistry.get",
        return_value=fake_adapter,
    ):
        disc = _FakeDiscovery(source="sentiment", meta={"query": "btc"})
        value = await hydrator.hydrate(disc)
    assert value is None
