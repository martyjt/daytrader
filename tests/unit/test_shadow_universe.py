"""Tests for multi-symbol shadow tournament aggregation.

The per-symbol and basket variants both delegate the actual walk-forward
to ``run_shadow_tournament``. These tests mock that out and lock the
aggregation behavior — wins-per-algo tally, basket mean metrics, and
basket stability scoring.
"""

from __future__ import annotations

from datetime import datetime
from uuid import uuid4

import pytest

from daytrader.research.shadow_trading import (
    ShadowCandidate,
    ShadowTournamentReport,
    UniverseTournamentReport,
    run_basket_tournament,
    run_per_symbol_universe_tournament,
)


def _make_report(
    symbol: str,
    *,
    primary_sharpe: float,
    cand_results: list[tuple[str, float, bool]],
) -> ShadowTournamentReport:
    """Build a fake per-symbol report — primary first, then candidates."""
    candidates = [
        ShadowCandidate(
            algo_id="primary",
            algo_name="Primary",
            sharpe=primary_sharpe,
            net_return_pct=primary_sharpe * 10,
            max_drawdown_pct=-5.0,
            num_trades=10,
            stability_score=1.0,
            is_primary=True,
        ),
    ]
    for aid, sharpe, beat in cand_results:
        candidates.append(ShadowCandidate(
            algo_id=aid,
            algo_name=aid.upper(),
            sharpe=sharpe,
            net_return_pct=sharpe * 10,
            max_drawdown_pct=-3.0,
            num_trades=8,
            stability_score=0.8 if beat else 0.2,
            is_primary=False,
            beat_primary=beat,
        ))
    return ShadowTournamentReport(
        tournament_id=uuid4(),
        target_symbol=symbol,
        target_timeframe="1d",
        window_start=datetime(2024, 1, 1),
        window_end=datetime(2024, 6, 1),
        primary_algo_id="primary",
        candidates=candidates,
        winners=[aid for aid, _, beat in cand_results if beat],
    )


@pytest.mark.asyncio
async def test_per_symbol_loop_runs_once_per_symbol(monkeypatch):
    """run_per_symbol_universe_tournament invokes inner func N times, one per symbol."""
    calls: list[str] = []

    async def fake_inner(*, symbol_str, **kwargs):
        calls.append(symbol_str)
        return _make_report(symbol_str, primary_sharpe=1.0, cand_results=[])

    monkeypatch.setattr(
        "daytrader.research.shadow_trading.run_shadow_tournament", fake_inner,
    )

    bundle = await run_per_symbol_universe_tournament(
        tenant_id=uuid4(),
        primary_algo_id="primary",
        candidate_algo_ids=["a", "b"],
        symbols=["BTC-USD", "ETH-USD", "SOL-USD"],
        universe_name="Top 3",
        timeframe_str="1d",
        start=datetime(2024, 1, 1),
        end=datetime(2024, 6, 1),
    )

    assert calls == ["BTC-USD", "ETH-USD", "SOL-USD"]
    assert isinstance(bundle, UniverseTournamentReport)
    assert bundle.universe_name == "Top 3"
    assert len(bundle.per_symbol_reports) == 3


@pytest.mark.asyncio
async def test_per_symbol_win_counts_tally_across_symbols(monkeypatch):
    """win_counts sums beat_primary flags across symbols, per algo."""
    # Algo A wins on BTC + ETH, loses on SOL. Algo B wins only on SOL.
    plan = {
        "BTC-USD": [("a", 2.0, True), ("b", 0.5, False)],
        "ETH-USD": [("a", 2.5, True), ("b", 0.7, False)],
        "SOL-USD": [("a", 0.3, False), ("b", 1.8, True)],
    }

    async def fake_inner(*, symbol_str, **kwargs):
        return _make_report(symbol_str, primary_sharpe=1.0, cand_results=plan[symbol_str])

    monkeypatch.setattr(
        "daytrader.research.shadow_trading.run_shadow_tournament", fake_inner,
    )

    bundle = await run_per_symbol_universe_tournament(
        tenant_id=uuid4(),
        primary_algo_id="primary",
        candidate_algo_ids=["a", "b"],
        symbols=list(plan),
        universe_name="Top 3",
        timeframe_str="1d",
        start=datetime(2024, 1, 1),
        end=datetime(2024, 6, 1),
    )

    counts = bundle.win_counts()
    assert counts == {"a": 2, "b": 1}


@pytest.mark.asyncio
async def test_basket_mean_metrics_across_symbols(monkeypatch):
    """Basket Sharpe/return/DD = mean across symbols; trades = sum."""
    plan = {
        "BTC-USD": [("a", 2.0, True)],
        "ETH-USD": [("a", 4.0, True)],
    }

    async def fake_inner(*, symbol_str, persist, **kwargs):
        # Basket should set persist=False for the per-symbol calls.
        assert persist is False
        return _make_report(symbol_str, primary_sharpe=1.0, cand_results=plan[symbol_str])

    monkeypatch.setattr(
        "daytrader.research.shadow_trading.run_shadow_tournament", fake_inner,
    )

    # Skip the DB write — we're testing aggregation, not persistence.
    import contextlib
    @contextlib.asynccontextmanager
    async def fake_session():
        class _Sess:
            def add(self, *a, **k): pass
            async def commit(self): pass
        yield _Sess()
    monkeypatch.setattr("daytrader.storage.database.get_session", fake_session)

    report = await run_basket_tournament(
        tenant_id=uuid4(),
        primary_algo_id="primary",
        candidate_algo_ids=["a"],
        symbols=list(plan),
        universe_name="Pair",
        timeframe_str="1d",
        start=datetime(2024, 1, 1),
        end=datetime(2024, 6, 1),
    )

    assert report.target_symbol == "basket"
    cand_a = next(c for c in report.candidates if c.algo_id == "a")
    # Mean of 2.0 and 4.0
    assert cand_a.sharpe == pytest.approx(3.0)
    # net_return_pct = sharpe*10 in fixture; mean of 20 and 40
    assert cand_a.net_return_pct == pytest.approx(30.0)
    # Total trades = 8 + 8
    assert cand_a.num_trades == 16


@pytest.mark.asyncio
async def test_basket_stability_is_fraction_of_symbols_beating_primary(monkeypatch):
    """Basket stability_score = (# symbols where cand sharpe > primary sharpe) / N."""
    # Cand A beats primary on 2 of 3 symbols.
    plan = {
        "BTC-USD": [("a", 2.0, True)],   # a > primary (1.0)
        "ETH-USD": [("a", 1.5, True)],   # a > primary (1.0)
        "SOL-USD": [("a", 0.5, False)],  # a < primary (1.0)
    }

    async def fake_inner(*, symbol_str, persist, **kwargs):
        return _make_report(symbol_str, primary_sharpe=1.0, cand_results=plan[symbol_str])

    monkeypatch.setattr(
        "daytrader.research.shadow_trading.run_shadow_tournament", fake_inner,
    )

    import contextlib
    @contextlib.asynccontextmanager
    async def fake_session():
        class _Sess:
            def add(self, *a, **k): pass
            async def commit(self): pass
        yield _Sess()
    monkeypatch.setattr("daytrader.storage.database.get_session", fake_session)

    report = await run_basket_tournament(
        tenant_id=uuid4(),
        primary_algo_id="primary",
        candidate_algo_ids=["a"],
        symbols=list(plan),
        universe_name="Triple",
        timeframe_str="1d",
        start=datetime(2024, 1, 1),
        end=datetime(2024, 6, 1),
    )

    cand_a = next(c for c in report.candidates if c.algo_id == "a")
    # 2 of 3 beats
    assert cand_a.stability_score == pytest.approx(2 / 3)
    # Mean Sharpe (2 + 1.5 + 0.5)/3 = 1.333... > primary (1.0); stability >= 0.5;
    # so beat_primary should be True
    assert cand_a.beat_primary is True


@pytest.mark.asyncio
async def test_basket_handles_all_errors_gracefully(monkeypatch):
    """If a candidate errors on every symbol, it gets a single error row."""
    plan = {
        "BTC-USD": [],
        "ETH-USD": [],
    }

    async def fake_inner(*, symbol_str, persist, **kwargs):
        # Build a report with primary OK but candidate "a" erroring out.
        rep = _make_report(symbol_str, primary_sharpe=1.0, cand_results=[])
        rep.candidates.append(ShadowCandidate(
            algo_id="a",
            algo_name="A",
            error="boom",
            is_primary=False,
        ))
        return rep

    monkeypatch.setattr(
        "daytrader.research.shadow_trading.run_shadow_tournament", fake_inner,
    )

    import contextlib
    @contextlib.asynccontextmanager
    async def fake_session():
        class _Sess:
            def add(self, *a, **k): pass
            async def commit(self): pass
        yield _Sess()
    monkeypatch.setattr("daytrader.storage.database.get_session", fake_session)

    report = await run_basket_tournament(
        tenant_id=uuid4(),
        primary_algo_id="primary",
        candidate_algo_ids=["a"],
        symbols=list(plan),
        universe_name="Pair",
        timeframe_str="1d",
        start=datetime(2024, 1, 1),
        end=datetime(2024, 6, 1),
    )

    cand_a = next(c for c in report.candidates if c.algo_id == "a")
    assert cand_a.error is not None
    assert cand_a.beat_primary is False
