"""Shadow tournament — candidate algorithms race a primary on the same data.

Every live (or backtest) strategy can be mirrored by a **shadow DAG**
of candidate algorithms trading the same symbol/timeframe/window in
paper mode. After a walk-forward the candidates that beat the primary
on both Sharpe and stability are flagged for promotion.

This is intentionally separate from Strategy Lab's ``Run Walk-Forward``:
* Strategy Lab runs one algo at a time (diagnostic).
* Shadow tournament runs N candidates side-by-side and scores each
  against the same primary-incumbent baseline (competitive).

Persists one ``ShadowRunModel`` row per candidate per tournament so the
promotion decision has an auditable history.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

import numpy as np


@dataclass
class ShadowCandidate:
    """One candidate's aggregated tournament result."""

    algo_id: str
    algo_name: str
    sharpe: float = 0.0
    net_return_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    num_trades: int = 0
    stability_score: float = 0.0  # fraction of WF folds beating primary
    beat_primary: bool = False
    is_primary: bool = False
    oos_equity_curve: list[float] = field(default_factory=list)
    error: str | None = None


@dataclass
class ShadowTournamentReport:
    """Complete tournament result — primary + candidates with winner flags."""

    tournament_id: UUID
    target_symbol: str
    target_timeframe: str
    window_start: datetime
    window_end: datetime
    primary_algo_id: str
    candidates: list[ShadowCandidate] = field(default_factory=list)
    winners: list[str] = field(default_factory=list)

    def ranked(self) -> list[ShadowCandidate]:
        """Candidates sorted by Sharpe descending (primary always first)."""
        primary = [c for c in self.candidates if c.is_primary]
        others = sorted(
            (c for c in self.candidates if not c.is_primary),
            key=lambda c: c.sharpe,
            reverse=True,
        )
        return primary + others


async def run_shadow_tournament(
    *,
    tenant_id: UUID,
    primary_algo_id: str,
    candidate_algo_ids: list[str],
    symbol_str: str,
    timeframe_str: str,
    start: datetime,
    end: datetime,
    initial_capital: float = 10_000.0,
    venue: str = "binance_spot",
    n_folds: int = 5,
    persist: bool = True,
) -> ShadowTournamentReport:
    """Run one tournament: walk-forward each algo on the same window.

    The primary's OOS Sharpe defines the bar to beat. For each fold, we
    compare the candidate's OOS Sharpe to the primary's; ``stability_score``
    is the fraction of folds the candidate won. ``beat_primary`` requires
    BOTH higher aggregate Sharpe AND ≥50% stability.
    """
    from datetime import datetime as _dt

    from ..algorithms.registry import AlgorithmRegistry
    from ..backtest.risk import RiskConfig
    from ..backtest.walk_forward import WalkForwardConfig, WalkForwardEngine
    from ..core.context import tenant_scope
    from ..core.types.bars import Timeframe
    from ..core.types.symbols import Symbol
    from ..data.adapters.registry import AdapterRegistry
    from ..storage.database import get_session
    from ..storage.models import ShadowRunModel

    symbol = Symbol.parse(symbol_str)
    timeframe = Timeframe(timeframe_str)
    tournament_id = uuid4()

    # Shared OHLCV data — every candidate races on identical bars. Reuse
    # the backtest engine's Parquet cache so repeat tournaments on the
    # same window are instant.
    from ..backtest.engine import _fetch_ohlcv_cached
    from ..core.types.symbols import AssetClass

    AdapterRegistry.auto_register()
    adapter_name = (
        "binance_public"
        if symbol.asset_class == AssetClass.CRYPTO
        and "binance_public" in AdapterRegistry.available()
        else "yfinance"
    )
    adapter = AdapterRegistry.get(adapter_name)
    data = await _fetch_ohlcv_cached(adapter, symbol, timeframe, start, end)
    if data.is_empty():
        return ShadowTournamentReport(
            tournament_id=tournament_id,
            target_symbol=symbol.key,
            target_timeframe=timeframe.value,
            window_start=start,
            window_end=end,
            primary_algo_id=primary_algo_id,
        )

    # Deduplicate candidates, keep primary at top.
    all_ids: list[str] = [primary_algo_id]
    for aid in candidate_algo_ids:
        if aid != primary_algo_id and aid not in all_ids:
            all_ids.append(aid)

    config = WalkForwardConfig(n_folds=n_folds)
    engine = WalkForwardEngine()
    risk_config = RiskConfig.disabled()

    primary_per_fold_sharpe: list[float] = []
    candidates_raw: list[tuple[str, Any]] = []

    for aid in all_ids:
        try:
            algorithm = AlgorithmRegistry.get(aid)
            wf = await engine.run(
                algorithm=algorithm,
                symbol=symbol,
                timeframe=timeframe,
                data=data,
                config=config,
                initial_capital=initial_capital,
                venue=venue,
                risk_config=risk_config,
            )
        except Exception as exc:  # noqa: BLE001
            candidates_raw.append((aid, exc))
            continue
        candidates_raw.append((aid, wf))
        if aid == primary_algo_id:
            primary_per_fold_sharpe = [f.oos_sharpe for f in wf.folds]

    report = ShadowTournamentReport(
        tournament_id=tournament_id,
        target_symbol=symbol.key,
        target_timeframe=timeframe.value,
        window_start=start,
        window_end=end,
        primary_algo_id=primary_algo_id,
    )

    # Score each candidate against the primary baseline.
    for aid, result in candidates_raw:
        algo_name = aid
        try:
            algo_name = AlgorithmRegistry.get(aid).manifest.name
        except Exception:
            pass
        is_primary = aid == primary_algo_id
        if isinstance(result, Exception):
            report.candidates.append(ShadowCandidate(
                algo_id=aid,
                algo_name=algo_name,
                is_primary=is_primary,
                error=str(result),
            ))
            continue

        wf = result
        cand_folds = [f.oos_sharpe for f in wf.folds]
        if is_primary or not primary_per_fold_sharpe:
            stability = 1.0 if is_primary else 0.0
        else:
            # Fraction of matched folds where candidate beat the primary.
            pairs = list(zip(cand_folds, primary_per_fold_sharpe))
            if not pairs:
                stability = 0.0
            else:
                wins = sum(1 for c, p in pairs if c > p)
                stability = wins / len(pairs)

        cand = ShadowCandidate(
            algo_id=aid,
            algo_name=algo_name,
            sharpe=float(wf.aggregate_oos_sharpe),
            net_return_pct=float(wf.aggregate_oos_return_pct),
            max_drawdown_pct=float(wf.aggregate_oos_max_drawdown_pct),
            num_trades=int(
                sum(getattr(f, "num_trades", 0) for f in wf.folds)
            ),
            stability_score=float(stability),
            is_primary=is_primary,
            oos_equity_curve=list(wf.oos_equity_curve or []),
        )

        # Promotion eligibility: Sharpe beats primary AND >=50% fold stability.
        primary_cand = next((c for c in report.candidates if c.is_primary), None)
        if not is_primary and primary_cand is not None:
            cand.beat_primary = bool(
                cand.sharpe > primary_cand.sharpe and cand.stability_score >= 0.5
            )
        report.candidates.append(cand)

    report.winners = [c.algo_id for c in report.candidates if c.beat_primary]

    # ---- Persist to shadow_runs table --------------------------------
    if persist:
        async with get_session() as session:
            with tenant_scope(tenant_id):
                for cand in report.candidates:
                    row = ShadowRunModel(
                        tenant_id=tenant_id,
                        tournament_id=tournament_id,
                        primary_algo_id=primary_algo_id,
                        candidate_algo_id=cand.algo_id,
                        target_symbol=symbol.key,
                        target_timeframe=timeframe.value,
                        window_start=start,
                        window_end=end,
                        sharpe=cand.sharpe,
                        net_return_pct=cand.net_return_pct,
                        max_drawdown_pct=cand.max_drawdown_pct,
                        num_trades=cand.num_trades,
                        stability_score=cand.stability_score,
                        is_primary=cand.is_primary,
                        beat_primary=cand.beat_primary,
                        promotion_status="pending",
                        meta={
                            "algo_name": cand.algo_name,
                            "error": cand.error,
                        },
                    )
                    session.add(row)
                await session.commit()

    # Fire an alert if any candidate beat the primary — actionable insight.
    if report.winners:
        try:
            from ..ui.alerts import alerts as _alerts

            winner_names = [
                next((c.algo_name for c in report.candidates if c.algo_id == w), w)
                for w in report.winners
            ]
            _alerts().add(
                level="info",
                title=f"Shadow tournament: {len(report.winners)} candidate(s) beat the primary",
                body=(
                    f"On {symbol.key} {timeframe.value}: "
                    f"{', '.join(winner_names[:3])}"
                    f"{' (+more)' if len(winner_names) > 3 else ''} "
                    f"beat {primary_algo_id}. Review in Research Lab → Shadow."
                ),
                source="shadow",
                data={
                    "tournament_id": str(tournament_id),
                    "winners": report.winners,
                    "primary": primary_algo_id,
                },
            )
        except Exception:
            pass

    return report


async def list_tournaments(
    *,
    tenant_id: UUID,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Summarize recent tournaments for the Shadow tab table."""
    from sqlalchemy import select

    from ..core.context import tenant_scope
    from ..storage.database import get_session
    from ..storage.models import ShadowRunModel

    async with get_session() as session:
        with tenant_scope(tenant_id):
            rows = (await session.execute(
                select(ShadowRunModel)
                .where(ShadowRunModel.tenant_id == tenant_id)
                .order_by(ShadowRunModel.created_at.desc())
                .limit(limit * 10)  # over-fetch so we can group
            )).scalars().all()

    # Group by tournament_id
    tournaments: dict[UUID, dict[str, Any]] = {}
    for r in rows:
        t = tournaments.setdefault(r.tournament_id, {
            "tournament_id": str(r.tournament_id),
            "created_at": r.created_at,
            "symbol": r.target_symbol,
            "timeframe": r.target_timeframe,
            "primary": r.primary_algo_id,
            "n_candidates": 0,
            "n_winners": 0,
            "candidates": [],
        })
        t["n_candidates"] += 0 if r.is_primary else 1
        if r.beat_primary:
            t["n_winners"] += 1
        t["candidates"].append({
            "algo_id": r.candidate_algo_id,
            "is_primary": r.is_primary,
            "sharpe": r.sharpe,
            "net_return_pct": r.net_return_pct,
            "stability": r.stability_score,
            "beat_primary": r.beat_primary,
            "status": r.promotion_status,
        })
    return sorted(
        tournaments.values(),
        key=lambda t: t["created_at"],
        reverse=True,
    )[:limit]
