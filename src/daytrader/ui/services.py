"""Thin service layer for the UI.

Each function manages its own async session and tenant scope so
NiceGUI page handlers don't need to worry about either.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Sequence
from uuid import UUID

from ..core.context import tenant_scope
from ..core.settings import get_settings
from ..storage.database import get_session
from ..storage.models import PersonaModel
from ..storage.repository import TenantRepository


def _tenant_id() -> UUID:
    return get_settings().default_tenant_id


async def list_personas() -> list[Any]:
    async with get_session() as session:
        repo = TenantRepository(session, PersonaModel)
        with tenant_scope(_tenant_id()):
            return list(await repo.get_all())


async def get_persona(persona_id: UUID) -> Any | None:
    async with get_session() as session:
        repo = TenantRepository(session, PersonaModel)
        with tenant_scope(_tenant_id()):
            return await repo.get(persona_id)


async def create_persona(
    *,
    name: str,
    asset_class: str = "crypto",
    base_currency: str = "USDT",
    initial_capital: Decimal = Decimal("10000"),
    current_equity: Decimal | None = None,
    risk_profile: str = "balanced",
    mode: str = "paper",
) -> Any:
    if current_equity is None:
        current_equity = initial_capital
    async with get_session() as session:
        repo = TenantRepository(session, PersonaModel)
        with tenant_scope(_tenant_id()):
            persona = await repo.create(
                name=name,
                mode=mode,
                asset_class=asset_class,
                base_currency=base_currency,
                initial_capital=initial_capital,
                current_equity=current_equity,
                risk_profile=risk_profile,
            )
            await session.commit()
            return persona


async def delete_persona(persona_id: UUID) -> bool:
    async with get_session() as session:
        repo = TenantRepository(session, PersonaModel)
        with tenant_scope(_tenant_id()):
            result = await repo.delete(persona_id)
            await session.commit()
            return result


async def count_personas() -> int:
    async with get_session() as session:
        repo = TenantRepository(session, PersonaModel)
        with tenant_scope(_tenant_id()):
            return await repo.count()


async def promote_to_paper(persona_id: UUID, final_equity: Decimal | None = None) -> Any:
    """Promote a persona from backtest to paper mode."""
    updates: dict[str, Any] = {"mode": "paper"}
    if final_equity is not None:
        updates["current_equity"] = final_equity
    async with get_session() as session:
        repo = TenantRepository(session, PersonaModel)
        with tenant_scope(_tenant_id()):
            persona = await repo.update(persona_id, **updates)
            await session.commit()
            return persona


async def run_backtest_service(
    *,
    algo_id: str,
    symbol_str: str,
    timeframe_str: str,
    start_str: str,
    end_str: str,
    capital: float = 10_000.0,
    venue: str = "binance_spot",
    risk_enabled: bool = False,
    algo_params: dict[str, Any] | None = None,
) -> Any:
    """Run a backtest and return the result.

    Returns a ``BacktestResult`` with equity_curve, trades, signals, kpis,
    total_fees_paid, and venue-specific fee modeling.
    """
    from datetime import datetime as dt

    from ..algorithms.registry import AlgorithmRegistry
    from ..backtest.engine import BacktestEngine
    from ..backtest.risk import RiskConfig
    from ..core.types.bars import Timeframe
    from ..core.types.symbols import Symbol

    algorithm = AlgorithmRegistry.get(algo_id)
    symbol = Symbol.parse(symbol_str)
    timeframe = Timeframe(timeframe_str)
    start = dt.strptime(start_str, "%Y-%m-%d")
    end = dt.strptime(end_str, "%Y-%m-%d")

    risk_config = RiskConfig.from_yaml() if risk_enabled else RiskConfig.disabled()

    engine = BacktestEngine()
    return await engine.run(
        algorithm=algorithm,
        symbol=symbol,
        timeframe=timeframe,
        start=start,
        end=end,
        initial_capital=capital,
        venue=venue,
        risk_config=risk_config,
        params=algo_params,
    )


async def run_walk_forward_service(
    *,
    algo_id: str,
    symbol_str: str,
    timeframe_str: str,
    start_str: str,
    end_str: str,
    capital: float = 10_000.0,
    venue: str = "binance_spot",
    n_folds: int = 5,
    risk_enabled: bool = False,
    algo_params: dict[str, Any] | None = None,
) -> Any:
    """Run walk-forward analysis and return the result."""
    from datetime import datetime as dt

    from ..algorithms.registry import AlgorithmRegistry
    from ..backtest.walk_forward import WalkForwardConfig, WalkForwardEngine
    from ..backtest.risk import RiskConfig
    from ..core.types.bars import Timeframe
    from ..core.types.symbols import Symbol
    from ..data.adapters.registry import AdapterRegistry

    algorithm = AlgorithmRegistry.get(algo_id)
    symbol = Symbol.parse(symbol_str)
    timeframe = Timeframe(timeframe_str)
    start = dt.strptime(start_str, "%Y-%m-%d")
    end = dt.strptime(end_str, "%Y-%m-%d")

    AdapterRegistry.auto_register()
    adapter = AdapterRegistry.get("yfinance")
    data = await adapter.fetch_ohlcv(symbol, timeframe, start, end)

    risk_config = RiskConfig.from_yaml() if risk_enabled else RiskConfig.disabled()

    config = WalkForwardConfig(n_folds=n_folds)
    engine = WalkForwardEngine()
    return await engine.run(
        algorithm=algorithm,
        symbol=symbol,
        timeframe=timeframe,
        data=data,
        config=config,
        initial_capital=capital,
        venue=venue,
        risk_config=risk_config,
    )


async def kill_all_trading(reason: str = "manual") -> int:
    """Activate kill switch, pause all personas. Returns count paused."""
    from ..execution.kill_switch import get_kill_switch

    return await get_kill_switch().activate(_tenant_id(), reason)


async def promote_to_live(
    persona_id: UUID,
    venue: str,
) -> tuple[Any, Any]:
    """Promote a persona from paper to live, enforcing paper gates.

    Returns ``(persona, gate_result)``.
    """
    from ..core.gates import GateEvaluator
    from ..core.types.common import utcnow
    from ..journal.writer import JournalWriter
    from ..storage.models import JournalEntryModel

    async with get_session() as session:
        with tenant_scope(_tenant_id()):
            repo = TenantRepository(session, PersonaModel)
            persona = await repo.get(persona_id)
            if persona is None:
                raise ValueError(f"Persona {persona_id} not found")
            if persona.mode != "paper":
                raise ValueError(
                    f"Persona must be in paper mode to promote (is {persona.mode})"
                )

            # Count trades from journal
            journal_repo = TenantRepository(session, JournalEntryModel)
            trade_entries = list(
                await journal_repo.get_all(
                    persona_id=persona_id, event_type="order_filled"
                )
            )
            num_trades = len(trade_entries)

            # Days active (handle both tz-aware and tz-naive datetimes)
            now = utcnow()
            created = persona.created_at
            if created.tzinfo is None:
                from datetime import timezone

                created = created.replace(tzinfo=timezone.utc)
            days_active = (now - created).days

            # Evaluate gates
            evaluator = GateEvaluator()
            gate_result = evaluator.evaluate_paper(days_active, num_trades)

            if gate_result.overall_pass:
                meta = dict(persona.meta or {})
                meta["venue"] = venue
                await repo.update(
                    persona_id, mode="live", meta=meta
                )
                await session.commit()

                writer = JournalWriter()
                await writer.log_mode_change(
                    _tenant_id(), persona_id, "paper", "live"
                )

            # Re-fetch to get updated state
            persona = await repo.get(persona_id)
            return persona, gate_result


async def list_recent_signals(limit: int = 100) -> list[Any]:
    """Fetch the most recent signal rows (Signal table) across all personas."""
    from sqlalchemy import select

    from ..storage.models import SignalModel

    async with get_session() as session:
        with tenant_scope(_tenant_id()):
            rows = (await session.execute(
                select(SignalModel)
                .where(SignalModel.tenant_id == _tenant_id())
                .order_by(SignalModel.created_at.desc())
                .limit(limit)
            )).scalars().all()
            return list(rows)


async def list_journal_entries(
    *,
    persona_id: UUID | None = None,
    event_type: str | None = None,
    limit: int = 200,
) -> list[Any]:
    """Query journal entries for the current tenant."""
    from ..storage.models import JournalEntryModel
    from sqlalchemy import select

    async with get_session() as session:
        with tenant_scope(_tenant_id()):
            stmt = (
                select(JournalEntryModel)
                .where(JournalEntryModel.tenant_id == _tenant_id())
                .order_by(JournalEntryModel.created_at.desc())
                .limit(limit)
            )
            if persona_id:
                stmt = stmt.where(JournalEntryModel.persona_id == persona_id)
            if event_type:
                stmt = stmt.where(JournalEntryModel.event_type == event_type)
            result = await session.execute(stmt)
            return list(result.scalars().all())


def evaluate_gates_service(
    *,
    backtest_result: Any = None,
    walk_forward_result: Any = None,
) -> list[Any]:
    """Evaluate all applicable promotion gates. Returns list of GateResults."""
    from ..core.gates import GateEvaluator

    evaluator = GateEvaluator()
    results = []

    if backtest_result is not None:
        results.append(evaluator.evaluate_backtest(backtest_result))
    if walk_forward_result is not None:
        results.append(evaluator.evaluate_walk_forward(walk_forward_result))

    return results


# ----------------------------------------------------------------------
# Exploration Agent (Discoveries tab)
# ----------------------------------------------------------------------


async def run_exploration_service(
    *,
    symbol_str: str,
    timeframe_str: str,
    start_str: str,
    end_str: str,
    task: str = "classification",
    n_folds: int = 5,
    fdr_alpha: float = 0.1,
    include_fred: bool = True,
    sentiment_queries: Sequence[str] | None = None,
) -> Any:
    """Trigger one Exploration Agent scan. Returns a ``ScanResult``."""
    from datetime import datetime as dt

    from ..research.exploration_agent import ExplorationAgent, ExplorationConfig

    agent = ExplorationAgent(ExplorationConfig(
        n_folds=n_folds,
        fdr_alpha=fdr_alpha,
        task=task,
        include_fred=include_fred,
        sentiment_queries=list(sentiment_queries or []),
    ))
    return await agent.scan(
        tenant_id=_tenant_id(),
        symbol_str=symbol_str,
        timeframe_str=timeframe_str,
        start=dt.strptime(start_str, "%Y-%m-%d"),
        end=dt.strptime(end_str, "%Y-%m-%d"),
    )


async def list_discoveries(
    *,
    target_symbol: str | None = None,
    significant_only: bool = False,
    limit: int = 200,
) -> list[Any]:
    """List recent discoveries for the current tenant."""
    from sqlalchemy import select

    from ..storage.models import DiscoveryModel

    async with get_session() as session:
        with tenant_scope(_tenant_id()):
            stmt = (
                select(DiscoveryModel)
                .where(DiscoveryModel.tenant_id == _tenant_id())
                .order_by(DiscoveryModel.created_at.desc())
                .limit(limit)
            )
            if target_symbol:
                stmt = stmt.where(DiscoveryModel.target_symbol == target_symbol)
            if significant_only:
                stmt = stmt.where(DiscoveryModel.significant.is_(True))
            result = await session.execute(stmt)
            return list(result.scalars().all())


# ----------------------------------------------------------------------
# Symbol universes + saved strategies
# ----------------------------------------------------------------------


async def list_universes() -> list[Any]:
    from sqlalchemy import select

    from ..storage.models import SymbolUniverseModel

    async with get_session() as session:
        with tenant_scope(_tenant_id()):
            rows = (await session.execute(
                select(SymbolUniverseModel)
                .where(SymbolUniverseModel.tenant_id == _tenant_id())
                .order_by(SymbolUniverseModel.updated_at.desc())
            )).scalars().all()
            return list(rows)


async def save_universe(
    *,
    name: str,
    symbols: Sequence[str],
    description: str = "",
) -> UUID:
    from ..storage.models import SymbolUniverseModel

    async with get_session() as session:
        with tenant_scope(_tenant_id()):
            row = SymbolUniverseModel(
                tenant_id=_tenant_id(),
                name=name.strip(),
                symbols=[s.strip() for s in symbols if s.strip()],
                description=description,
            )
            session.add(row)
            await session.flush()
            rid = row.id
            await session.commit()
            return rid


async def delete_universe(universe_id: UUID) -> bool:
    from sqlalchemy import delete

    from ..storage.models import SymbolUniverseModel

    async with get_session() as session:
        with tenant_scope(_tenant_id()):
            result = await session.execute(
                delete(SymbolUniverseModel)
                .where(SymbolUniverseModel.tenant_id == _tenant_id())
                .where(SymbolUniverseModel.id == universe_id)
            )
            await session.commit()
            return result.rowcount > 0


async def list_strategies() -> list[Any]:
    from sqlalchemy import select

    from ..storage.models import StrategyConfigModel

    async with get_session() as session:
        with tenant_scope(_tenant_id()):
            rows = (await session.execute(
                select(StrategyConfigModel)
                .where(StrategyConfigModel.tenant_id == _tenant_id())
                .order_by(StrategyConfigModel.updated_at.desc())
            )).scalars().all()
            return list(rows)


async def save_strategy_config(
    *,
    name: str,
    algo_id: str,
    symbol: str,
    timeframe: str,
    venue: str = "binance_spot",
    algo_params: dict[str, Any] | None = None,
    description: str = "",
    tags: Sequence[str] | None = None,
) -> UUID:
    from ..storage.models import StrategyConfigModel

    async with get_session() as session:
        with tenant_scope(_tenant_id()):
            row = StrategyConfigModel(
                tenant_id=_tenant_id(),
                name=name.strip(),
                algo_id=algo_id,
                symbol=symbol,
                timeframe=timeframe,
                venue=venue,
                algo_params=dict(algo_params or {}),
                description=description,
                tags=list(tags or []),
            )
            session.add(row)
            await session.flush()
            rid = row.id
            await session.commit()
            return rid


async def delete_strategy_config(strategy_id: UUID) -> bool:
    from sqlalchemy import delete

    from ..storage.models import StrategyConfigModel

    async with get_session() as session:
        with tenant_scope(_tenant_id()):
            result = await session.execute(
                delete(StrategyConfigModel)
                .where(StrategyConfigModel.tenant_id == _tenant_id())
                .where(StrategyConfigModel.id == strategy_id)
            )
            await session.commit()
            return result.rowcount > 0


async def run_portfolio_backtest_service(
    *,
    algo_id: str,
    symbols: Sequence[str],
    timeframe_str: str,
    start_str: str,
    end_str: str,
    total_capital: float = 10_000.0,
    venue: str = "binance_spot",
    algo_params: dict[str, Any] | None = None,
) -> Any:
    """Backtest a single algorithm across a universe of symbols."""
    from datetime import datetime as dt

    from ..backtest.portfolio import run_portfolio_backtest

    return await run_portfolio_backtest(
        algo_id=algo_id,
        symbols=list(symbols),
        timeframe_str=timeframe_str,
        start=dt.strptime(start_str, "%Y-%m-%d"),
        end=dt.strptime(end_str, "%Y-%m-%d"),
        total_capital=total_capital,
        venue=venue,
        algo_params=algo_params,
    )


async def run_shadow_tournament_service(
    *,
    primary_algo_id: str,
    candidate_algo_ids: Sequence[str],
    symbol_str: str,
    timeframe_str: str,
    start_str: str,
    end_str: str,
    initial_capital: float = 10_000.0,
    venue: str = "binance_spot",
    n_folds: int = 5,
) -> Any:
    """Run a shadow tournament and persist results."""
    from datetime import datetime as dt

    from ..research.shadow_trading import run_shadow_tournament

    return await run_shadow_tournament(
        tenant_id=_tenant_id(),
        primary_algo_id=primary_algo_id,
        candidate_algo_ids=list(candidate_algo_ids),
        symbol_str=symbol_str,
        timeframe_str=timeframe_str,
        start=dt.strptime(start_str, "%Y-%m-%d"),
        end=dt.strptime(end_str, "%Y-%m-%d"),
        initial_capital=initial_capital,
        venue=venue,
        n_folds=n_folds,
    )


async def list_shadow_tournaments_service(limit: int = 50) -> list[Any]:
    from ..research.shadow_trading import list_tournaments

    return await list_tournaments(tenant_id=_tenant_id(), limit=limit)


async def update_shadow_status_service(
    tournament_id: UUID,
    candidate_algo_id: str,
    status: str,
) -> int:
    """Update all shadow_runs for (tournament, candidate) to the given status.

    ``status``: ``"promoted" | "dismissed" | "pending"``.
    Returns the number of rows updated.
    """
    from sqlalchemy import update

    from ..storage.models import ShadowRunModel

    if status not in ("promoted", "dismissed", "pending"):
        raise ValueError(f"Invalid status {status!r}")

    async with get_session() as session:
        with tenant_scope(_tenant_id()):
            result = await session.execute(
                update(ShadowRunModel)
                .where(ShadowRunModel.tenant_id == _tenant_id())
                .where(ShadowRunModel.tournament_id == tournament_id)
                .where(ShadowRunModel.candidate_algo_id == candidate_algo_id)
                .values(promotion_status=status)
            )
            await session.commit()

    # Record an alert so the user sees the action in the bell dropdown.
    try:
        from .alerts import alerts as _alerts

        verb = {"promoted": "promoted", "dismissed": "dismissed", "pending": "reset"}[status]
        _alerts().add(
            level="info" if status != "dismissed" else "warning",
            title=f"Shadow candidate {verb}: {candidate_algo_id}",
            body=f"Tournament {tournament_id} · {verb} {candidate_algo_id}",
            source="shadow",
            data={"tournament_id": str(tournament_id), "candidate": candidate_algo_id},
        )
    except Exception:
        pass
    return result.rowcount


async def run_correlation_scan_service(
    *,
    lookback_hours: int = 72,
    bucket_seconds: int = 3600,
    warn_threshold: float = 0.7,
    breach_threshold: float = 0.9,
) -> Any:
    """Run the cross-persona signal-correlation scan for the current tenant."""
    from ..risk.correlation_monitor import scan_persona_correlations

    return await scan_persona_correlations(
        tenant_id=_tenant_id(),
        lookback_hours=lookback_hours,
        bucket_seconds=bucket_seconds,
        warn_threshold=warn_threshold,
        breach_threshold=breach_threshold,
    )


async def update_discovery_status(discovery_id: UUID, status: str) -> bool:
    """Update a discovery's status (new/promoted/dismissed)."""
    from sqlalchemy import update

    from ..storage.models import DiscoveryModel

    if status not in ("new", "promoted", "dismissed"):
        raise ValueError(f"invalid status {status!r}")
    async with get_session() as session:
        with tenant_scope(_tenant_id()):
            result = await session.execute(
                update(DiscoveryModel)
                .where(DiscoveryModel.id == discovery_id)
                .where(DiscoveryModel.tenant_id == _tenant_id())
                .values(status=status)
            )
            await session.commit()
            return result.rowcount > 0
