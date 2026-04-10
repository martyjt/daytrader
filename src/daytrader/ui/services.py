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
) -> Any:
    """Run a backtest and return the result.

    Returns a ``BacktestResult`` with equity_curve, trades, signals, kpis.
    """
    from datetime import datetime as dt

    from ..algorithms.registry import AlgorithmRegistry
    from ..backtest.engine import BacktestEngine
    from ..core.types.bars import Timeframe
    from ..core.types.symbols import Symbol

    algorithm = AlgorithmRegistry.get(algo_id)
    symbol = Symbol.parse(symbol_str)
    timeframe = Timeframe(timeframe_str)
    start = dt.strptime(start_str, "%Y-%m-%d")
    end = dt.strptime(end_str, "%Y-%m-%d")

    engine = BacktestEngine()
    return await engine.run(
        algorithm=algorithm,
        symbol=symbol,
        timeframe=timeframe,
        start=start,
        end=end,
        initial_capital=capital,
    )
