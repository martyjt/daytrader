"""Portfolio-level backtesting across a symbol universe.

Runs an independent backtest for each symbol in the universe using the
same algorithm + venue, then aggregates the equity curves into a
portfolio view with equal capital allocation (1/N per symbol).

Intentionally simple for v1:

* Equal-weight static allocation — no rebalancing or correlation-aware sizing.
* Each symbol is backtested independently; no cross-symbol signals.
* Portfolio equity = sum of per-symbol equity curves.
* Portfolio KPIs are computed on the aggregate equity curve.

This is the minimum viable multi-symbol capability. Rebalancing,
hierarchical risk budgeting, and factor-portfolio construction are
Phase 8+ work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID

import numpy as np


@dataclass
class PortfolioSymbolResult:
    """Per-symbol backtest summary in the portfolio context."""

    symbol: str
    net_return_pct: float
    sharpe: float
    max_drawdown_pct: float
    num_trades: int
    final_equity: float
    equity_curve: list[float] = field(default_factory=list)
    error: str | None = None


@dataclass
class PortfolioBacktestResult:
    """Aggregate portfolio backtest output."""

    symbols: list[PortfolioSymbolResult] = field(default_factory=list)
    portfolio_equity_curve: list[float] = field(default_factory=list)
    initial_total_capital: float = 0.0
    final_total_equity: float = 0.0
    portfolio_return_pct: float = 0.0
    portfolio_sharpe: float = 0.0
    portfolio_max_drawdown_pct: float = 0.0

    @property
    def best_symbol(self) -> PortfolioSymbolResult | None:
        ok = [s for s in self.symbols if s.error is None]
        if not ok:
            return None
        return max(ok, key=lambda s: s.sharpe)

    @property
    def worst_symbol(self) -> PortfolioSymbolResult | None:
        ok = [s for s in self.symbols if s.error is None]
        if not ok:
            return None
        return min(ok, key=lambda s: s.sharpe)


async def run_portfolio_backtest(
    *,
    algo_id: str,
    symbols: list[str],
    timeframe_str: str,
    start: datetime,
    end: datetime,
    total_capital: float = 10_000.0,
    venue: str = "binance_spot",
    algo_params: dict[str, Any] | None = None,
    tenant_id: UUID | None = None,
) -> PortfolioBacktestResult:
    """Backtest ``algo_id`` on each symbol with equal 1/N capital allocation.

    ``tenant_id`` is required to resolve a tenant's sandboxed plugin.
    Built-in algorithms are reachable without it. Returns aggregate +
    per-symbol results.
    """
    from ..algorithms.registry import AlgorithmRegistry
    from ..backtest.engine import BacktestEngine
    from ..backtest.risk import RiskConfig
    from ..core.types.bars import Timeframe
    from ..core.types.symbols import Symbol

    if not symbols:
        return PortfolioBacktestResult(initial_total_capital=total_capital)

    timeframe = Timeframe(timeframe_str)
    per_symbol_capital = total_capital / len(symbols)
    algorithm = AlgorithmRegistry.get(algo_id, tenant_id=tenant_id)
    engine = BacktestEngine()

    results: list[PortfolioSymbolResult] = []
    for symbol_str in symbols:
        symbol = Symbol.parse(symbol_str)
        try:
            bt = await engine.run(
                algorithm=algorithm,
                symbol=symbol,
                timeframe=timeframe,
                start=start,
                end=end,
                initial_capital=per_symbol_capital,
                venue=venue,
                risk_config=RiskConfig.disabled(),
                params=algo_params,
            )
        except Exception as exc:  # noqa: BLE001
            results.append(PortfolioSymbolResult(
                symbol=symbol.key,
                net_return_pct=0.0,
                sharpe=0.0,
                max_drawdown_pct=0.0,
                num_trades=0,
                final_equity=per_symbol_capital,
                equity_curve=[],
                error=str(exc),
            ))
            continue

        kpis = bt.kpis or {}
        results.append(PortfolioSymbolResult(
            symbol=symbol.key,
            net_return_pct=float(kpis.get("net_return_pct", 0.0)),
            sharpe=float(kpis.get("sharpe_ratio", 0.0)),
            max_drawdown_pct=float(kpis.get("max_drawdown_pct", 0.0)),
            num_trades=int(kpis.get("num_trades", 0)),
            final_equity=float(bt.final_equity),
            equity_curve=list(bt.equity_curve or []),
        ))

    # Build the portfolio equity curve: per-bar sum of all symbols' equity curves.
    # Symbols may have different bar counts — we truncate to the shortest so
    # every bar represents a real (not padded) equity total.
    ok = [r for r in results if r.error is None and r.equity_curve]
    if ok:
        min_len = min(len(r.equity_curve) for r in ok)
        portfolio_curve = [0.0] * min_len
        for r in ok:
            for i in range(min_len):
                portfolio_curve[i] += r.equity_curve[i]
    else:
        portfolio_curve = []

    # Aggregate KPIs on the portfolio curve.
    if portfolio_curve:
        initial = float(total_capital)
        final = float(portfolio_curve[-1])
        rets = np.diff(np.array(portfolio_curve)) / np.array(portfolio_curve[:-1])
        rets = np.where(np.isfinite(rets), rets, 0.0)
        sharpe = float(np.sqrt(252) * rets.mean() / rets.std()) if rets.std() > 0 else 0.0

        running_peak = np.maximum.accumulate(np.array(portfolio_curve))
        drawdowns = (running_peak - np.array(portfolio_curve)) / running_peak
        max_dd = float(drawdowns.max() * 100) if len(drawdowns) else 0.0

        portfolio_return_pct = (final / initial - 1.0) * 100.0
    else:
        final = total_capital
        sharpe = 0.0
        max_dd = 0.0
        portfolio_return_pct = 0.0

    return PortfolioBacktestResult(
        symbols=results,
        portfolio_equity_curve=portfolio_curve,
        initial_total_capital=total_capital,
        final_total_equity=final,
        portfolio_return_pct=portfolio_return_pct,
        portfolio_sharpe=sharpe,
        portfolio_max_drawdown_pct=max_dd,
    )
