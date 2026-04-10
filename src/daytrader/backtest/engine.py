"""Backtest engine with realistic fee modeling.

Runs an algorithm bar-by-bar over historical OHLCV data, tracks
positions with an all-in/all-out model, applies venue-specific fees
(commission + spread + slippage), and computes standard KPIs with
gross vs net breakdown.

Phase 0: single-position, long-only, no shorting.
Phase 1: position sizing, stop-loss/TP, multi-position, fee sensitivity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import uuid4

import numpy as np
import polars as pl

from ..algorithms.base import Algorithm
from ..core.context import AlgorithmContext
from ..core.types.bars import Bar, Timeframe
from ..core.types.signals import Signal
from ..core.types.symbols import Symbol
from .fees import FeeModel, FeeSchedule, VENUE_PROFILES


@dataclass
class BacktestResult:
    """Output of a backtest run."""

    equity_curve: list[float]
    timestamps: list[Any]
    trades: list[dict[str, Any]]
    signals: list[Signal]
    kpis: dict[str, float]
    initial_capital: float
    final_equity: float
    total_fees_paid: float = 0.0
    venue: str = "custom"


class BacktestEngine:
    """Run an algorithm against OHLCV data and produce a BacktestResult.

    Usage::

        engine = BacktestEngine()

        # With venue-specific fees:
        result = await engine.run(
            algorithm=BuyHoldAlgorithm(),
            symbol=Symbol.parse("BTC-USD"),
            timeframe=Timeframe.D1,
            start=datetime(2024, 1, 1),
            end=datetime(2024, 12, 31),
            venue="binance_spot",
            data=df,
        )

        # Backward compatible (flat commission):
        result = await engine.run(..., commission_bps=10)
    """

    async def run(
        self,
        *,
        algorithm: Algorithm,
        symbol: Symbol,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
        initial_capital: float = 10_000.0,
        commission_bps: float = 10.0,
        venue: str | None = None,
        fee_model: FeeModel | None = None,
        data: pl.DataFrame | None = None,
        adapter: Any = None,
    ) -> BacktestResult:
        """Execute a backtest.

        Fee resolution order:
        1. Explicit ``fee_model`` (highest priority)
        2. ``venue`` name (looked up from VENUE_PROFILES)
        3. ``commission_bps`` flat rate (backward compat)
        """
        # Resolve fee model
        if fee_model is None:
            if venue and venue in VENUE_PROFILES:
                fee_model = FeeModel(VENUE_PROFILES[venue])
            else:
                fee_model = FeeModel(FeeSchedule.from_flat_bps(commission_bps))

        resolved_venue = fee_model.schedule.venue

        # Fetch data if not provided
        if data is None:
            if adapter is None:
                from ..data.adapters.registry import AdapterRegistry

                AdapterRegistry.auto_register()
                adapter = AdapterRegistry.get("yfinance")
            data = await adapter.fetch_ohlcv(symbol, timeframe, start, end)

        if data.is_empty():
            return BacktestResult(
                equity_curve=[],
                timestamps=[],
                trades=[],
                signals=[],
                kpis=_empty_kpis(),
                initial_capital=initial_capital,
                final_equity=initial_capital,
                venue=resolved_venue,
            )

        return self._simulate(
            algorithm, symbol, timeframe, data,
            initial_capital, fee_model, resolved_venue,
        )

    def _simulate(
        self,
        algorithm: Algorithm,
        symbol: Symbol,
        timeframe: Timeframe,
        data: pl.DataFrame,
        initial_capital: float,
        fee_model: FeeModel,
        venue: str,
    ) -> BacktestResult:
        closes = data["close"].to_numpy().astype(float)
        opens = data["open"].to_numpy().astype(float)
        highs = data["high"].to_numpy().astype(float)
        lows = data["low"].to_numpy().astype(float)
        volumes = data["volume"].to_numpy().astype(float)
        timestamps = data["timestamp"].to_list()

        n_bars = len(closes)
        warmup = algorithm.warmup_bars()

        cash = initial_capital
        position_qty = 0.0
        entry_price = 0.0
        equity_curve: list[float] = []
        trades: list[dict[str, Any]] = []
        all_signals: list[Signal] = []
        total_fees = 0.0

        tenant_id = uuid4()
        persona_id = uuid4()

        for i in range(n_bars):
            current_price = closes[i]

            # Mark-to-market before signal processing
            equity = cash + position_qty * current_price

            if i < warmup:
                equity_curve.append(equity)
                continue

            # Estimate current volatility (ATR-like, for slippage scaling)
            vol_pct = 0.0
            if i >= 14 and current_price > 0:
                recent_returns = np.diff(closes[max(0, i - 14) : i + 1]) / closes[max(0, i - 14) : i]
                vol_pct = float(np.std(recent_returns) * 100) if len(recent_returns) > 1 else 0.0

            # Build AlgorithmContext for this bar
            bar = Bar(
                timestamp=timestamps[i],
                open=Decimal(str(opens[i])),
                high=Decimal(str(highs[i])),
                low=Decimal(str(lows[i])),
                close=Decimal(str(closes[i])),
                volume=Decimal(str(volumes[i])),
            )

            emitted: list[Signal] = []
            ctx = AlgorithmContext(
                tenant_id=tenant_id,
                persona_id=persona_id,
                algorithm_id=algorithm.manifest.id,
                symbol=symbol,
                timeframe=timeframe,
                now=timestamps[i],
                bar=bar,
                history_arrays={
                    "close": closes[: i + 1],
                    "open": opens[: i + 1],
                    "high": highs[: i + 1],
                    "low": lows[: i + 1],
                    "volume": volumes[: i + 1],
                },
                features={},
                params={},
                emit_fn=emitted.append,
                log_fn=lambda msg, fields: None,
            )

            algorithm.on_bar(ctx)
            all_signals.extend(emitted)

            # Simple execution: first signal drives action
            if emitted:
                sig = emitted[0]

                if sig.score > 0.5 and position_qty == 0:
                    # BUY — go all-in
                    fee = fee_model.trade_cost(
                        cash, volatility_pct=vol_pct,
                    )
                    total_fees += fee
                    investable = cash - fee
                    position_qty = investable / current_price
                    entry_price = current_price
                    cash = 0.0
                    trades.append({
                        "bar": i,
                        "timestamp": str(timestamps[i]),
                        "action": "buy",
                        "price": current_price,
                        "quantity": position_qty,
                        "fee": round(fee, 4),
                    })

                elif sig.score < -0.5 and position_qty > 0:
                    # SELL — close entire position
                    sale = position_qty * current_price
                    fee = fee_model.trade_cost(
                        sale, volatility_pct=vol_pct,
                    )
                    total_fees += fee
                    cash = sale - fee
                    trades.append({
                        "bar": i,
                        "timestamp": str(timestamps[i]),
                        "action": "sell",
                        "price": current_price,
                        "quantity": position_qty,
                        "pnl": cash - initial_capital,
                        "fee": round(fee, 4),
                    })
                    position_qty = 0.0

            equity = cash + position_qty * current_price
            equity_curve.append(equity)

        # Final equity
        final_equity = equity_curve[-1] if equity_curve else initial_capital

        kpis = _compute_kpis(
            equity_curve, trades, initial_capital, total_fees, n_bars,
        )

        return BacktestResult(
            equity_curve=equity_curve,
            timestamps=[str(t) for t in timestamps],
            trades=trades,
            signals=all_signals,
            kpis=kpis,
            initial_capital=initial_capital,
            final_equity=final_equity,
            total_fees_paid=round(total_fees, 4),
            venue=venue,
        )


# ---------------------------------------------------------------------------
# KPI helpers
# ---------------------------------------------------------------------------


def _empty_kpis() -> dict[str, float]:
    return {
        "total_return_pct": 0.0,
        "net_return_pct": 0.0,
        "gross_return_pct": 0.0,
        "sharpe_ratio": 0.0,
        "max_drawdown_pct": 0.0,
        "num_trades": 0,
        "win_rate_pct": 0.0,
        "profit_factor": 0.0,
        "total_fees": 0.0,
        "fee_drag_pct": 0.0,
        "avg_trades_per_day": 0.0,
    }


def _compute_kpis(
    equity_curve: list[float],
    trades: list[dict],
    initial_capital: float,
    total_fees: float,
    n_bars: int,
) -> dict[str, float]:
    if len(equity_curve) < 2:
        return _empty_kpis()

    eq = np.array(equity_curve, dtype=float)
    final = eq[-1]

    # Returns
    net_return = (final - initial_capital) / initial_capital * 100
    fee_drag = total_fees / initial_capital * 100
    gross_return = net_return + fee_drag  # approximate

    # Daily returns
    returns = np.diff(eq) / eq[:-1]
    returns = returns[np.isfinite(returns)]

    # Sharpe ratio (annualized, assuming daily bars)
    if len(returns) > 1 and np.std(returns) > 0:
        sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(252))
    else:
        sharpe = 0.0

    # Max drawdown
    peak = np.maximum.accumulate(eq)
    drawdown = (eq - peak) / peak
    max_dd = float(np.min(drawdown) * 100)

    # Win rate (from closed buy→sell pairs)
    buys = [t for t in trades if t["action"] == "buy"]
    sells = [t for t in trades if t["action"] == "sell"]
    if sells:
        wins = sum(
            1 for b, s in zip(buys, sells) if s["price"] > b["price"]
        )
        win_rate = wins / len(sells) * 100
    else:
        win_rate = 0.0

    # Profit factor
    gains = sum(
        s["price"] - b["price"]
        for b, s in zip(buys, sells)
        if s["price"] > b["price"]
    )
    losses = sum(
        b["price"] - s["price"]
        for b, s in zip(buys, sells)
        if s["price"] <= b["price"]
    )
    profit_factor = gains / losses if losses > 0 else 0.0

    # Trade frequency
    avg_trades_per_day = len(trades) / max(n_bars, 1)

    return {
        "total_return_pct": round(net_return, 2),
        "net_return_pct": round(net_return, 2),
        "gross_return_pct": round(gross_return, 2),
        "sharpe_ratio": round(sharpe, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "num_trades": len(trades),
        "win_rate_pct": round(win_rate, 1),
        "profit_factor": round(profit_factor, 2),
        "total_fees": round(total_fees, 2),
        "fee_drag_pct": round(fee_drag, 2),
        "avg_trades_per_day": round(avg_trades_per_day, 3),
    }
