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
from ..data.features.store import FeatureStore
from .fees import FeeModel, FeeSchedule, VENUE_PROFILES
from .risk import RiskConfig, compute_atr, check_stop_loss, check_take_profit, stop_loss_price, take_profit_price, DailyPnLTracker


# Module-level cache for OHLCV fetches. First-time backtest hits the
# adapter; subsequent runs with the same (symbol, timeframe, start, end)
# read from on-disk Parquet. Cache lives at data/features/*.parquet.
from pathlib import Path as _Path

_FEATURE_STORE = FeatureStore(
    _Path(__file__).resolve().parents[3] / "data" / "features"
)


async def _fetch_ohlcv_cached(
    adapter: Any,
    symbol: Symbol,
    timeframe: Timeframe,
    start: datetime,
    end: datetime,
) -> pl.DataFrame:
    """Read-through cache wrapper around ``adapter.fetch_ohlcv``.

    Cache key is ``(symbol, timeframe, start, end)``. Non-empty
    DataFrames are persisted; empty results are not cached so a
    transient failure won't poison the cache.
    """
    cache_key = f"{symbol.base}_{symbol.quote}"
    tf_str = timeframe.value
    cached = _FEATURE_STORE.get(cache_key, tf_str, start, end)
    if cached is not None and not cached.is_empty():
        return cached
    df = await adapter.fetch_ohlcv(symbol, timeframe, start, end)
    if not df.is_empty():
        _FEATURE_STORE.put(cache_key, tf_str, start, end, df)
    return df


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
    debug_logs: list[dict[str, Any]] = field(default_factory=list)
    risk_events: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


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
        risk_config: RiskConfig | None = None,
        params: dict[str, Any] | None = None,
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

        # Fetch data if not provided (with on-disk cache)
        if data is None:
            if adapter is None:
                from ..data.adapters.registry import AdapterRegistry

                AdapterRegistry.auto_register()
                adapter = AdapterRegistry.get("yfinance")
            data = await _fetch_ohlcv_cached(adapter, symbol, timeframe, start, end)

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

        # Resolve params: manifest defaults merged with user overrides
        resolved_params = algorithm.manifest.param_defaults()
        if params:
            resolved_params.update(params)

        # Sandboxed plugins must run their per-bar work in the worker
        # subprocess. Doing one round-trip per bar would be ruinous; instead
        # we batch the whole bar series into a single ``replay_bars`` RPC,
        # then hand the resulting signal-by-bar dict to ``_simulate``. The
        # CPU-bound trading-decision loop stays in a worker thread.
        from ..algorithms.sandbox import SandboxedAlgorithm

        sandbox_signals: dict[int, list[Signal]] | None = None
        sandbox_logs: dict[int, list[dict[str, Any]]] | None = None
        if isinstance(algorithm, SandboxedAlgorithm):
            sandbox_signals, sandbox_logs = await self._precompute_sandbox(
                algorithm=algorithm, symbol=symbol, timeframe=timeframe,
                data=data, params=resolved_params,
            )

        # Run the CPU-bound simulation in a worker thread so the
        # asyncio event loop stays responsive. Without this, long
        # composite backtests (5+ seconds) block NiceGUI's websocket
        # heartbeat and the client sees "connection lost".
        import asyncio

        return await asyncio.to_thread(
            self._simulate,
            algorithm, symbol, timeframe, data,
            initial_capital, fee_model, resolved_venue,
            risk_config=risk_config,
            params=resolved_params,
            sandbox_signals=sandbox_signals,
            sandbox_logs=sandbox_logs,
        )

    async def _precompute_sandbox(
        self,
        *,
        algorithm: "SandboxedAlgorithm",
        symbol: Symbol,
        timeframe: Timeframe,
        data: pl.DataFrame,
        params: dict[str, Any],
    ) -> tuple[dict[int, list[Signal]], dict[int, list[dict[str, Any]]]]:
        """Run a sandboxed algorithm over every post-warmup bar in one RPC.

        Returns ``(signals, logs)`` keyed by bar index. The trading-decision
        loop in ``_simulate`` reads from these dicts instead of calling
        ``algorithm.on_bar`` per bar.
        """
        closes = data["close"].to_numpy().astype(float)
        opens = data["open"].to_numpy().astype(float)
        highs = data["high"].to_numpy().astype(float)
        lows = data["low"].to_numpy().astype(float)
        volumes = data["volume"].to_numpy().astype(float)
        timestamps = data["timestamp"].to_list()
        n_bars = len(closes)
        warmup = algorithm.warmup_bars()

        signals: dict[int, list[Signal]] = {}
        logs: dict[int, list[dict[str, Any]]] = {}
        if n_bars <= warmup:
            return signals, logs

        tenant_id = uuid4()
        persona_id = uuid4()
        pre_contexts: list[AlgorithmContext] = []
        pre_emitted: list[list[Signal]] = []
        pre_logs: list[list[dict[str, Any]]] = []
        pre_indices: list[int] = []
        for j in range(warmup, n_bars):
            em: list[Signal] = []
            dl: list[dict[str, Any]] = []
            pre_contexts.append(AlgorithmContext(
                tenant_id=tenant_id,
                persona_id=persona_id,
                algorithm_id=algorithm.manifest.id,
                symbol=symbol,
                timeframe=timeframe,
                now=timestamps[j],
                bar=Bar(
                    timestamp=timestamps[j],
                    open=Decimal(str(opens[j])),
                    high=Decimal(str(highs[j])),
                    low=Decimal(str(lows[j])),
                    close=Decimal(str(closes[j])),
                    volume=Decimal(str(volumes[j])),
                ),
                history_arrays={
                    "close": closes[: j + 1],
                    "open": opens[: j + 1],
                    "high": highs[: j + 1],
                    "low": lows[: j + 1],
                    "volume": volumes[: j + 1],
                },
                features={},
                params=params or {},
                emit_fn=em.append,
                log_fn=lambda msg, fields, _l=dl, _i=j, _ts=timestamps[j]: _l.append(
                    {"bar": _i, "timestamp": str(_ts), "message": msg, **fields}
                ),
            ))
            pre_emitted.append(em)
            pre_logs.append(dl)
            pre_indices.append(j)

        await algorithm.replay_bars(pre_contexts)
        for idx, em, dl in zip(pre_indices, pre_emitted, pre_logs):
            signals[idx] = em
            logs[idx] = dl
        return signals, logs

    def _simulate(
        self,
        algorithm: Algorithm,
        symbol: Symbol,
        timeframe: Timeframe,
        data: pl.DataFrame,
        initial_capital: float,
        fee_model: FeeModel,
        venue: str,
        *,
        risk_config: RiskConfig | None = None,
        params: dict[str, Any] | None = None,
        sandbox_signals: dict[int, list[Signal]] | None = None,
        sandbox_logs: dict[int, list[dict[str, Any]]] | None = None,
    ) -> BacktestResult:
        if risk_config is None:
            risk_config = RiskConfig.disabled()

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
        entry_bar = 0
        equity_curve: list[float] = []
        trades: list[dict[str, Any]] = []
        all_signals: list[Signal] = []
        total_fees = 0.0
        debug_logs: list[dict[str, Any]] = []
        risk_events: list[dict[str, Any]] = []

        # Risk layer 2: daily P&L tracker
        daily_tracker = DailyPnLTracker(initial_capital, risk_config.daily_loss_limit_pct)

        tenant_id = uuid4()
        persona_id = uuid4()

        # Sandboxed plugin signals are pre-computed in ``run()`` (which is
        # async) and passed in as a dict keyed by bar index. The per-bar
        # path below reads them out instead of calling on_bar.
        is_sandboxed = sandbox_signals is not None
        sandbox_signals = sandbox_signals or {}
        sandbox_logs = sandbox_logs or {}

        import time as _time
        for i in range(n_bars):
            # Release the GIL briefly every 100 bars so the asyncio
            # event loop (running on the main thread) can service the
            # NiceGUI websocket heartbeat during long backtests.
            if i and i % 100 == 0:
                _time.sleep(0)
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

            # --- Risk Layer 1: per-trade SL/TP/max-hold checks ---
            risk_exit = False
            if risk_config.enabled and position_qty > 0:
                atr = compute_atr(
                    highs[: i + 1], lows[: i + 1], closes[: i + 1],
                    period=risk_config.atr_period,
                )

                if atr > 0:
                    # Stop-loss check
                    if check_stop_loss(lows[i], entry_price, atr, risk_config.stop_loss_atr_mult):
                        exit_price = stop_loss_price(entry_price, atr, risk_config.stop_loss_atr_mult)
                        exit_price = max(exit_price, lows[i])  # can't fill below bar low
                        sale = position_qty * exit_price
                        fee = fee_model.trade_cost(sale, volatility_pct=vol_pct)
                        total_fees += fee
                        cash = sale - fee
                        event = {
                            "bar": i, "timestamp": str(timestamps[i]),
                            "type": "stop_loss", "exit_price": round(exit_price, 4),
                            "entry_price": round(entry_price, 4), "atr": round(atr, 4),
                        }
                        risk_events.append(event)
                        trades.append({
                            "bar": i, "timestamp": str(timestamps[i]),
                            "action": "sell", "price": exit_price,
                            "quantity": position_qty,
                            "pnl": cash - initial_capital,
                            "fee": round(fee, 4),
                            "exit_reason": "stop_loss",
                        })
                        position_qty = 0.0
                        risk_exit = True

                    # Take-profit check
                    elif check_take_profit(highs[i], entry_price, atr, risk_config.take_profit_atr_mult):
                        exit_price = take_profit_price(entry_price, atr, risk_config.take_profit_atr_mult)
                        exit_price = min(exit_price, highs[i])  # can't fill above bar high
                        sale = position_qty * exit_price
                        fee = fee_model.trade_cost(sale, volatility_pct=vol_pct)
                        total_fees += fee
                        cash = sale - fee
                        event = {
                            "bar": i, "timestamp": str(timestamps[i]),
                            "type": "take_profit", "exit_price": round(exit_price, 4),
                            "entry_price": round(entry_price, 4), "atr": round(atr, 4),
                        }
                        risk_events.append(event)
                        trades.append({
                            "bar": i, "timestamp": str(timestamps[i]),
                            "action": "sell", "price": exit_price,
                            "quantity": position_qty,
                            "pnl": cash - initial_capital,
                            "fee": round(fee, 4),
                            "exit_reason": "take_profit",
                        })
                        position_qty = 0.0
                        risk_exit = True

                # Max hold bars check
                if not risk_exit and (i - entry_bar) >= risk_config.max_hold_bars:
                    sale = position_qty * current_price
                    fee = fee_model.trade_cost(sale, volatility_pct=vol_pct)
                    total_fees += fee
                    cash = sale - fee
                    event = {
                        "bar": i, "timestamp": str(timestamps[i]),
                        "type": "max_hold", "bars_held": i - entry_bar,
                    }
                    risk_events.append(event)
                    trades.append({
                        "bar": i, "timestamp": str(timestamps[i]),
                        "action": "sell", "price": current_price,
                        "quantity": position_qty,
                        "pnl": cash - initial_capital,
                        "fee": round(fee, 4),
                        "exit_reason": "max_hold",
                    })
                    position_qty = 0.0
                    risk_exit = True

            # --- Risk Layer 2: daily loss limit ---
            daily_halted = False
            if risk_config.enabled:
                equity = cash + position_qty * current_price
                daily_halted = daily_tracker.update(equity, timestamps[i])
                if daily_halted and not risk_exit:
                    # Force-close any open position
                    if position_qty > 0:
                        sale = position_qty * current_price
                        fee = fee_model.trade_cost(sale, volatility_pct=vol_pct)
                        total_fees += fee
                        cash = sale - fee
                        event = {
                            "bar": i, "timestamp": str(timestamps[i]),
                            "type": "daily_limit",
                        }
                        risk_events.append(event)
                        trades.append({
                            "bar": i, "timestamp": str(timestamps[i]),
                            "action": "sell", "price": current_price,
                            "quantity": position_qty,
                            "pnl": cash - initial_capital,
                            "fee": round(fee, 4),
                            "exit_reason": "daily_limit",
                        })
                        position_qty = 0.0
                        risk_exit = True

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
            if is_sandboxed:
                # Pull the signal we computed pre-loop in the replay_bars batch.
                emitted = list(sandbox_signals.get(i, []))
                debug_logs.extend(sandbox_logs.get(i, []))
            else:
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
                    params=params or {},
                    emit_fn=emitted.append,
                    log_fn=lambda msg, fields, _i=i, _ts=timestamps[i]: debug_logs.append(
                        {"bar": _i, "timestamp": str(_ts), "message": msg, **fields}
                    ),
                )

                algorithm.on_bar(ctx)
            all_signals.extend(emitted)

            # Skip signal execution if risk exit already happened or daily halted
            if not risk_exit and not daily_halted and emitted:
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
                    entry_bar = i
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

        # Surface data-sufficiency warnings so zero-trade results aren't
        # silently misinterpreted as "strategy didn't work".
        warnings_list: list[str] = []
        signal_bars = max(0, n_bars - warmup)
        if n_bars == 0:
            warnings_list.append(
                "No data returned for this symbol/date range. "
                "Try a different symbol, a wider date range, or a different adapter."
            )
        elif signal_bars < 10:
            warnings_list.append(
                f"Only {signal_bars} usable bars after the {warmup}-bar warmup "
                f"({n_bars} total). Results will be unreliable — extend the date range "
                f"or use a shorter timeframe."
            )
        elif len(trades) == 0 and signal_bars >= 10:
            warnings_list.append(
                f"Algorithm produced no trades across {signal_bars} signal bars. "
                f"The strategy may be too restrictive for this data, or thresholds "
                f"may need tuning."
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
            debug_logs=debug_logs,
            risk_events=risk_events,
            warnings=warnings_list,
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
