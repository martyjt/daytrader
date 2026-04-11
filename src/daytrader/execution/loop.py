"""Live trading loop — background asyncio task for live/paper execution.

Polls active personas, fetches the latest bar, runs the algorithm,
risk-checks the signal, submits orders, and journals everything.

Usage::

    loop = TradingLoop(journal=writer, kill_switch=ks, global_risk=monitor)
    await loop.start()
    ...
    await loop.stop()
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4

import numpy as np
import polars as pl

from ..algorithms.base import Algorithm
from ..algorithms.registry import AlgorithmRegistry
from ..core.context import AlgorithmContext, tenant_scope
from ..core.types.bars import Bar, Timeframe
from ..core.types.common import utcnow
from ..core.types.orders import Order, OrderSide, OrderStatus, OrderType
from ..core.types.signals import Signal
from ..core.types.symbols import Symbol
from ..data.adapters.registry import AdapterRegistry
from ..storage.database import get_session
from ..storage.models import PersonaModel
from ..storage.repository import TenantRepository

from .registry import ExecutionRegistry

logger = logging.getLogger(__name__)

# Default poll interval in seconds
DEFAULT_POLL_SECONDS = 30


class TradingLoop:
    """Background loop that drives live and paper trading."""

    def __init__(
        self,
        *,
        journal: Any = None,
        kill_switch: Any = None,
        global_risk: Any = None,
        poll_seconds: float = DEFAULT_POLL_SECONDS,
        tenant_id: UUID | None = None,
    ) -> None:
        self._journal = journal
        self._kill_switch = kill_switch
        self._global_risk = global_risk
        self._poll_seconds = poll_seconds
        self._tenant_id = tenant_id
        self._task: asyncio.Task | None = None
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    async def start(self) -> None:
        """Start the background trading loop."""
        if self._task is not None:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("Trading loop started (poll every %.0fs)", self._poll_seconds)

    async def stop(self) -> None:
        """Gracefully stop the loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("Trading loop stopped")

    async def _run_loop(self) -> None:
        """Main loop: poll → process → sleep → repeat."""
        while self._running:
            # Respect kill switch
            if self._kill_switch and self._kill_switch.is_activated:
                await asyncio.sleep(self._poll_seconds)
                continue

            try:
                await self._process_cycle()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Trading loop cycle error")

            await asyncio.sleep(self._poll_seconds)

    async def _process_cycle(self) -> None:
        """One iteration: process all active personas."""
        personas = await self._get_active_personas()
        if not personas:
            return

        # Global risk check (Layer 3): aggregate equity
        if self._global_risk:
            total_equity = sum(float(p.current_equity) for p in personas)
            if await self._global_risk.check_drawdown(total_equity):
                logger.warning("Global drawdown breach — activating kill switch")
                if self._kill_switch and self._tenant_id:
                    await self._kill_switch.activate(
                        self._tenant_id, reason="global_drawdown"
                    )
                return

        for persona in personas:
            try:
                await self._process_persona(persona)
            except Exception:
                logger.exception("Error processing persona %s", persona.name)

    async def _process_persona(self, persona: Any) -> None:
        """Fetch bar → run algo → risk check → submit order → journal."""
        meta = persona.meta or {}
        algo_id = meta.get("algorithm_id")
        symbol_str = meta.get("symbol")
        timeframe_str = meta.get("timeframe", "1d")
        algo_params = meta.get("params", {})

        if not algo_id or not symbol_str:
            return  # Persona not configured for live trading

        # Resolve algorithm
        try:
            algorithm = AlgorithmRegistry.get(algo_id)
        except KeyError:
            logger.warning("Algorithm %s not found for persona %s", algo_id, persona.name)
            return

        symbol = Symbol.parse(symbol_str)
        timeframe = Timeframe(timeframe_str)

        # Resolve data adapter
        adapter = self._resolve_data_adapter(persona.asset_class)
        if adapter is None:
            return

        # Fetch recent bars (enough for warmup + 1)
        warmup = max(algorithm.warmup_bars(), 200)
        now = utcnow()
        lookback_days = max(warmup * _timeframe_to_days(timeframe), 30)
        from datetime import timedelta

        start = now - timedelta(days=lookback_days)
        try:
            data = await adapter.fetch_ohlcv(symbol, timeframe, start, now)
        except Exception:
            logger.exception("Failed to fetch data for %s", symbol_str)
            return

        if data.is_empty() or len(data) < warmup:
            return

        # Record data freshness for staleness monitoring
        if self._global_risk:
            last_ts = data["timestamp"].to_list()[-1]
            if isinstance(last_ts, datetime):
                self._global_risk.record_data_timestamp(symbol_str, last_ts)

        # Build context from the latest bar
        closes = data["close"].to_numpy().astype(float)
        opens = data["open"].to_numpy().astype(float)
        highs = data["high"].to_numpy().astype(float)
        lows = data["low"].to_numpy().astype(float)
        volumes = data["volume"].to_numpy().astype(float)
        timestamps = data["timestamp"].to_list()

        i = len(closes) - 1  # Latest bar
        bar = Bar(
            timestamp=timestamps[i],
            open=Decimal(str(opens[i])),
            high=Decimal(str(highs[i])),
            low=Decimal(str(lows[i])),
            close=Decimal(str(closes[i])),
            volume=Decimal(str(volumes[i])),
        )

        emitted: list[Signal] = []
        debug_logs: list[dict] = []

        ctx = AlgorithmContext(
            tenant_id=persona.tenant_id,
            persona_id=persona.id,
            algorithm_id=algo_id,
            symbol=symbol,
            timeframe=timeframe,
            now=timestamps[i],
            bar=bar,
            history_arrays={
                "close": closes,
                "open": opens,
                "high": highs,
                "low": lows,
                "volume": volumes,
            },
            features={},
            params=algo_params,
            emit_fn=emitted.append,
            log_fn=lambda msg, fields: debug_logs.append(
                {"message": msg, **fields}
            ),
        )

        # Run algorithm
        algorithm.on_bar(ctx)

        if not emitted:
            return

        signal = emitted[0]

        # Journal the signal
        if self._journal:
            await self._journal.log_signal_emitted(
                persona.tenant_id, persona.id, signal
            )

        # Determine action
        current_price = closes[i]

        # Resolve execution adapter
        executor = self._resolve_executor(persona)
        if executor is None:
            return

        # Get current positions to decide buy/sell
        positions = await executor.get_positions(persona.id)
        has_position = bool(positions.get(symbol.key, Decimal(0)))

        order: Order | None = None

        if signal.score > 0.5 and not has_position:
            # BUY signal — go in
            balance = await executor.get_balance(persona.id)
            if balance > 0:
                qty = float(balance) / current_price
                order = Order(
                    id=uuid4(),
                    persona_id=persona.id,
                    symbol_key=symbol.key,
                    side=OrderSide.BUY,
                    type=OrderType.MARKET,
                    quantity=Decimal(str(round(qty, 8))),
                    status=OrderStatus.PENDING,
                    created_at=utcnow(),
                    price=Decimal(str(current_price)),
                    reason=signal.reason,
                )

        elif signal.score < -0.5 and has_position:
            # SELL signal — close position
            qty = positions.get(symbol.key, Decimal(0))
            if qty > 0:
                order = Order(
                    id=uuid4(),
                    persona_id=persona.id,
                    symbol_key=symbol.key,
                    side=OrderSide.SELL,
                    type=OrderType.MARKET,
                    quantity=qty,
                    status=OrderStatus.PENDING,
                    created_at=utcnow(),
                    price=Decimal(str(current_price)),
                    reason=signal.reason,
                )

        if order is None:
            return

        # Journal order submission
        if self._journal:
            await self._journal.log_order_submitted(
                persona.tenant_id, persona.id, order
            )

        # Submit order
        filled = await executor.submit_order(order)

        # Journal fill/rejection
        if self._journal:
            if filled.status == OrderStatus.FILLED:
                await self._journal.log_order_filled(
                    persona.tenant_id, persona.id, filled
                )
            elif filled.status == OrderStatus.REJECTED:
                await self._journal.log_order_cancelled(
                    persona.tenant_id, persona.id, filled
                )

        # Update persona equity in DB
        new_balance = await executor.get_balance(persona.id)
        new_positions = await executor.get_positions(persona.id)
        position_value = sum(
            float(qty) * current_price
            for qty in new_positions.values()
        )
        new_equity = Decimal(str(float(new_balance) + position_value))

        await self._update_persona_equity(persona, new_equity)

    async def _get_active_personas(self) -> list[Any]:
        """Query all personas in live or paper mode."""
        if self._tenant_id is None:
            return []
        async with get_session() as session:
            with tenant_scope(self._tenant_id):
                repo = TenantRepository(session, PersonaModel)
                live = list(await repo.get_all(mode="live"))
                paper = list(await repo.get_all(mode="paper"))
                return live + paper

    async def _update_persona_equity(
        self, persona: Any, equity: Decimal
    ) -> None:
        """Update the persona's current equity in the DB."""
        async with get_session() as session:
            with tenant_scope(persona.tenant_id):
                repo = TenantRepository(session, PersonaModel)
                await repo.update(persona.id, current_equity=equity)
                await session.commit()

    def _resolve_executor(self, persona: Any) -> Any | None:
        """Resolve the execution adapter for a persona."""
        meta = persona.meta or {}
        mode = persona.mode

        if mode == "paper":
            try:
                return ExecutionRegistry.get("paper")
            except KeyError:
                logger.error("PaperExecutor not registered")
                return None

        # Live mode — resolve by asset class or venue preference
        venue = meta.get("venue")
        if venue:
            try:
                return ExecutionRegistry.get(venue)
            except KeyError:
                logger.error("Execution adapter %s not registered", venue)
                return None

        # Fallback: asset_class → default adapter
        if persona.asset_class == "crypto":
            try:
                return ExecutionRegistry.get("binance")
            except KeyError:
                logger.error("No crypto execution adapter registered")
                return None
        elif persona.asset_class == "equities":
            try:
                return ExecutionRegistry.get("alpaca")
            except KeyError:
                logger.error("No equities execution adapter registered")
                return None

        logger.error("Cannot resolve executor for asset_class=%s", persona.asset_class)
        return None

    @staticmethod
    def _resolve_data_adapter(asset_class: str) -> Any | None:
        """Resolve the data adapter for an asset class."""
        try:
            if asset_class == "equities":
                # Prefer alpaca for equities if available
                try:
                    return AdapterRegistry.get("alpaca")
                except KeyError:
                    pass
            return AdapterRegistry.get("yfinance")
        except KeyError:
            logger.error("No data adapter available for %s", asset_class)
            return None


def _timeframe_to_days(tf: Timeframe) -> float:
    """Approximate days per bar for lookback calculation."""
    return {
        Timeframe.M1: 1 / 1440,
        Timeframe.M5: 5 / 1440,
        Timeframe.M15: 15 / 1440,
        Timeframe.M30: 30 / 1440,
        Timeframe.H1: 1 / 24,
        Timeframe.H4: 4 / 24,
        Timeframe.D1: 1.0,
        Timeframe.W1: 7.0,
    }.get(tf, 1.0)
