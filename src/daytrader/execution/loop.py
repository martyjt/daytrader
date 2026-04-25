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
from ..core.pubsub import SignalEvent, signal_bus
from ..core.types.bars import Bar, Timeframe
from ..core.types.common import utcnow
from ..core.types.orders import Order, OrderSide, OrderStatus, OrderType
from ..core.types.signals import Signal
from ..core.types.symbols import Symbol
from ..data.adapters.registry import AdapterRegistry
from ..storage.database import get_session
from ..storage.models import (
    DiscoveryModel,
    PersonaModel,
    SignalModel,
    StrategyConfigModel,
)
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
        (
            algo_id,
            symbol_str,
            timeframe_str,
            algo_params,
            venue,
            source_discovery_id,
        ) = await self._resolve_persona_config(persona)

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

        # Hydrate any discovered feature attached to this strategy
        # config. ``features`` stays empty for plain strategies that
        # never bound a discovery — the existing algorithms don't
        # read from it, so this is a pure additive change.
        features: dict[str, float] = {}
        if source_discovery_id is not None:
            features = await self._hydrate_discovered_features(
                tenant_id=persona.tenant_id,
                discovery_id=source_discovery_id,
                now=timestamps[i],
            )

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
            features=features,
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

        # Persist the signal row, then fan it out to live UI subscribers.
        # Publication happens *after* commit so a subscriber that re-reads
        # the DB will see the same row that fired the event.
        await self._persist_and_publish_signal(persona, signal)

        # Determine action
        current_price = closes[i]

        # Resolve execution adapter (venue from resolved config takes precedence)
        executor = await self._resolve_executor(persona, venue=venue)
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

    async def _resolve_persona_config(
        self, persona: Any
    ) -> tuple[str | None, str | None, str, dict, str | None, UUID | None]:
        """Resolve trading config for a persona — read-through to StrategyConfig.

        Resolution order:
        1. If ``meta.strategy_config_id`` is set, load StrategyConfigModel by
           id and return its (algo_id, symbol, timeframe, algo_params,
           venue, source_discovery_id). This makes edits to the saved
           strategy propagate live.
        2. If the strategy was deleted or the id is malformed, log and fall
           through to (3).
        3. Use embedded ``persona.meta`` keys: ``algo_id`` (with
           ``algorithm_id`` accepted as a back-compat alias), ``symbol``,
           ``timeframe`` (default ``"1d"``), ``params``, ``venue``.
           ``source_discovery_id`` is always ``None`` for the embedded
           path — discovery binding only flows through StrategyConfig.

        Returns a 6-tuple. ``algo_id`` and ``symbol`` may be ``None``
        (caller treats that as "not configured for live"); ``timeframe``
        always falls back to ``"1d"`` and ``params`` always falls back
        to ``{}``.
        """
        meta = persona.meta or {}
        strategy_id_raw = meta.get("strategy_config_id")

        if strategy_id_raw:
            try:
                sid = (
                    strategy_id_raw
                    if isinstance(strategy_id_raw, UUID)
                    else UUID(str(strategy_id_raw))
                )
            except (ValueError, TypeError):
                logger.warning(
                    "Invalid strategy_config_id %r on persona %s",
                    strategy_id_raw, persona.name,
                )
                sid = None

            if sid is not None:
                async with get_session() as session:
                    with tenant_scope(persona.tenant_id):
                        repo = TenantRepository(session, StrategyConfigModel)
                        strategy = await repo.get(sid)
                if strategy is not None:
                    return (
                        strategy.algo_id,
                        strategy.symbol,
                        strategy.timeframe,
                        dict(strategy.algo_params or {}),
                        strategy.venue,
                        strategy.source_discovery_id,
                    )
                logger.warning(
                    "strategy_config_id %s on persona %s no longer exists; "
                    "falling back to embedded meta",
                    sid, persona.name,
                )

        algo_id = meta.get("algo_id") or meta.get("algorithm_id")
        return (
            algo_id,
            meta.get("symbol"),
            meta.get("timeframe", "1d"),
            dict(meta.get("params") or {}),
            meta.get("venue"),
            None,
        )

    async def _hydrate_discovered_features(
        self,
        *,
        tenant_id: UUID,
        discovery_id: UUID,
        now: datetime,
    ) -> dict[str, float]:
        """Resolve a promoted Discovery's feature value at ``now``.

        Returns a dict ready to drop into ``AlgorithmContext.features``.
        On any failure (discovery deleted, hydration upstream error,
        empty response) returns ``{}`` so the algorithm sees a missing
        feature and stays flat — never raises into the trading cycle.
        """
        try:
            async with get_session() as session:
                with tenant_scope(tenant_id):
                    repo = TenantRepository(session, DiscoveryModel)
                    discovery = await repo.get(discovery_id)
        except Exception:
            logger.exception(
                "Failed to load discovery %s for tenant %s",
                discovery_id, tenant_id,
            )
            return {}

        if discovery is None:
            logger.warning(
                "StrategyConfig references discovery %s which no longer exists",
                discovery_id,
            )
            return {}

        from ..research.feature_hydration import get_feature_hydrator

        value = await get_feature_hydrator().hydrate(discovery, now=now)
        if value is None:
            return {}
        return {discovery.candidate_name: float(value)}

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

    async def _persist_and_publish_signal(
        self, persona: Any, signal: Signal
    ) -> None:
        """Insert a ``SignalModel`` row and broadcast on the in-process bus.

        Persistence failures are logged but never propagated — a row that
        can't be written must not stop the live trading cycle.
        """
        try:
            async with get_session() as session:
                with tenant_scope(persona.tenant_id):
                    repo = TenantRepository(session, SignalModel)
                    row = await repo.create(
                        id=signal.id,
                        persona_id=persona.id,
                        symbol_key=signal.symbol_key,
                        score=signal.score,
                        confidence=signal.confidence,
                        source=signal.source,
                        reason=signal.reason or "",
                        meta=dict(signal.metadata or {}),
                    )
                    await session.commit()
                    created_at = row.created_at
        except Exception:
            logger.exception(
                "Failed to persist signal %s for persona %s", signal.id, persona.id
            )
            return

        try:
            signal_bus().publish(
                persona.tenant_id,
                SignalEvent(
                    tenant_id=persona.tenant_id,
                    persona_id=persona.id,
                    signal_id=signal.id,
                    symbol_key=signal.symbol_key,
                    score=signal.score,
                    confidence=signal.confidence,
                    source=signal.source,
                    reason=signal.reason or "",
                    created_at=created_at.isoformat() if created_at else "",
                ),
            )
        except Exception:
            logger.exception("signal_bus publish failed for signal %s", signal.id)

    async def _update_persona_equity(
        self, persona: Any, equity: Decimal
    ) -> None:
        """Update the persona's current equity in the DB."""
        async with get_session() as session:
            with tenant_scope(persona.tenant_id):
                repo = TenantRepository(session, PersonaModel)
                await repo.update(persona.id, current_equity=equity)
                await session.commit()

    async def _resolve_executor(
        self, persona: Any, venue: str | None = None
    ) -> Any | None:
        """Resolve the execution adapter for a persona.

        Paper personas use the global ``PaperExecutor``. Live personas resolve
        a *per-tenant* adapter via ``ExecutionRegistry.get_for_tenant`` —
        which reads the tenant's encrypted broker credentials.

        ``venue`` overrides ``persona.meta.venue`` when provided — used by the
        resolution chain so a venue from a bound StrategyConfig takes
        precedence over any stale value embedded in the persona's meta.
        """
        meta = persona.meta or {}
        mode = persona.mode

        if mode == "paper":
            try:
                return ExecutionRegistry.get("paper")
            except KeyError:
                logger.error("PaperExecutor not registered")
                return None

        # Live mode — resolve broker name, then per-tenant credentials
        if venue is None:
            venue = meta.get("venue")
        if not venue:
            if persona.asset_class == "crypto":
                venue = "binance"
            elif persona.asset_class == "equities":
                venue = "alpaca"
            else:
                logger.error(
                    "Cannot resolve executor for asset_class=%s",
                    persona.asset_class,
                )
                return None

        adapter = await ExecutionRegistry.get_for_tenant(persona.tenant_id, venue)
        if adapter is None:
            logger.error(
                "No %s credentials on file for tenant %s — persona %s cannot trade live",
                venue, persona.tenant_id, persona.name,
            )
        return adapter

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
