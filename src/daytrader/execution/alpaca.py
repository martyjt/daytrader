"""Alpaca execution adapter — US equities trading via alpaca-py.

Submits and cancels orders on Alpaca.  Paper mode is enabled by default.
Uses ``asyncio.to_thread()`` because the alpaca-py ``TradingClient`` is
synchronous — same pattern as the Alpaca data adapter.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import replace
from decimal import Decimal
from typing import Any
from uuid import UUID

from ..core.types.orders import Order, OrderSide, OrderStatus, OrderType
from .base import ExecutionAdapter

logger = logging.getLogger(__name__)


class AlpacaExecutor(ExecutionAdapter):
    """US equities execution via Alpaca."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        paper: bool = True,
    ) -> None:
        self._api_key = api_key
        self._api_secret = api_secret
        self._paper = paper
        # Map our UUIDs to Alpaca order IDs for cancel support.
        self._order_id_map: dict[UUID, str] = {}

    @property
    def name(self) -> str:
        return "alpaca"

    async def submit_order(self, order: Order) -> Order:
        try:
            result = await asyncio.to_thread(self._submit_sync, order)

            alpaca_id = str(result.id)
            self._order_id_map[order.id] = alpaca_id

            status = _map_alpaca_status(str(result.status))
            filled_qty = Decimal(str(result.filled_qty or 0))
            avg_price = (
                Decimal(str(result.filled_avg_price))
                if result.filled_avg_price
                else order.price
            )

            return replace(
                order,
                status=status,
                filled_quantity=filled_qty,
                avg_fill_price=avg_price,
                metadata={**order.metadata, "alpaca_order_id": alpaca_id},
            )

        except Exception as exc:
            logger.error("Alpaca submit_order failed: %s", exc)
            return replace(
                order,
                status=OrderStatus.REJECTED,
                reason=f"Alpaca API error: {exc}",
            )

    def _submit_sync(self, order: Order) -> Any:
        """Synchronous order submission (runs in thread)."""
        from alpaca.trading.client import TradingClient
        from alpaca.trading.enums import OrderSide as AlpacaSide, TimeInForce
        from alpaca.trading.requests import (
            LimitOrderRequest,
            MarketOrderRequest,
            StopLimitOrderRequest,
            StopOrderRequest,
        )

        client = TradingClient(self._api_key, self._api_secret, paper=self._paper)
        ticker = self.to_alpaca_ticker(order.symbol_key)
        side = AlpacaSide.BUY if order.side == OrderSide.BUY else AlpacaSide.SELL

        if order.type == OrderType.MARKET:
            request = MarketOrderRequest(
                symbol=ticker,
                qty=float(order.quantity),
                side=side,
                time_in_force=TimeInForce.DAY,
            )
        elif order.type == OrderType.LIMIT:
            request = LimitOrderRequest(
                symbol=ticker,
                qty=float(order.quantity),
                side=side,
                time_in_force=TimeInForce.DAY,
                limit_price=float(order.price) if order.price else None,
            )
        elif order.type == OrderType.STOP:
            request = StopOrderRequest(
                symbol=ticker,
                qty=float(order.quantity),
                side=side,
                time_in_force=TimeInForce.DAY,
                stop_price=float(order.stop_price) if order.stop_price else None,
            )
        elif order.type == OrderType.STOP_LIMIT:
            request = StopLimitOrderRequest(
                symbol=ticker,
                qty=float(order.quantity),
                side=side,
                time_in_force=TimeInForce.DAY,
                limit_price=float(order.price) if order.price else None,
                stop_price=float(order.stop_price) if order.stop_price else None,
            )
        else:
            raise ValueError(f"Unsupported order type: {order.type}")

        return client.submit_order(request)

    async def cancel_order(self, order_id: UUID) -> bool:
        alpaca_id = self._order_id_map.get(order_id)
        if not alpaca_id:
            return False
        try:
            await asyncio.to_thread(self._cancel_sync, alpaca_id)
            return True
        except Exception as exc:
            logger.error("Alpaca cancel_order failed: %s", exc)
            return False

    def _cancel_sync(self, alpaca_id: str) -> None:
        from alpaca.trading.client import TradingClient

        client = TradingClient(self._api_key, self._api_secret, paper=self._paper)
        client.cancel_order_by_id(alpaca_id)

    async def get_positions(self, persona_id: UUID) -> dict[str, Decimal]:
        try:
            positions_raw = await asyncio.to_thread(self._get_positions_sync)
            return {
                p.symbol: Decimal(str(p.qty))
                for p in positions_raw
                if Decimal(str(p.qty)) != 0
            }
        except Exception as exc:
            logger.error("Alpaca get_positions failed: %s", exc)
            return {}

    def _get_positions_sync(self) -> list:
        from alpaca.trading.client import TradingClient

        client = TradingClient(self._api_key, self._api_secret, paper=self._paper)
        return client.get_all_positions()

    async def get_balance(self, persona_id: UUID) -> Decimal:
        try:
            balance = await asyncio.to_thread(self._get_balance_sync)
            return balance
        except Exception as exc:
            logger.error("Alpaca get_balance failed: %s", exc)
            return Decimal(0)

    def _get_balance_sync(self) -> Decimal:
        from alpaca.trading.client import TradingClient

        client = TradingClient(self._api_key, self._api_secret, paper=self._paper)
        account = client.get_account()
        return Decimal(str(account.cash))

    @staticmethod
    def to_alpaca_ticker(symbol_key: str) -> str:
        """Convert a symbol_key to Alpaca ticker format.

        ``equities:AAPL/USD@alpaca`` → ``AAPL``
        ``AAPL`` → ``AAPL``  (passthrough)
        """
        key = symbol_key
        # Strip asset_class prefix
        if ":" in key:
            key = key.split(":", 1)[1]
        # Strip venue suffix
        if "@" in key:
            key = key.split("@", 1)[0]
        # Strip quote currency
        if "/" in key:
            key = key.split("/", 1)[0]
        return key


def _map_alpaca_status(status: str) -> OrderStatus:
    """Map Alpaca order status to our OrderStatus."""
    status_lower = status.lower()
    return {
        "new": OrderStatus.OPEN,
        "accepted": OrderStatus.OPEN,
        "partially_filled": OrderStatus.PARTIALLY_FILLED,
        "filled": OrderStatus.FILLED,
        "done_for_day": OrderStatus.FILLED,
        "canceled": OrderStatus.CANCELLED,
        "expired": OrderStatus.CANCELLED,
        "replaced": OrderStatus.CANCELLED,
        "pending_cancel": OrderStatus.OPEN,
        "pending_replace": OrderStatus.OPEN,
        "rejected": OrderStatus.REJECTED,
    }.get(status_lower, OrderStatus.PENDING)
