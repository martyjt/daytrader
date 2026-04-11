"""Binance execution adapter — live crypto trading via ccxt async.

Submits and cancels orders on Binance (or Binance testnet) using the
ccxt async API.  Testnet mode is enabled by default for safety.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from decimal import Decimal
from typing import Any
from uuid import UUID

from ..core.types.orders import Order, OrderSide, OrderStatus, OrderType
from .base import ExecutionAdapter

logger = logging.getLogger(__name__)


class BinanceExecutor(ExecutionAdapter):
    """Live crypto execution via Binance (ccxt async)."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool = True,
    ) -> None:
        self._api_key = api_key
        self._api_secret = api_secret
        self._testnet = testnet
        self._exchange: Any = None
        # Map our order UUIDs to Binance's order IDs for cancel support.
        self._order_id_map: dict[UUID, str] = {}

    @property
    def name(self) -> str:
        return "binance"

    async def _get_exchange(self) -> Any:
        """Lazy-init the ccxt async exchange instance."""
        if self._exchange is None:
            import ccxt.async_support as ccxt_async

            self._exchange = ccxt_async.binance(
                {
                    "apiKey": self._api_key,
                    "secret": self._api_secret,
                    "sandbox": self._testnet,
                    "enableRateLimit": True,
                }
            )
        return self._exchange

    async def submit_order(self, order: Order) -> Order:
        try:
            exchange = await self._get_exchange()
            ticker = self.to_binance_ticker(order.symbol_key)
            side = "buy" if order.side == OrderSide.BUY else "sell"
            ccxt_type = _order_type_to_ccxt(order.type)
            price = float(order.price) if order.price else None

            params: dict[str, Any] = {}
            if order.type == OrderType.STOP_LIMIT and order.stop_price:
                params["stopPrice"] = float(order.stop_price)

            result = await exchange.create_order(
                symbol=ticker,
                type=ccxt_type,
                side=side,
                amount=float(order.quantity),
                price=price,
                params=params,
            )

            binance_id = str(result.get("id", ""))
            self._order_id_map[order.id] = binance_id

            status = _map_ccxt_status(result.get("status", ""))
            filled_qty = Decimal(str(result.get("filled", 0)))
            avg_price = (
                Decimal(str(result["average"]))
                if result.get("average")
                else order.price
            )

            return replace(
                order,
                status=status,
                filled_quantity=filled_qty,
                avg_fill_price=avg_price,
                metadata={**order.metadata, "binance_order_id": binance_id},
            )

        except Exception as exc:
            logger.error("Binance submit_order failed: %s", exc)
            return replace(
                order,
                status=OrderStatus.REJECTED,
                reason=f"Binance API error: {exc}",
            )

    async def cancel_order(self, order_id: UUID) -> bool:
        binance_id = self._order_id_map.get(order_id)
        if not binance_id:
            return False
        try:
            exchange = await self._get_exchange()
            await exchange.cancel_order(binance_id)
            return True
        except Exception as exc:
            logger.error("Binance cancel_order failed: %s", exc)
            return False

    async def get_positions(self, persona_id: UUID) -> dict[str, Decimal]:
        try:
            exchange = await self._get_exchange()
            balance = await exchange.fetch_balance()
            positions: dict[str, Decimal] = {}
            for asset, info in balance.get("total", {}).items():
                qty = Decimal(str(info))
                if qty != 0 and asset not in ("USDT", "USD", "BUSD"):
                    positions[asset] = qty
            return positions
        except Exception as exc:
            logger.error("Binance get_positions failed: %s", exc)
            return {}

    async def get_balance(self, persona_id: UUID) -> Decimal:
        try:
            exchange = await self._get_exchange()
            balance = await exchange.fetch_balance()
            free_usdt = balance.get("free", {}).get("USDT", 0)
            return Decimal(str(free_usdt))
        except Exception as exc:
            logger.error("Binance get_balance failed: %s", exc)
            return Decimal(0)

    async def close(self) -> None:
        """Close the ccxt exchange connection."""
        if self._exchange is not None:
            await self._exchange.close()
            self._exchange = None

    @staticmethod
    def to_binance_ticker(symbol_key: str) -> str:
        """Convert a symbol_key to ccxt Binance format.

        ``crypto:BTC/USDT@binance`` → ``BTC/USDT``
        ``BTC/USDT`` → ``BTC/USDT``  (passthrough)
        """
        key = symbol_key
        # Strip asset_class prefix
        if ":" in key:
            key = key.split(":", 1)[1]
        # Strip venue suffix
        if "@" in key:
            key = key.split("@", 1)[0]
        return key


def _order_type_to_ccxt(order_type: OrderType) -> str:
    """Map our OrderType to ccxt type string."""
    return {
        OrderType.MARKET: "market",
        OrderType.LIMIT: "limit",
        OrderType.STOP: "stop",
        OrderType.STOP_LIMIT: "stop_limit",
    }.get(order_type, "market")


def _map_ccxt_status(ccxt_status: str) -> OrderStatus:
    """Map ccxt order status string to our OrderStatus."""
    return {
        "open": OrderStatus.OPEN,
        "closed": OrderStatus.FILLED,
        "canceled": OrderStatus.CANCELLED,
        "expired": OrderStatus.CANCELLED,
        "rejected": OrderStatus.REJECTED,
    }.get(ccxt_status, OrderStatus.PENDING)
