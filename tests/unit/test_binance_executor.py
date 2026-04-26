"""Tests for the Binance execution adapter."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock
from uuid import uuid4

from daytrader.core.types.orders import Order, OrderSide, OrderStatus, OrderType
from daytrader.execution.binance import BinanceExecutor, _map_ccxt_status, _order_type_to_ccxt

# ---------------------------------------------------------------------------
# Ticker conversion
# ---------------------------------------------------------------------------


def test_to_binance_ticker_full_key():
    assert BinanceExecutor.to_binance_ticker("crypto:BTC/USDT@binance") == "BTC/USDT"


def test_to_binance_ticker_no_venue():
    assert BinanceExecutor.to_binance_ticker("crypto:ETH/USDT") == "ETH/USDT"


def test_to_binance_ticker_passthrough():
    assert BinanceExecutor.to_binance_ticker("BTC/USDT") == "BTC/USDT"


# ---------------------------------------------------------------------------
# Helper mappings
# ---------------------------------------------------------------------------


def test_order_type_to_ccxt():
    assert _order_type_to_ccxt(OrderType.MARKET) == "market"
    assert _order_type_to_ccxt(OrderType.LIMIT) == "limit"
    assert _order_type_to_ccxt(OrderType.STOP) == "stop"
    assert _order_type_to_ccxt(OrderType.STOP_LIMIT) == "stop_limit"


def test_map_ccxt_status():
    assert _map_ccxt_status("open") == OrderStatus.OPEN
    assert _map_ccxt_status("closed") == OrderStatus.FILLED
    assert _map_ccxt_status("canceled") == OrderStatus.CANCELLED
    assert _map_ccxt_status("rejected") == OrderStatus.REJECTED
    assert _map_ccxt_status("unknown") == OrderStatus.PENDING


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


def test_name():
    ex = BinanceExecutor("key", "secret", testnet=True)
    assert ex.name == "binance"


# ---------------------------------------------------------------------------
# submit_order
# ---------------------------------------------------------------------------


def _make_order(
    side=OrderSide.BUY,
    order_type=OrderType.MARKET,
    price=Decimal("50000"),
    quantity=Decimal("0.1"),
) -> Order:
    return Order(
        id=uuid4(),
        persona_id=uuid4(),
        symbol_key="crypto:BTC/USDT@binance",
        side=side,
        type=order_type,
        quantity=quantity,
        status=OrderStatus.PENDING,
        created_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
        price=price,
    )


async def test_submit_order_success():
    ex = BinanceExecutor("key", "secret", testnet=True)

    mock_exchange = AsyncMock()
    mock_exchange.create_order = AsyncMock(
        return_value={
            "id": "binance-123",
            "status": "closed",
            "filled": 0.1,
            "average": 50000.0,
        }
    )
    ex._exchange = mock_exchange

    order = _make_order()
    result = await ex.submit_order(order)

    assert result.status == OrderStatus.FILLED
    assert result.filled_quantity == Decimal("0.1")
    assert result.avg_fill_price == Decimal("50000.0")
    assert result.metadata["binance_order_id"] == "binance-123"
    assert order.id in ex._order_id_map


async def test_submit_order_api_error():
    ex = BinanceExecutor("key", "secret", testnet=True)

    mock_exchange = AsyncMock()
    mock_exchange.create_order = AsyncMock(side_effect=Exception("API down"))
    ex._exchange = mock_exchange

    order = _make_order()
    result = await ex.submit_order(order)

    assert result.status == OrderStatus.REJECTED
    assert "API down" in result.reason


# ---------------------------------------------------------------------------
# cancel_order
# ---------------------------------------------------------------------------


async def test_cancel_order_success():
    ex = BinanceExecutor("key", "secret", testnet=True)
    our_id = uuid4()
    ex._order_id_map[our_id] = "binance-456"

    mock_exchange = AsyncMock()
    mock_exchange.cancel_order = AsyncMock(return_value={})
    ex._exchange = mock_exchange

    assert await ex.cancel_order(our_id) is True


async def test_cancel_order_unknown_id():
    ex = BinanceExecutor("key", "secret", testnet=True)
    assert await ex.cancel_order(uuid4()) is False


# ---------------------------------------------------------------------------
# get_positions / get_balance
# ---------------------------------------------------------------------------


async def test_get_positions():
    ex = BinanceExecutor("key", "secret", testnet=True)
    mock_exchange = AsyncMock()
    mock_exchange.fetch_balance = AsyncMock(
        return_value={
            "total": {"BTC": 0.5, "ETH": 1.2, "USDT": 1000.0},
        }
    )
    ex._exchange = mock_exchange

    positions = await ex.get_positions(uuid4())
    assert positions["BTC"] == Decimal("0.5")
    assert positions["ETH"] == Decimal("1.2")
    assert "USDT" not in positions  # Quote currency excluded


async def test_get_balance():
    ex = BinanceExecutor("key", "secret", testnet=True)
    mock_exchange = AsyncMock()
    mock_exchange.fetch_balance = AsyncMock(
        return_value={"free": {"USDT": 5000.0}}
    )
    ex._exchange = mock_exchange

    balance = await ex.get_balance(uuid4())
    assert balance == Decimal("5000.0")


async def test_get_balance_error_returns_zero():
    ex = BinanceExecutor("key", "secret", testnet=True)
    mock_exchange = AsyncMock()
    mock_exchange.fetch_balance = AsyncMock(side_effect=Exception("timeout"))
    ex._exchange = mock_exchange

    balance = await ex.get_balance(uuid4())
    assert balance == Decimal(0)
