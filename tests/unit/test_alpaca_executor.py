"""Tests for the Alpaca execution adapter."""

from __future__ import annotations

from decimal import Decimal
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from daytrader.core.types.orders import Order, OrderSide, OrderStatus, OrderType
from daytrader.execution.alpaca import AlpacaExecutor, _map_alpaca_status


# ---------------------------------------------------------------------------
# Ticker conversion
# ---------------------------------------------------------------------------


def test_to_alpaca_ticker_full_key():
    assert AlpacaExecutor.to_alpaca_ticker("equities:AAPL/USD@alpaca") == "AAPL"


def test_to_alpaca_ticker_no_venue():
    assert AlpacaExecutor.to_alpaca_ticker("equities:MSFT/USD") == "MSFT"


def test_to_alpaca_ticker_passthrough():
    assert AlpacaExecutor.to_alpaca_ticker("TSLA") == "TSLA"


def test_to_alpaca_ticker_with_slash():
    assert AlpacaExecutor.to_alpaca_ticker("AAPL/USD") == "AAPL"


# ---------------------------------------------------------------------------
# Status mapping
# ---------------------------------------------------------------------------


def test_map_alpaca_status():
    assert _map_alpaca_status("new") == OrderStatus.OPEN
    assert _map_alpaca_status("filled") == OrderStatus.FILLED
    assert _map_alpaca_status("partially_filled") == OrderStatus.PARTIALLY_FILLED
    assert _map_alpaca_status("canceled") == OrderStatus.CANCELLED
    assert _map_alpaca_status("rejected") == OrderStatus.REJECTED
    assert _map_alpaca_status("unknown") == OrderStatus.PENDING


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


def test_name():
    ex = AlpacaExecutor("key", "secret", paper=True)
    assert ex.name == "alpaca"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_order(
    side=OrderSide.BUY,
    order_type=OrderType.MARKET,
    price=Decimal("150"),
    quantity=Decimal("10"),
) -> Order:
    return Order(
        id=uuid4(),
        persona_id=uuid4(),
        symbol_key="equities:AAPL/USD@alpaca",
        side=side,
        type=order_type,
        quantity=quantity,
        status=OrderStatus.PENDING,
        created_at=datetime.now(timezone.utc),
        price=price,
    )


def _mock_alpaca_order(status="filled", filled_qty=10, avg_price=150.0):
    return SimpleNamespace(
        id="alpaca-order-789",
        status=status,
        filled_qty=filled_qty,
        filled_avg_price=avg_price,
    )


# ---------------------------------------------------------------------------
# submit_order
# ---------------------------------------------------------------------------


async def test_submit_order_market():
    ex = AlpacaExecutor("key", "secret", paper=True)
    order = _make_order()

    mock_result = _mock_alpaca_order()
    with patch.object(ex, "_submit_sync", return_value=mock_result):
        result = await ex.submit_order(order)

    assert result.status == OrderStatus.FILLED
    assert result.filled_quantity == Decimal("10")
    assert result.avg_fill_price == Decimal("150.0")
    assert result.metadata["alpaca_order_id"] == "alpaca-order-789"


async def test_submit_order_limit():
    ex = AlpacaExecutor("key", "secret", paper=True)
    order = _make_order(order_type=OrderType.LIMIT, price=Decimal("148"))

    mock_result = _mock_alpaca_order(status="new", filled_qty=0, avg_price=None)
    with patch.object(ex, "_submit_sync", return_value=mock_result):
        result = await ex.submit_order(order)

    assert result.status == OrderStatus.OPEN
    assert result.filled_quantity == Decimal("0")


async def test_submit_order_api_error():
    ex = AlpacaExecutor("key", "secret", paper=True)
    order = _make_order()

    with patch.object(ex, "_submit_sync", side_effect=Exception("Forbidden")):
        result = await ex.submit_order(order)

    assert result.status == OrderStatus.REJECTED
    assert "Forbidden" in result.reason


# ---------------------------------------------------------------------------
# cancel_order
# ---------------------------------------------------------------------------


async def test_cancel_order_success():
    ex = AlpacaExecutor("key", "secret", paper=True)
    our_id = uuid4()
    ex._order_id_map[our_id] = "alpaca-cancel-id"

    with patch.object(ex, "_cancel_sync"):
        assert await ex.cancel_order(our_id) is True


async def test_cancel_order_unknown_id():
    ex = AlpacaExecutor("key", "secret", paper=True)
    assert await ex.cancel_order(uuid4()) is False


async def test_cancel_order_api_error():
    ex = AlpacaExecutor("key", "secret", paper=True)
    our_id = uuid4()
    ex._order_id_map[our_id] = "alpaca-fail"

    with patch.object(ex, "_cancel_sync", side_effect=Exception("Not Found")):
        assert await ex.cancel_order(our_id) is False


# ---------------------------------------------------------------------------
# get_positions / get_balance
# ---------------------------------------------------------------------------


async def test_get_positions():
    ex = AlpacaExecutor("key", "secret", paper=True)
    mock_positions = [
        SimpleNamespace(symbol="AAPL", qty="10"),
        SimpleNamespace(symbol="MSFT", qty="5"),
        SimpleNamespace(symbol="TSLA", qty="0"),
    ]
    with patch.object(ex, "_get_positions_sync", return_value=mock_positions):
        positions = await ex.get_positions(uuid4())

    assert positions["AAPL"] == Decimal("10")
    assert positions["MSFT"] == Decimal("5")
    assert "TSLA" not in positions  # Zero qty excluded


async def test_get_balance():
    ex = AlpacaExecutor("key", "secret", paper=True)

    with patch.object(ex, "_get_balance_sync", return_value=Decimal("25000")):
        balance = await ex.get_balance(uuid4())

    assert balance == Decimal("25000")


async def test_get_balance_error():
    ex = AlpacaExecutor("key", "secret", paper=True)

    with patch.object(ex, "_get_balance_sync", side_effect=Exception("timeout")):
        balance = await ex.get_balance(uuid4())

    assert balance == Decimal(0)
