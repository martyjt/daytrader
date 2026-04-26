"""Tests for the paper trading executor."""

from datetime import UTC, datetime
from decimal import Decimal
from uuid import uuid4

from daytrader.core.types.orders import Order, OrderSide, OrderStatus, OrderType
from daytrader.execution.paper import PaperExecutor


def _make_order(
    persona_id,
    *,
    side=OrderSide.BUY,
    price=Decimal("100"),
    quantity=Decimal("10"),
    symbol="crypto:BTC/USDT",
) -> Order:
    return Order(
        id=uuid4(),
        persona_id=persona_id,
        symbol_key=symbol,
        side=side,
        type=OrderType.MARKET,
        quantity=quantity,
        status=OrderStatus.PENDING,
        created_at=datetime.now(UTC),
        price=price,
    )


async def test_buy_reduces_cash():
    pid = uuid4()
    ex = PaperExecutor()
    ex.initialize_persona(pid, Decimal("10000"))

    order = _make_order(pid, price=Decimal("100"), quantity=Decimal("10"))
    filled = await ex.submit_order(order)

    assert filled.status == OrderStatus.FILLED
    assert await ex.get_balance(pid) == Decimal("9000")
    positions = await ex.get_positions(pid)
    assert positions["crypto:BTC/USDT"] == Decimal("10")


async def test_sell_increases_cash():
    pid = uuid4()
    ex = PaperExecutor()
    ex.initialize_persona(pid, Decimal("10000"))

    # Buy first
    buy = _make_order(pid, side=OrderSide.BUY, price=Decimal("100"), quantity=Decimal("10"))
    await ex.submit_order(buy)

    # Sell
    sell = _make_order(pid, side=OrderSide.SELL, price=Decimal("110"), quantity=Decimal("10"))
    filled = await ex.submit_order(sell)

    assert filled.status == OrderStatus.FILLED
    assert await ex.get_balance(pid) == Decimal("10100")  # 9000 + 1100
    positions = await ex.get_positions(pid)
    assert len(positions) == 0  # Flat


async def test_insufficient_funds_rejected():
    pid = uuid4()
    ex = PaperExecutor()
    ex.initialize_persona(pid, Decimal("100"))

    order = _make_order(pid, price=Decimal("100"), quantity=Decimal("10"))
    filled = await ex.submit_order(order)

    assert filled.status == OrderStatus.REJECTED
    assert "Insufficient funds" in filled.reason


async def test_insufficient_position_rejected():
    pid = uuid4()
    ex = PaperExecutor()
    ex.initialize_persona(pid, Decimal("10000"))

    sell = _make_order(pid, side=OrderSide.SELL, price=Decimal("100"), quantity=Decimal("10"))
    filled = await ex.submit_order(sell)

    assert filled.status == OrderStatus.REJECTED
    assert "Insufficient position" in filled.reason


async def test_uninitialized_persona_rejected():
    pid = uuid4()
    ex = PaperExecutor()

    order = _make_order(pid)
    filled = await ex.submit_order(order)

    assert filled.status == OrderStatus.REJECTED


async def test_order_history():
    pid = uuid4()
    ex = PaperExecutor()
    ex.initialize_persona(pid, Decimal("10000"))

    buy = _make_order(pid, price=Decimal("100"), quantity=Decimal("5"))
    await ex.submit_order(buy)

    history = ex.get_order_history(pid)
    assert len(history) == 1
    assert history[0].status == OrderStatus.FILLED
