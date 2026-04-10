"""Tests for the fee model and venue profiles."""

import pytest

from daytrader.backtest.fees import (
    FeeModel,
    FeeSchedule,
    VENUE_PROFILES,
)


# ---- FeeSchedule ---------------------------------------------------------


def test_from_flat_bps_backward_compat():
    s = FeeSchedule.from_flat_bps(10)
    assert s.maker_bps == 10
    assert s.taker_bps == 10
    assert s.spread_bps == 0
    assert s.slippage_base_bps == 0
    assert s.venue == "custom"


def test_total_round_trip_bps():
    s = FeeSchedule(
        venue="test",
        maker_bps=5, taker_bps=10,
        spread_bps=4, slippage_model="fixed", slippage_base_bps=3,
    )
    # one side = taker(10) + spread/2(2) + slippage(3) = 15
    assert s.total_one_side_bps == 15
    assert s.total_round_trip_bps == 30


# ---- FeeModel core -------------------------------------------------------


def test_flat_fee_model():
    model = FeeModel(FeeSchedule.from_flat_bps(10))
    cost = model.trade_cost(10_000)
    # 10 bps of $10k = $10
    assert cost == pytest.approx(10.0)


def test_round_trip_is_double():
    model = FeeModel(FeeSchedule.from_flat_bps(10))
    assert model.round_trip_cost(10_000) == pytest.approx(20.0)


def test_maker_cheaper_than_taker():
    s = FeeSchedule(
        venue="test", maker_bps=2, taker_bps=10,
        spread_bps=0, slippage_model="fixed", slippage_base_bps=0,
    )
    model = FeeModel(s)
    assert model.trade_cost(10_000, maker=True) < model.trade_cost(10_000)


def test_spread_adds_to_cost():
    no_spread = FeeModel(FeeSchedule(
        venue="a", maker_bps=10, taker_bps=10,
        spread_bps=0, slippage_model="fixed", slippage_base_bps=0,
    ))
    with_spread = FeeModel(FeeSchedule(
        venue="b", maker_bps=10, taker_bps=10,
        spread_bps=10, slippage_model="fixed", slippage_base_bps=0,
    ))
    assert with_spread.trade_cost(10_000) > no_spread.trade_cost(10_000)


def test_min_trade_fee_enforced():
    model = FeeModel(FeeSchedule(
        venue="ib", maker_bps=0, taker_bps=0,
        spread_bps=0, slippage_model="fixed", slippage_base_bps=0,
        min_trade_fee=1.0,
    ))
    # Even a tiny trade pays at least $1
    assert model.trade_cost(10) == 1.0


# ---- Slippage models ------------------------------------------------------


def test_fixed_slippage():
    model = FeeModel(FeeSchedule(
        venue="test", maker_bps=0, taker_bps=0,
        spread_bps=0, slippage_model="fixed", slippage_base_bps=5,
    ))
    cost = model.trade_cost(10_000)
    assert cost == pytest.approx(5.0)  # 5 bps of $10k


def test_volatility_scaled_slippage():
    model = FeeModel(FeeSchedule(
        venue="test", maker_bps=0, taker_bps=0,
        spread_bps=0, slippage_model="volatility_scaled", slippage_base_bps=5,
    ))
    calm = model.trade_cost(10_000, volatility_pct=0.0)
    volatile = model.trade_cost(10_000, volatility_pct=4.0)
    assert volatile > calm


def test_volume_scaled_slippage():
    model = FeeModel(FeeSchedule(
        venue="test", maker_bps=0, taker_bps=0,
        spread_bps=0, slippage_model="volume_scaled", slippage_base_bps=5,
    ))
    small = model.trade_cost(10_000, volume_ratio=0.01)
    large = model.trade_cost(10_000, volume_ratio=1.0)
    assert large > small


# ---- Venue profiles -------------------------------------------------------


def test_binance_cheaper_than_coinbase():
    bn = FeeModel(VENUE_PROFILES["binance_spot"])
    cb = FeeModel(VENUE_PROFILES["coinbase"])
    assert bn.trade_cost(10_000) < cb.trade_cost(10_000)


def test_alpaca_zero_commission():
    al = FeeModel(VENUE_PROFILES["alpaca"])
    # Alpaca has 0 commission — cost is purely spread + slippage
    assert VENUE_PROFILES["alpaca"].maker_bps == 0
    assert VENUE_PROFILES["alpaca"].taker_bps == 0
    # Still cheaper than Coinbase even with spread+slippage
    cb = FeeModel(VENUE_PROFILES["coinbase"])
    assert al.trade_cost(10_000) < cb.trade_cost(10_000)


def test_zero_fees_profile():
    model = FeeModel(VENUE_PROFILES["zero_fees"])
    assert model.trade_cost(10_000) == 0.0
    assert model.round_trip_cost(10_000) == 0.0


def test_effective_round_trip_bps():
    model = FeeModel(VENUE_PROFILES["binance_spot"])
    bps = model.effective_round_trip_bps(10_000)
    # Should be around 30 bps (10 commission + 1 spread/2 + 3 slippage) × 2
    assert 20 < bps < 40


def test_all_profiles_loadable():
    for name, schedule in VENUE_PROFILES.items():
        model = FeeModel(schedule)
        cost = model.trade_cost(10_000)
        assert cost >= 0, f"{name} produced negative cost"
