"""Tests for the walk-forward engine."""

from datetime import datetime, timedelta

import polars as pl
import pytest

from daytrader.algorithms.builtin.buy_hold import BuyHoldAlgorithm
from daytrader.backtest.walk_forward import (
    WalkForwardConfig,
    WalkForwardEngine,
)
from daytrader.core.types.bars import Timeframe
from daytrader.core.types.symbols import AssetClass, Symbol


def _linear_ohlcv(n: int = 200, start_price: float = 100.0) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n)],
            "open": [start_price + i * 0.5 for i in range(n)],
            "high": [start_price + i * 0.5 + 2 for i in range(n)],
            "low": [start_price + i * 0.5 - 1 for i in range(n)],
            "close": [start_price + i * 0.5 for i in range(n)],
            "volume": [1000.0] * n,
        }
    )


def _symbol() -> Symbol:
    return Symbol("TEST", "USD", AssetClass.CRYPTO)


# ---------------------------------------------------------------------------
# Fold splitting
# ---------------------------------------------------------------------------


def test_split_folds_anchored_correct_count():
    """Anchored split should produce n_folds (train, test) pairs."""
    engine = WalkForwardEngine()
    data = _linear_ohlcv(300)
    config = WalkForwardConfig(n_folds=5, anchored=True, min_train_bars=10)
    splits = engine._split_folds(data, config)
    assert len(splits) == 5


def test_split_folds_sliding_correct_count():
    """Sliding split should produce n_folds pairs."""
    engine = WalkForwardEngine()
    data = _linear_ohlcv(300)
    config = WalkForwardConfig(n_folds=5, anchored=False, min_train_bars=10)
    splits = engine._split_folds(data, config)
    assert len(splits) == 5


def test_split_folds_anchored_expanding_train():
    """In anchored mode, train set should grow with each fold."""
    engine = WalkForwardEngine()
    data = _linear_ohlcv(300)
    config = WalkForwardConfig(n_folds=3, anchored=True, min_train_bars=10)
    splits = engine._split_folds(data, config)
    train_sizes = [len(train) for train, _ in splits]
    assert train_sizes == sorted(train_sizes)
    assert train_sizes[-1] > train_sizes[0]


def test_split_folds_non_overlapping_test():
    """Test sets should not overlap across folds."""
    engine = WalkForwardEngine()
    data = _linear_ohlcv(300)
    config = WalkForwardConfig(n_folds=5, anchored=True, min_train_bars=10)
    splits = engine._split_folds(data, config)
    test_indices = []
    for _, test_df in splits:
        timestamps = test_df["timestamp"].to_list()
        test_indices.extend(timestamps)
    assert len(test_indices) == len(set(test_indices))


def test_split_folds_with_gap():
    """Gap bars should create a gap between train and test."""
    engine = WalkForwardEngine()
    data = _linear_ohlcv(300)
    config = WalkForwardConfig(n_folds=3, anchored=True, min_train_bars=10, gap_bars=5)
    splits = engine._split_folds(data, config)
    assert len(splits) >= 1
    for train_df, test_df in splits:
        train_end = train_df["timestamp"].to_list()[-1]
        test_start = test_df["timestamp"].to_list()[0]
        assert test_start > train_end


def test_split_folds_too_small_dataset():
    """Too few bars for requested folds should return empty list."""
    engine = WalkForwardEngine()
    data = _linear_ohlcv(10)
    config = WalkForwardConfig(n_folds=5, anchored=True, min_train_bars=100)
    splits = engine._split_folds(data, config)
    assert splits == []


# ---------------------------------------------------------------------------
# Full walk-forward runs
# ---------------------------------------------------------------------------


async def test_walk_forward_buy_hold_rising():
    """Walk-forward with BuyHold on rising data should produce results."""
    engine = WalkForwardEngine()
    data = _linear_ohlcv(300)
    config = WalkForwardConfig(n_folds=3, anchored=True, min_train_bars=10)

    result = await engine.run(
        algorithm=BuyHoldAlgorithm(),
        symbol=_symbol(),
        timeframe=Timeframe.D1,
        data=data,
        config=config,
        commission_bps=0,
    )

    assert len(result.folds) == 3
    assert len(result.per_fold_oos_sharpes) == 3
    assert result.total_bars == 300
    assert len(result.oos_equity_curve) > 0


async def test_walk_forward_oos_equity_curve_length():
    """Concatenated OOS equity curve should equal sum of test fold sizes."""
    engine = WalkForwardEngine()
    data = _linear_ohlcv(300)
    config = WalkForwardConfig(n_folds=3, anchored=True, min_train_bars=10)

    result = await engine.run(
        algorithm=BuyHoldAlgorithm(),
        symbol=_symbol(),
        timeframe=Timeframe.D1,
        data=data,
        config=config,
        commission_bps=0,
    )

    expected_len = sum(len(f.test_result.equity_curve) for f in result.folds)
    assert len(result.oos_equity_curve) == expected_len


async def test_walk_forward_fold_periods_are_sequential():
    """Fold test periods should be sequential (non-overlapping)."""
    engine = WalkForwardEngine()
    data = _linear_ohlcv(300)
    config = WalkForwardConfig(n_folds=3, anchored=True, min_train_bars=10)

    result = await engine.run(
        algorithm=BuyHoldAlgorithm(),
        symbol=_symbol(),
        timeframe=Timeframe.D1,
        data=data,
        config=config,
        commission_bps=0,
    )

    for i in range(1, len(result.folds)):
        prev = result.folds[i - 1]
        curr = result.folds[i]
        assert curr.test_start >= prev.test_end


async def test_walk_forward_too_small_raises():
    """Walk-forward with insufficient data should raise ValueError."""
    engine = WalkForwardEngine()
    data = _linear_ohlcv(10)
    config = WalkForwardConfig(n_folds=5, anchored=True, min_train_bars=100)

    with pytest.raises(ValueError, match="Cannot create"):
        await engine.run(
            algorithm=BuyHoldAlgorithm(),
            symbol=_symbol(),
            timeframe=Timeframe.D1,
            data=data,
            config=config,
        )


async def test_walk_forward_default_config():
    """Walk-forward with default config should work on sufficient data."""
    engine = WalkForwardEngine()
    data = _linear_ohlcv(700)

    result = await engine.run(
        algorithm=BuyHoldAlgorithm(),
        symbol=_symbol(),
        timeframe=Timeframe.D1,
        data=data,
        commission_bps=0,
    )

    assert len(result.folds) == 5
