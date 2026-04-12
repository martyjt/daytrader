"""Walk-forward testing engine.

Splits historical data into K folds, runs the backtest engine on each
fold (train then test), and aggregates out-of-sample (OOS) results.
This is the Ritual's second gate: strategies must prove they generalise
beyond in-sample data.

Supports anchored (expanding) and sliding window splits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import polars as pl

from ..algorithms.base import Algorithm
from ..core.types.bars import Timeframe
from ..core.types.symbols import Symbol
from .engine import BacktestEngine, BacktestResult
from .fees import FeeModel, FeeSchedule, VENUE_PROFILES
from .risk import RiskConfig


@dataclass(frozen=True)
class WalkForwardConfig:
    """Configuration for walk-forward analysis."""

    n_folds: int = 5
    anchored: bool = True
    min_train_bars: int = 100
    gap_bars: int = 0


@dataclass
class FoldResult:
    """Result from one walk-forward fold."""

    fold_index: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_result: BacktestResult
    test_result: BacktestResult
    oos_sharpe: float
    oos_return_pct: float


@dataclass
class WalkForwardResult:
    """Aggregate result from a complete walk-forward analysis."""

    folds: list[FoldResult]
    aggregate_oos_sharpe: float
    aggregate_oos_return_pct: float
    aggregate_oos_max_drawdown_pct: float
    per_fold_oos_sharpes: list[float]
    config: WalkForwardConfig
    total_bars: int
    oos_equity_curve: list[float] = field(default_factory=list)
    oos_timestamps: list[Any] = field(default_factory=list)


class WalkForwardEngine:
    """Orchestrates walk-forward analysis across K folds.

    For each fold:
    1. Split data into train and test sets
    2. If the algorithm has a ``train()`` method, call it on train data
    3. Run the backtest engine on the test set
    4. Collect OOS metrics

    Usage::

        engine = WalkForwardEngine()
        result = await engine.run(
            algorithm=my_algo,
            symbol=Symbol.parse("BTC-USD"),
            timeframe=Timeframe.D1,
            data=ohlcv_df,
            config=WalkForwardConfig(n_folds=5),
        )
    """

    def __init__(self, backtest_engine: BacktestEngine | None = None) -> None:
        self._engine = backtest_engine or BacktestEngine()

    async def run(
        self,
        *,
        algorithm: Algorithm,
        symbol: Symbol,
        timeframe: Timeframe,
        data: pl.DataFrame,
        config: WalkForwardConfig | None = None,
        initial_capital: float = 10_000.0,
        commission_bps: float = 10.0,
        venue: str | None = None,
        fee_model: FeeModel | None = None,
        risk_config: RiskConfig | None = None,
    ) -> WalkForwardResult:
        if config is None:
            config = WalkForwardConfig()

        if fee_model is None:
            if venue and venue in VENUE_PROFILES:
                fee_model = FeeModel(VENUE_PROFILES[venue])
            else:
                fee_model = FeeModel(FeeSchedule.from_flat_bps(commission_bps))

        resolved_venue = fee_model.schedule.venue
        total_bars = len(data)

        # Auto-scale config to fit available data. Walk-forward requires at
        # least ~min_train_bars + n_folds * 2 bars; if we don't have that,
        # reduce min_train_bars (and folds if needed) so the split can run
        # on short histories (e.g. weekly timeframes over a year).
        min_bars_needed = config.min_train_bars + config.n_folds * 2
        if total_bars < min_bars_needed:
            # Reserve half the data for initial training, split the rest
            # across the folds. Keep at least 10 bars for training and
            # 2 bars per test fold.
            adjusted_min_train = max(10, total_bars // 2)
            remaining = total_bars - adjusted_min_train
            adjusted_folds = min(config.n_folds, max(1, remaining // 2))
            config = WalkForwardConfig(
                n_folds=adjusted_folds,
                anchored=config.anchored,
                min_train_bars=adjusted_min_train,
                gap_bars=config.gap_bars,
            )

        splits = self._split_folds(data, config)
        if not splits:
            raise ValueError(
                f"Cannot create walk-forward folds from {total_bars} bars. "
                f"Try a longer date range or a shorter timeframe."
            )

        fold_results: list[FoldResult] = []
        all_oos_equity: list[float] = []
        all_oos_timestamps: list[Any] = []

        import asyncio

        for fold_idx, (train_df, test_df) in enumerate(splits):
            # Train the algorithm if it supports it
            if hasattr(algorithm, "train") and callable(algorithm.train):
                await asyncio.to_thread(algorithm.train, train_df)

            # Run backtest on train set (in worker thread to keep
            # NiceGUI's websocket heartbeat alive during long runs).
            train_result = await asyncio.to_thread(
                self._engine._simulate,
                algorithm, symbol, timeframe, train_df,
                initial_capital, fee_model, resolved_venue,
                risk_config=risk_config,
            )

            # Run backtest on test set
            test_result = await asyncio.to_thread(
                self._engine._simulate,
                algorithm, symbol, timeframe, test_df,
                initial_capital, fee_model, resolved_venue,
                risk_config=risk_config,
            )

            train_timestamps = train_df["timestamp"].to_list()
            test_timestamps = test_df["timestamp"].to_list()

            fold = FoldResult(
                fold_index=fold_idx,
                train_start=train_timestamps[0],
                train_end=train_timestamps[-1],
                test_start=test_timestamps[0],
                test_end=test_timestamps[-1],
                train_result=train_result,
                test_result=test_result,
                oos_sharpe=test_result.kpis.get("sharpe_ratio", 0.0),
                oos_return_pct=test_result.kpis.get("total_return_pct", 0.0),
            )
            fold_results.append(fold)

            all_oos_equity.extend(test_result.equity_curve)
            all_oos_timestamps.extend(test_result.timestamps)

        # Aggregate OOS metrics from concatenated equity curve
        per_fold_sharpes = [f.oos_sharpe for f in fold_results]
        agg_sharpe, agg_return, agg_drawdown = _aggregate_oos_metrics(
            all_oos_equity, initial_capital,
        )

        return WalkForwardResult(
            folds=fold_results,
            aggregate_oos_sharpe=agg_sharpe,
            aggregate_oos_return_pct=agg_return,
            aggregate_oos_max_drawdown_pct=agg_drawdown,
            per_fold_oos_sharpes=per_fold_sharpes,
            config=config,
            total_bars=total_bars,
            oos_equity_curve=all_oos_equity,
            oos_timestamps=all_oos_timestamps,
        )

    def _split_folds(
        self,
        data: pl.DataFrame,
        config: WalkForwardConfig,
    ) -> list[tuple[pl.DataFrame, pl.DataFrame]]:
        """Split data into (train, test) pairs.

        Anchored (expanding window):
            Fold k: train=[0, boundary_k), test=[boundary_k + gap, boundary_{k+1})

        Sliding window:
            Fold k: train=[boundary_k - train_size, boundary_k), test=[boundary_k + gap, boundary_{k+1})
        """
        n = len(data)
        n_folds = config.n_folds
        gap = config.gap_bars

        # Reserve first portion for initial training, split rest into test folds
        test_size = n // (n_folds + 1)
        if test_size < 2:
            return []

        initial_train_size = test_size
        if initial_train_size < config.min_train_bars:
            # Adjust: use min_train_bars for initial training
            initial_train_size = config.min_train_bars
            remaining = n - initial_train_size
            test_size = remaining // n_folds
            if test_size < 2:
                return []

        splits: list[tuple[pl.DataFrame, pl.DataFrame]] = []

        for k in range(n_folds):
            if config.anchored:
                train_start = 0
            else:
                train_start = max(0, initial_train_size + k * test_size - initial_train_size)

            train_end = initial_train_size + k * test_size
            test_start = train_end + gap
            test_end = min(train_end + test_size, n)

            if test_start >= n or test_start >= test_end:
                break

            train_df = data.slice(train_start, train_end - train_start)
            test_df = data.slice(test_start, test_end - test_start)

            if len(train_df) < config.min_train_bars:
                continue

            splits.append((train_df, test_df))

        return splits


def _aggregate_oos_metrics(
    oos_equity: list[float],
    initial_capital: float,
) -> tuple[float, float, float]:
    """Compute aggregate Sharpe, return, and drawdown from concatenated OOS curve."""
    if len(oos_equity) < 2:
        return 0.0, 0.0, 0.0

    eq = np.array(oos_equity, dtype=float)
    final = eq[-1]

    # Return
    total_return = (final - initial_capital) / initial_capital * 100

    # Sharpe from returns
    returns = np.diff(eq) / eq[:-1]
    returns = returns[np.isfinite(returns)]
    if len(returns) > 1 and np.std(returns) > 0:
        sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(252))
    else:
        sharpe = 0.0

    # Max drawdown
    peak = np.maximum.accumulate(eq)
    drawdown = (eq - peak) / peak
    max_dd = float(np.min(drawdown) * 100)

    return round(sharpe, 2), round(total_return, 2), round(max_dd, 2)
