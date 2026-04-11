"""Tests for the Regime-Switching HMM algorithm."""

from datetime import datetime, timedelta
from decimal import Decimal
from uuid import uuid4

import numpy as np
import polars as pl

from daytrader.algorithms.builtin.regime_hmm import (
    RegimeHMMAlgorithm,
    _build_hmm_features,
)
from daytrader.algorithms.registry import AlgorithmRegistry
from daytrader.backtest.engine import BacktestEngine
from daytrader.core.context import AlgorithmContext
from daytrader.core.types.bars import Bar, Timeframe
from daytrader.core.types.symbols import AssetClass, Symbol


def _trending_ohlcv(n: int = 300, start_price: float = 100.0) -> pl.DataFrame:
    """Generate trending data with some noise for HMM training."""
    np.random.seed(42)
    prices = start_price + np.cumsum(np.random.randn(n) * 0.5 + 0.1)
    prices = np.maximum(prices, 1.0)
    return pl.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n)],
            "open": prices.tolist(),
            "high": (prices + np.abs(np.random.randn(n)) * 2).tolist(),
            "low": (prices - np.abs(np.random.randn(n)) * 2).tolist(),
            "close": prices.tolist(),
            "volume": (1000 + np.abs(np.random.randn(n)) * 500).tolist(),
        }
    )


def _regime_ohlcv(n: int = 300) -> pl.DataFrame:
    """Generate data with distinct regimes: up-trend then down-trend."""
    np.random.seed(99)
    half = n // 2
    # First half: uptrend
    up = 100.0 + np.cumsum(np.random.randn(half) * 0.3 + 0.3)
    # Second half: downtrend
    down = up[-1] + np.cumsum(np.random.randn(n - half) * 0.3 - 0.3)
    prices = np.concatenate([up, down])
    prices = np.maximum(prices, 1.0)
    return pl.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n)],
            "open": prices.tolist(),
            "high": (prices + np.abs(np.random.randn(n)) * 2).tolist(),
            "low": (prices - np.abs(np.random.randn(n)) * 2).tolist(),
            "close": prices.tolist(),
            "volume": (1000 + np.abs(np.random.randn(n)) * 500).tolist(),
        }
    )


def _symbol() -> Symbol:
    return Symbol("TEST", "USD", AssetClass.CRYPTO)


def _make_ctx(
    algo_id: str,
    data: pl.DataFrame,
    bar_idx: int = -1,
) -> tuple[AlgorithmContext, list, list]:
    """Build an AlgorithmContext from a DataFrame row."""
    closes = data["close"].to_numpy().astype(float)
    highs = data["high"].to_numpy().astype(float)
    lows = data["low"].to_numpy().astype(float)
    volumes = data["volume"].to_numpy().astype(float)

    emitted: list = []
    logs: list = []
    ctx = AlgorithmContext(
        tenant_id=uuid4(),
        persona_id=uuid4(),
        algorithm_id=algo_id,
        symbol=_symbol(),
        timeframe=Timeframe.D1,
        now=datetime(2024, 7, 10),
        bar=Bar(
            timestamp=datetime(2024, 7, 10),
            open=Decimal(str(closes[bar_idx])),
            high=Decimal(str(highs[bar_idx])),
            low=Decimal(str(lows[bar_idx])),
            close=Decimal(str(closes[bar_idx])),
            volume=Decimal(str(volumes[bar_idx])),
        ),
        history_arrays={
            "close": closes,
            "high": highs,
            "low": lows,
            "volume": volumes,
        },
        features={},
        params={},
        emit_fn=emitted.append,
        log_fn=lambda msg, fields: logs.append({"message": msg, **fields}),
    )
    return ctx, emitted, logs


# ---------------------------------------------------------------------------
# Manifest and properties
# ---------------------------------------------------------------------------


def test_manifest():
    algo = RegimeHMMAlgorithm()
    m = algo.manifest
    assert m.id == "regime_hmm"
    assert m.name == "Regime-Switching HMM"
    assert "crypto" in m.asset_classes
    assert "equities" in m.asset_classes
    assert len(m.params) == 4


def test_warmup_bars():
    algo = RegimeHMMAlgorithm(lookback=60)
    assert algo.warmup_bars() == 60

    algo2 = RegimeHMMAlgorithm(lookback=100)
    assert algo2.warmup_bars() == 100


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def test_feature_matrix_shape():
    n = 100
    closes = np.arange(50.0, 50.0 + n, dtype=float)
    highs = closes + 1
    lows = closes - 1
    volumes = np.ones(n) * 1000

    features = _build_hmm_features(closes, highs, lows, volumes)
    assert features.shape == (n, 6)


def test_feature_matrix_has_valid_values():
    """After warmup, features should not be all NaN."""
    data = _trending_ohlcv(100)
    closes = data["close"].to_numpy().astype(float)
    highs = data["high"].to_numpy().astype(float)
    lows = data["low"].to_numpy().astype(float)
    volumes = data["volume"].to_numpy().astype(float)

    features = _build_hmm_features(closes, highs, lows, volumes)
    valid_mask = ~np.isnan(features).any(axis=1)
    assert valid_mask.sum() > 50  # at least half should be valid


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def test_train_sets_is_trained():
    algo = RegimeHMMAlgorithm(lookback=30, auto_train=False)
    data = _trending_ohlcv(300)
    assert not algo._is_trained
    algo.train(data)
    assert algo._is_trained


def test_train_insufficient_data():
    """Training with too few rows should not crash and leave untrained."""
    algo = RegimeHMMAlgorithm(lookback=30, auto_train=False)
    data = _trending_ohlcv(20)
    algo.train(data)
    assert not algo._is_trained


def test_regime_labels_assigned():
    """After training, all states should have labels."""
    algo = RegimeHMMAlgorithm(n_regimes=3, lookback=30, auto_train=False)
    data = _trending_ohlcv(300)
    algo.train(data)
    assert algo._is_trained
    assert len(algo._regime_labels) == 3
    assert "bull" in algo._regime_labels.values()
    assert "bear" in algo._regime_labels.values()
    assert "sideways" in algo._regime_labels.values()


def test_regime_labels_two_states():
    algo = RegimeHMMAlgorithm(n_regimes=2, lookback=30, auto_train=False)
    data = _trending_ohlcv(300)
    algo.train(data)
    assert algo._is_trained
    assert len(algo._regime_labels) == 2
    assert "bull" in algo._regime_labels.values()
    assert "bear" in algo._regime_labels.values()


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def test_on_bar_before_training_returns_none():
    """Without training and auto_train=False, on_bar should return None."""
    algo = RegimeHMMAlgorithm(auto_train=False)
    data = _trending_ohlcv(100)
    ctx, _, _ = _make_ctx("regime_hmm", data)
    result = algo.on_bar(ctx)
    assert result is None


def test_on_bar_after_training_emits():
    """After training on sufficient data, on_bar should emit or log."""
    algo = RegimeHMMAlgorithm(
        lookback=60, auto_train=False, score_threshold=0.3,
    )
    data = _trending_ohlcv(300)
    algo.train(data)

    ctx, _, logs = _make_ctx("regime_hmm", data)
    signal = algo.on_bar(ctx)

    # Should have logged regime info
    assert len(logs) >= 1
    assert logs[0]["message"] == "regime_hmm"
    assert "regime" in logs[0]
    assert "max_probability" in logs[0]

    # Signal emitted (with low threshold it should emit)
    if signal is not None:
        assert -1.0 <= signal.score <= 1.0
        assert 0.0 <= signal.confidence <= 1.0
        assert "regime" in signal.metadata
        assert "transition_matrix" in signal.metadata


def test_metadata_contains_transition_matrix():
    """Signal metadata should contain the transition matrix."""
    algo = RegimeHMMAlgorithm(
        n_regimes=3, lookback=30, auto_train=False, score_threshold=0.3,
    )
    data = _regime_ohlcv(300)
    algo.train(data)

    ctx, _, _ = _make_ctx("regime_hmm", data)
    signal = algo.on_bar(ctx)

    if signal is not None:
        tm = signal.metadata["transition_matrix"]
        assert len(tm) == 3
        assert len(tm[0]) == 3
        # Each row should sum to ~1.0
        for row in tm:
            assert abs(sum(row) - 1.0) < 0.01


# ---------------------------------------------------------------------------
# End-to-end with backtest engine
# ---------------------------------------------------------------------------


async def test_regime_hmm_end_to_end_backtest():
    """HMM should work with BacktestEngine on sufficient data."""
    algo = RegimeHMMAlgorithm(lookback=30, auto_train=True)
    engine = BacktestEngine()
    data = _trending_ohlcv(300)

    result = await engine.run(
        algorithm=algo,
        symbol=_symbol(),
        timeframe=Timeframe.D1,
        start=datetime(2024, 1, 1),
        end=datetime(2024, 10, 27),
        data=data,
        commission_bps=0,
    )

    assert len(result.equity_curve) == 300
    assert result.final_equity > 0


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_includes_regime_hmm():
    AlgorithmRegistry.clear()
    AlgorithmRegistry.auto_register()
    assert "regime_hmm" in AlgorithmRegistry.available()
    algo = AlgorithmRegistry.get("regime_hmm")
    assert algo.manifest.id == "regime_hmm"
