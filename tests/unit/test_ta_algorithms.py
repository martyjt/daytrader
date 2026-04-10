"""Tests for all 9 technical analysis algorithms.

Each algorithm is tested for: manifest correctness, warmup_bars,
signal generation on synthetic data, score range, and end-to-end
backtest integration.
"""

from datetime import datetime, timedelta

import numpy as np
import polars as pl

from daytrader.algorithms.builtin.ema_crossover import EMACrossoverAlgorithm
from daytrader.algorithms.builtin.rsi_mean_reversion import RSIMeanReversionAlgorithm
from daytrader.algorithms.builtin.macd_signal import MACDSignalAlgorithm
from daytrader.algorithms.builtin.bollinger_bands import BollingerBandsAlgorithm
from daytrader.algorithms.builtin.stochastic_rsi import StochasticRSIAlgorithm
from daytrader.algorithms.builtin.vwap_bands import VWAPBandsAlgorithm
from daytrader.algorithms.builtin.supertrend import SupertrendAlgorithm
from daytrader.algorithms.builtin.adx_trend_filter import ADXTrendFilterAlgorithm
from daytrader.algorithms.builtin.donchian_breakout import DonchianBreakoutAlgorithm
from daytrader.backtest.engine import BacktestEngine
from daytrader.core.types.bars import Timeframe
from daytrader.core.types.symbols import AssetClass, Symbol

ALL_ALGOS = [
    EMACrossoverAlgorithm,
    RSIMeanReversionAlgorithm,
    MACDSignalAlgorithm,
    BollingerBandsAlgorithm,
    StochasticRSIAlgorithm,
    VWAPBandsAlgorithm,
    SupertrendAlgorithm,
    ADXTrendFilterAlgorithm,
    DonchianBreakoutAlgorithm,
]


def _symbol() -> Symbol:
    return Symbol("TEST", "USD", AssetClass.CRYPTO)


def _trending_up(n: int = 200) -> pl.DataFrame:
    """Steadily rising prices — should trigger trend-following algos."""
    np.random.seed(42)
    noise = np.random.randn(n) * 0.5
    prices = 100.0 + np.cumsum(np.ones(n) * 0.5 + noise)
    prices = np.maximum(prices, 10.0)
    return pl.DataFrame({
        "timestamp": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n)],
        "open": prices.tolist(),
        "high": (prices + np.abs(np.random.randn(n)) * 2).tolist(),
        "low": (prices - np.abs(np.random.randn(n)) * 2).tolist(),
        "close": prices.tolist(),
        "volume": (1000 + np.random.randn(n) * 100).tolist(),
    })


def _volatile(n: int = 200) -> pl.DataFrame:
    """Volatile prices oscillating — should trigger mean reversion algos."""
    np.random.seed(123)
    t = np.linspace(0, 8 * np.pi, n)
    prices = 100.0 + 20.0 * np.sin(t) + np.random.randn(n) * 2
    prices = np.maximum(prices, 10.0)
    return pl.DataFrame({
        "timestamp": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n)],
        "open": prices.tolist(),
        "high": (prices + np.abs(np.random.randn(n)) * 3).tolist(),
        "low": (prices - np.abs(np.random.randn(n)) * 3).tolist(),
        "close": prices.tolist(),
        "volume": (1000 + np.abs(np.random.randn(n)) * 500).tolist(),
    })


# ---------------------------------------------------------------------------
# Manifest and warmup tests (parametric across all algos)
# ---------------------------------------------------------------------------

def test_all_manifests_have_id():
    for AlgoClass in ALL_ALGOS:
        algo = AlgoClass()
        m = algo.manifest
        assert m.id, f"{AlgoClass.__name__} has empty id"
        assert m.name, f"{AlgoClass.__name__} has empty name"
        assert len(m.asset_classes) > 0


def test_all_manifests_have_params():
    for AlgoClass in ALL_ALGOS:
        algo = AlgoClass()
        m = algo.manifest
        assert isinstance(m.params, list)
        assert len(m.params) >= 2, f"{m.id} should have at least 2 params"
        for p in m.params:
            assert p.name, f"{m.id} has param with empty name"


def test_all_warmup_positive():
    for AlgoClass in ALL_ALGOS:
        algo = AlgoClass()
        assert algo.warmup_bars() > 0, f"{algo.manifest.id} warmup should be > 0"


def test_all_param_defaults():
    for AlgoClass in ALL_ALGOS:
        algo = AlgoClass()
        defaults = algo.manifest.param_defaults()
        assert isinstance(defaults, dict)
        assert len(defaults) == len(algo.manifest.params)


# ---------------------------------------------------------------------------
# End-to-end backtest tests
# ---------------------------------------------------------------------------

async def test_ema_crossover_e2e():
    result = await BacktestEngine().run(
        algorithm=EMACrossoverAlgorithm(),
        symbol=_symbol(), timeframe=Timeframe.D1,
        start=datetime(2024, 1, 1), end=datetime(2024, 7, 19),
        data=_trending_up(200), commission_bps=0,
    )
    assert len(result.equity_curve) == 200
    assert result.final_equity > 0


async def test_rsi_mean_reversion_e2e():
    result = await BacktestEngine().run(
        algorithm=RSIMeanReversionAlgorithm(),
        symbol=_symbol(), timeframe=Timeframe.D1,
        start=datetime(2024, 1, 1), end=datetime(2024, 7, 19),
        data=_volatile(200), commission_bps=0,
    )
    assert len(result.equity_curve) == 200
    assert result.final_equity > 0


async def test_macd_signal_e2e():
    result = await BacktestEngine().run(
        algorithm=MACDSignalAlgorithm(),
        symbol=_symbol(), timeframe=Timeframe.D1,
        start=datetime(2024, 1, 1), end=datetime(2024, 7, 19),
        data=_trending_up(200), commission_bps=0,
    )
    assert len(result.equity_curve) == 200
    assert result.final_equity > 0


async def test_bollinger_bands_e2e():
    result = await BacktestEngine().run(
        algorithm=BollingerBandsAlgorithm(),
        symbol=_symbol(), timeframe=Timeframe.D1,
        start=datetime(2024, 1, 1), end=datetime(2024, 7, 19),
        data=_volatile(200), commission_bps=0,
    )
    assert len(result.equity_curve) == 200
    assert result.final_equity > 0


async def test_stochastic_rsi_e2e():
    result = await BacktestEngine().run(
        algorithm=StochasticRSIAlgorithm(),
        symbol=_symbol(), timeframe=Timeframe.D1,
        start=datetime(2024, 1, 1), end=datetime(2024, 7, 19),
        data=_volatile(200), commission_bps=0,
    )
    assert len(result.equity_curve) == 200
    assert result.final_equity > 0


async def test_vwap_bands_e2e():
    result = await BacktestEngine().run(
        algorithm=VWAPBandsAlgorithm(),
        symbol=_symbol(), timeframe=Timeframe.D1,
        start=datetime(2024, 1, 1), end=datetime(2024, 7, 19),
        data=_volatile(200), commission_bps=0,
    )
    assert len(result.equity_curve) == 200
    assert result.final_equity > 0


async def test_supertrend_e2e():
    result = await BacktestEngine().run(
        algorithm=SupertrendAlgorithm(),
        symbol=_symbol(), timeframe=Timeframe.D1,
        start=datetime(2024, 1, 1), end=datetime(2024, 7, 19),
        data=_trending_up(200), commission_bps=0,
    )
    assert len(result.equity_curve) == 200
    assert result.final_equity > 0


async def test_adx_trend_filter_e2e():
    result = await BacktestEngine().run(
        algorithm=ADXTrendFilterAlgorithm(),
        symbol=_symbol(), timeframe=Timeframe.D1,
        start=datetime(2024, 1, 1), end=datetime(2024, 7, 19),
        data=_trending_up(200), commission_bps=0,
    )
    assert len(result.equity_curve) == 200
    assert result.final_equity > 0


async def test_donchian_breakout_e2e():
    result = await BacktestEngine().run(
        algorithm=DonchianBreakoutAlgorithm(),
        symbol=_symbol(), timeframe=Timeframe.D1,
        start=datetime(2024, 1, 1), end=datetime(2024, 7, 19),
        data=_trending_up(200), commission_bps=0,
    )
    assert len(result.equity_curve) == 200
    assert result.final_equity > 0


# ---------------------------------------------------------------------------
# Signal score range validation
# ---------------------------------------------------------------------------

async def test_all_signals_in_valid_range():
    """Every signal emitted by every algorithm should have score in [-1, 1]."""
    for AlgoClass in ALL_ALGOS:
        algo = AlgoClass()
        data = _volatile(200)
        result = await BacktestEngine().run(
            algorithm=algo,
            symbol=_symbol(), timeframe=Timeframe.D1,
            start=datetime(2024, 1, 1), end=datetime(2024, 7, 19),
            data=data, commission_bps=0,
        )
        for sig in result.signals:
            assert -1.0 <= sig.score <= 1.0, (
                f"{algo.manifest.id}: signal score {sig.score} out of range"
            )
            assert 0.0 <= sig.confidence <= 1.0, (
                f"{algo.manifest.id}: signal confidence {sig.confidence} out of range"
            )
