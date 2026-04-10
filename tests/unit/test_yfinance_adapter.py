"""Tests for the yfinance data adapter.

Uses mocked yfinance to avoid network calls. The adapter's ticker
conversion and schema normalization are the main things under test.
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import polars as pl
import pytest

from daytrader.core.types.bars import Timeframe
from daytrader.core.types.symbols import AssetClass, Symbol
from daytrader.data.adapters.yfinance_adapter import YFinanceAdapter


def _mock_history() -> pd.DataFrame:
    """Mimic what ``yfinance.Ticker.history()`` returns."""
    dates = pd.date_range("2024-01-02", periods=5, freq="D", name="Date")
    return pd.DataFrame(
        {
            "Open": [42000.0, 42500.0, 43000.0, 42800.0, 43200.0],
            "High": [42800.0, 43200.0, 43500.0, 43300.0, 43800.0],
            "Low": [41500.0, 42000.0, 42500.0, 42200.0, 42700.0],
            "Close": [42500.0, 43000.0, 42800.0, 43200.0, 43500.0],
            "Volume": [1000, 1200, 900, 1100, 1300],
            "Dividends": [0, 0, 0, 0, 0],
            "Stock Splits": [0, 0, 0, 0, 0],
        },
        index=dates,
    )


# ---- ticker conversion (no network, no mocking) --------------------------


def test_ticker_crypto():
    sym = Symbol("BTC", "USD", AssetClass.CRYPTO)
    assert YFinanceAdapter.to_yfinance_ticker(sym) == "BTC-USD"


def test_ticker_equity():
    sym = Symbol("AAPL", "USD", AssetClass.EQUITIES)
    assert YFinanceAdapter.to_yfinance_ticker(sym) == "AAPL"


def test_ticker_forex():
    sym = Symbol("EUR", "USD", AssetClass.FOREX)
    assert YFinanceAdapter.to_yfinance_ticker(sym) == "EURUSD=X"


def test_ticker_commodity():
    sym = Symbol("GC=F", "USD", AssetClass.COMMODITIES)
    assert YFinanceAdapter.to_yfinance_ticker(sym) == "GC=F"


# ---- capabilities --------------------------------------------------------


def test_capabilities():
    adapter = YFinanceAdapter()
    caps = adapter.capabilities()
    assert AssetClass.CRYPTO in caps.asset_classes
    assert AssetClass.EQUITIES in caps.asset_classes
    assert Timeframe.D1 in caps.timeframes
    assert Timeframe.H4 not in caps.timeframes  # yfinance doesn't support 4h


# ---- fetch_ohlcv with mocked yfinance ------------------------------------


async def test_fetch_ohlcv_returns_polars_df():
    """Full flow: mock yfinance, call fetch_ohlcv, verify Polars output."""
    import sys
    from unittest.mock import MagicMock

    mock_yf = MagicMock()
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = _mock_history()
    mock_yf.Ticker.return_value = mock_ticker

    # Temporarily inject mock into sys.modules
    original = sys.modules.get("yfinance")
    sys.modules["yfinance"] = mock_yf
    try:
        adapter = YFinanceAdapter()
        sym = Symbol("BTC", "USD", AssetClass.CRYPTO)

        df = await adapter.fetch_ohlcv(
            sym,
            Timeframe.D1,
            datetime(2024, 1, 1),
            datetime(2024, 1, 10),
        )

        assert isinstance(df, pl.DataFrame)
        assert df.columns == [
            "timestamp", "open", "high", "low", "close", "volume",
        ]
        assert len(df) == 5
        mock_yf.Ticker.assert_called_once_with("BTC-USD")
    finally:
        if original is not None:
            sys.modules["yfinance"] = original
        else:
            sys.modules.pop("yfinance", None)


async def test_fetch_ohlcv_empty():
    """Empty result from yfinance returns an empty Polars DataFrame."""
    import sys
    from unittest.mock import MagicMock

    mock_yf = MagicMock()
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = pd.DataFrame()
    mock_yf.Ticker.return_value = mock_ticker

    original = sys.modules.get("yfinance")
    sys.modules["yfinance"] = mock_yf
    try:
        adapter = YFinanceAdapter()
        sym = Symbol("AAPL", "USD", AssetClass.EQUITIES)
        df = await adapter.fetch_ohlcv(
            sym, Timeframe.D1, datetime(2024, 1, 1), datetime(2024, 1, 10),
        )
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 0
    finally:
        if original is not None:
            sys.modules["yfinance"] = original
        else:
            sys.modules.pop("yfinance", None)


async def test_unsupported_timeframe():
    adapter = YFinanceAdapter()
    sym = Symbol("BTC", "USD", AssetClass.CRYPTO)
    with pytest.raises(ValueError, match="does not support"):
        await adapter.fetch_ohlcv(
            sym, Timeframe.H4, datetime(2024, 1, 1), datetime(2024, 1, 10),
        )
