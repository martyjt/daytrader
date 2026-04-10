"""Tests for the Alpaca equities adapter (mocked — no real API calls)."""

from daytrader.core.types.symbols import AssetClass, Symbol
from daytrader.data.adapters.alpaca_adapter import AlpacaAdapter


def test_name():
    adapter = AlpacaAdapter(api_key="test", api_secret="test")
    assert adapter.name == "alpaca"


def test_capabilities():
    adapter = AlpacaAdapter(api_key="test", api_secret="test")
    caps = adapter.capabilities()
    assert AssetClass.EQUITIES in caps.asset_classes
    assert caps.max_history_days == 3650
    assert caps.rate_limit_per_minute == 200


def test_ticker_conversion():
    aapl = Symbol("AAPL", "USD", AssetClass.EQUITIES)
    assert AlpacaAdapter.to_alpaca_ticker(aapl) == "AAPL"

    msft = Symbol("MSFT", "USD", AssetClass.EQUITIES)
    assert AlpacaAdapter.to_alpaca_ticker(msft) == "MSFT"


def test_ticker_conversion_base_only():
    """Alpaca ticker is always just the base symbol."""
    sym = Symbol("TSLA", "USD", AssetClass.EQUITIES, venue="alpaca")
    assert AlpacaAdapter.to_alpaca_ticker(sym) == "TSLA"
