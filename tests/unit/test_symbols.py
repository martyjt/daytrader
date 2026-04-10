from daytrader.core.types.symbols import AssetClass, Symbol


def test_parse_crypto_slash():
    s = Symbol.parse("BTC/USDT", AssetClass.CRYPTO)
    assert s.base == "BTC"
    assert s.quote == "USDT"
    assert s.canonical == "BTC/USDT"


def test_parse_crypto_dash():
    s = Symbol.parse("ETH-USDT", AssetClass.CRYPTO)
    assert s.base == "ETH"
    assert s.quote == "USDT"


def test_parse_crypto_smashed():
    s = Symbol.parse("BTCUSDT", AssetClass.CRYPTO)
    assert s.base == "BTC"
    assert s.quote == "USDT"


def test_parse_smashed_usdc():
    s = Symbol.parse("ETHUSDC", AssetClass.CRYPTO)
    assert s.base == "ETH"
    assert s.quote == "USDC"


def test_parse_equity_bare_ticker():
    s = Symbol.parse("AAPL", AssetClass.EQUITIES)
    assert s.base == "AAPL"
    assert s.quote == "USD"


def test_key_includes_venue():
    s = Symbol("BTC", "USDT", AssetClass.CRYPTO, venue="binance")
    assert s.key == "crypto:BTC/USDT@binance"


def test_key_without_venue():
    s = Symbol("AAPL", "USD", AssetClass.EQUITIES)
    assert s.key == "equities:AAPL/USD"
