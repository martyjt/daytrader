"""Tests for the Polars/Parquet feature store."""

from datetime import datetime

import polars as pl

from daytrader.data.features.store import FeatureStore


def _sample_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
            "volume": [1000.0, 1100.0],
        }
    )


def test_put_and_get(tmp_path):
    store = FeatureStore(cache_dir=tmp_path)
    df = _sample_df()
    start = datetime(2024, 1, 1)
    end = datetime(2024, 1, 2)

    store.put("crypto:BTC/USDT", "1d", start, end, df)

    result = store.get("crypto:BTC/USDT", "1d", start, end)
    assert result is not None
    assert result.shape == df.shape
    assert result.columns == df.columns


def test_get_missing_returns_none(tmp_path):
    store = FeatureStore(cache_dir=tmp_path)
    result = store.get("crypto:BTC/USDT", "1d", datetime(2024, 1, 1), datetime(2024, 1, 2))
    assert result is None


def test_has(tmp_path):
    store = FeatureStore(cache_dir=tmp_path)
    start = datetime(2024, 1, 1)
    end = datetime(2024, 1, 2)

    assert not store.has("crypto:BTC/USDT", "1d", start, end)
    store.put("crypto:BTC/USDT", "1d", start, end, _sample_df())
    assert store.has("crypto:BTC/USDT", "1d", start, end)


def test_invalidate_specific(tmp_path):
    store = FeatureStore(cache_dir=tmp_path)
    start = datetime(2024, 1, 1)
    end = datetime(2024, 1, 2)

    store.put("crypto:BTC/USDT", "1d", start, end, _sample_df())
    store.put("crypto:ETH/USDT", "1d", start, end, _sample_df())

    count = store.invalidate("crypto:BTC/USDT")
    assert count == 1
    assert store.get("crypto:BTC/USDT", "1d", start, end) is None
    assert store.get("crypto:ETH/USDT", "1d", start, end) is not None


def test_invalidate_all(tmp_path):
    store = FeatureStore(cache_dir=tmp_path)
    start = datetime(2024, 1, 1)
    end = datetime(2024, 1, 2)

    store.put("crypto:BTC/USDT", "1d", start, end, _sample_df())
    store.put("crypto:ETH/USDT", "1d", start, end, _sample_df())

    count = store.invalidate()
    assert count == 2
    assert store.list_cached() == []


def test_list_cached(tmp_path):
    store = FeatureStore(cache_dir=tmp_path)
    assert store.list_cached() == []

    store.put("crypto:BTC/USDT", "1d", datetime(2024, 1, 1), datetime(2024, 1, 2), _sample_df())
    cached = store.list_cached()
    assert len(cached) == 1
    assert cached[0].endswith(".parquet")
