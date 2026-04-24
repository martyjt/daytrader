"""Data layer: adapters for market data, feature store for Parquet caching."""

from .adapters.base import AdapterCapabilities, AdapterHealth, DataAdapter
from .adapters.registry import AdapterRegistry
from .features.store import FeatureStore

__all__ = [
    "AdapterCapabilities",
    "AdapterHealth",
    "AdapterRegistry",
    "DataAdapter",
    "FeatureStore",
]
