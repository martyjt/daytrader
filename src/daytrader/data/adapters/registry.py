"""Adapter registry — discover and retrieve data adapters by name."""

from __future__ import annotations

from typing import ClassVar

from .base import DataAdapter


class AdapterRegistry:
    """Simple registry of data adapters.

    Phase 2 will add auto-discovery via entry_points.
    """

    _adapters: ClassVar[dict[str, DataAdapter]] = {}

    @classmethod
    def register(cls, adapter: DataAdapter) -> None:
        cls._adapters[adapter.name] = adapter

    @classmethod
    def get(cls, name: str) -> DataAdapter:
        if name not in cls._adapters:
            raise KeyError(
                f"Adapter {name!r} not registered. "
                f"Available: {sorted(cls._adapters)}"
            )
        return cls._adapters[name]

    @classmethod
    def available(cls) -> list[str]:
        return sorted(cls._adapters)

    @classmethod
    def auto_register(cls) -> None:
        """Register all built-in adapters.

        Unkeyed adapters (yfinance, binance_public) always register.
        Keyed adapters register only when their API key is set in env.
        """
        from .yfinance_adapter import YFinanceAdapter

        if "yfinance" not in cls._adapters:
            cls.register(YFinanceAdapter())

        if "binance_public" not in cls._adapters:
            from .binance_public_adapter import BinancePublicAdapter

            cls.register(BinancePublicAdapter())

        # Load settings once for conditional registrations.
        try:
            from ...core.settings import get_settings

            settings = get_settings()
        except Exception:
            settings = None

        if settings is not None:
            if "alpaca" not in cls._adapters:
                try:
                    api_key = settings.alpaca_api_key.get_secret_value()
                    if api_key:
                        from .alpaca_adapter import AlpacaAdapter

                        cls.register(
                            AlpacaAdapter(
                                api_key=api_key,
                                api_secret=settings.alpaca_api_secret.get_secret_value(),
                                paper=settings.alpaca_paper,
                            )
                        )
                except Exception:
                    pass  # Alpaca not configured

            if "alpha_vantage" not in cls._adapters:
                try:
                    av_key = settings.alpha_vantage_api_key.get_secret_value()
                    if av_key:
                        from .alpha_vantage_adapter import AlphaVantageAdapter

                        cls.register(AlphaVantageAdapter(api_key=av_key))
                except Exception:
                    pass

            if "twelve_data" not in cls._adapters:
                try:
                    td_key = settings.twelve_data_api_key.get_secret_value()
                    if td_key:
                        from .twelve_data_adapter import TwelveDataAdapter

                        cls.register(TwelveDataAdapter(api_key=td_key))
                except Exception:
                    pass

    @classmethod
    def clear(cls) -> None:
        """Remove all registered adapters (for testing)."""
        cls._adapters.clear()
