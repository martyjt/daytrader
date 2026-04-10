"""Adapter registry — discover and retrieve data adapters by name."""

from __future__ import annotations

from .base import DataAdapter


class AdapterRegistry:
    """Simple registry of data adapters.

    Phase 2 will add auto-discovery via entry_points.
    """

    _adapters: dict[str, DataAdapter] = {}

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
        """Register all built-in adapters."""
        from .yfinance_adapter import YFinanceAdapter

        if "yfinance" not in cls._adapters:
            cls.register(YFinanceAdapter())

        if "alpaca" not in cls._adapters:
            try:
                from ...core.settings import get_settings

                settings = get_settings()
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

    @classmethod
    def clear(cls) -> None:
        """Remove all registered adapters (for testing)."""
        cls._adapters.clear()
