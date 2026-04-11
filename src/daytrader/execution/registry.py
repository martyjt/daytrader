"""Execution adapter registry — discover and retrieve execution adapters by name."""

from __future__ import annotations

from .base import ExecutionAdapter


class ExecutionRegistry:
    """Simple registry of execution adapters.

    Mirrors the data adapter registry in ``data/adapters/registry.py``.
    """

    _adapters: dict[str, ExecutionAdapter] = {}

    @classmethod
    def register(cls, adapter: ExecutionAdapter) -> None:
        cls._adapters[adapter.name] = adapter

    @classmethod
    def get(cls, name: str) -> ExecutionAdapter:
        if name not in cls._adapters:
            raise KeyError(
                f"Execution adapter {name!r} not registered. "
                f"Available: {sorted(cls._adapters)}"
            )
        return cls._adapters[name]

    @classmethod
    def available(cls) -> list[str]:
        return sorted(cls._adapters)

    @classmethod
    def auto_register(cls) -> None:
        """Register built-in execution adapters."""
        from .paper import PaperExecutor

        if "paper" not in cls._adapters:
            cls.register(PaperExecutor())

        # Binance — register if API key is configured
        if "binance" not in cls._adapters:
            try:
                from ..core.settings import get_settings

                settings = get_settings()
                api_key = settings.binance_api_key.get_secret_value()
                if api_key:
                    from .binance import BinanceExecutor

                    cls.register(
                        BinanceExecutor(
                            api_key=api_key,
                            api_secret=settings.binance_api_secret.get_secret_value(),
                            testnet=settings.binance_testnet,
                        )
                    )
            except Exception:
                pass  # Binance not configured or import failed

        # Alpaca — register if API key is configured
        if "alpaca" not in cls._adapters:
            try:
                from ..core.settings import get_settings

                settings = get_settings()
                api_key = settings.alpaca_api_key.get_secret_value()
                if api_key:
                    from .alpaca import AlpacaExecutor

                    cls.register(
                        AlpacaExecutor(
                            api_key=api_key,
                            api_secret=settings.alpaca_api_secret.get_secret_value(),
                            paper=settings.alpaca_paper,
                        )
                    )
            except Exception:
                pass  # Alpaca not configured or import failed

    @classmethod
    def clear(cls) -> None:
        """Remove all registered adapters (for testing)."""
        cls._adapters.clear()
