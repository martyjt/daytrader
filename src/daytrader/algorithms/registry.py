"""Algorithm registry — discover and retrieve algorithms by id."""

from __future__ import annotations

from .base import Algorithm


class AlgorithmRegistry:
    """Registry of available algorithms (built-in + plugins)."""

    _algorithms: dict[str, Algorithm] = {}

    @classmethod
    def register(cls, algo: Algorithm) -> None:
        cls._algorithms[algo.manifest.id] = algo

    @classmethod
    def get(cls, algo_id: str) -> Algorithm:
        if algo_id not in cls._algorithms:
            raise KeyError(
                f"Algorithm {algo_id!r} not registered. "
                f"Available: {sorted(cls._algorithms)}"
            )
        return cls._algorithms[algo_id]

    @classmethod
    def available(cls) -> list[str]:
        return sorted(cls._algorithms)

    @classmethod
    def all(cls) -> dict[str, Algorithm]:
        return dict(cls._algorithms)

    @classmethod
    def auto_register(cls) -> None:
        """Register all built-in algorithms."""
        from .builtin.buy_hold import BuyHoldAlgorithm

        if "buy_hold" not in cls._algorithms:
            cls.register(BuyHoldAlgorithm())

        if "xgboost_trend" not in cls._algorithms:
            try:
                from .builtin.xgboost_trend import XGBoostTrendAlgorithm

                cls.register(XGBoostTrendAlgorithm())
            except ImportError:
                pass  # xgboost not installed

    @classmethod
    def clear(cls) -> None:
        cls._algorithms.clear()
