"""Algorithm registry — discover and retrieve algorithms by id."""

from __future__ import annotations

from pathlib import Path

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

        # Phase 2 technical algorithms
        _phase2 = [
            ("ema_crossover", "EMACrossoverAlgorithm", "ema_crossover"),
            ("rsi_mean_reversion", "RSIMeanReversionAlgorithm", "rsi_mean_reversion"),
            ("macd_signal", "MACDSignalAlgorithm", "macd_signal"),
            ("bollinger_bands", "BollingerBandsAlgorithm", "bollinger_bands"),
            ("stochastic_rsi", "StochasticRSIAlgorithm", "stochastic_rsi"),
            ("vwap_bands", "VWAPBandsAlgorithm", "vwap_bands"),
            ("supertrend", "SupertrendAlgorithm", "supertrend"),
            ("adx_trend_filter", "ADXTrendFilterAlgorithm", "adx_trend_filter"),
            ("donchian_breakout", "DonchianBreakoutAlgorithm", "donchian_breakout"),
        ]
        for algo_id, class_name, module_name in _phase2:
            if algo_id not in cls._algorithms:
                try:
                    import importlib
                    mod = importlib.import_module(f".builtin.{module_name}", package=__package__)
                    algo_class = getattr(mod, class_name)
                    cls.register(algo_class())
                except Exception:
                    pass

    @classmethod
    def load_plugins(cls, plugin_dir: Path | str = "plugins") -> None:
        """Discover and register plugins from the given directory."""
        from .plugin_loader import PluginLoader

        loader = PluginLoader(plugin_dir)
        for result in loader.load_all():
            if result.success and result.algorithm:
                cls.register(result.algorithm)

    @classmethod
    def clear(cls) -> None:
        cls._algorithms.clear()
