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

        # Phase 3 ML algorithm (optional dependency)
        if "regime_hmm" not in cls._algorithms:
            try:
                from .builtin.regime_hmm import RegimeHMMAlgorithm

                cls.register(RegimeHMMAlgorithm())
            except ImportError:
                pass  # hmmlearn not installed

        # Phase 3 technical algorithms
        _phase3 = [
            ("ichimoku_cloud", "IchimokuCloudAlgorithm", "ichimoku_cloud"),
            ("volume_profile", "VolumeProfileAlgorithm", "volume_profile"),
            ("williams_r", "WilliamsRAlgorithm", "williams_r"),
            ("cci_reversal", "CCIReversalAlgorithm", "cci_reversal"),
            ("keltner_channel", "KeltnerChannelAlgorithm", "keltner_channel"),
            ("obv_divergence", "OBVDivergenceAlgorithm", "obv_divergence"),
            ("rsi_divergence", "RSIDivergenceAlgorithm", "rsi_divergence"),
            ("mean_reversion_zscore", "MeanReversionZScoreAlgorithm", "mean_reversion_zscore"),
            ("triple_ema", "TripleEMACrossoverAlgorithm", "triple_ema"),
        ]
        for algo_id, class_name, module_name in _phase3:
            if algo_id not in cls._algorithms:
                try:
                    import importlib
                    mod = importlib.import_module(f".builtin.{module_name}", package=__package__)
                    algo_class = getattr(mod, class_name)
                    cls.register(algo_class())
                except Exception:
                    pass

        # Phase 5 PyTorch deep learning algorithms (optional dependency)
        _phase5 = [
            ("lstm_trend", "LSTMTrendAlgorithm", "lstm_trend"),
            ("transformer_trend", "TransformerTrendAlgorithm", "transformer_trend"),
            ("cnn_lstm_trend", "CNNLSTMTrendAlgorithm", "cnn_lstm_trend"),
        ]
        for algo_id, class_name, module_name in _phase5:
            if algo_id not in cls._algorithms:
                try:
                    import importlib
                    mod = importlib.import_module(f".builtin.{module_name}", package=__package__)
                    algo_class = getattr(mod, class_name)
                    cls.register(algo_class())
                except Exception:
                    pass  # torch not installed

    @classmethod
    def load_plugins(cls, plugin_dir: Path | str = "plugins") -> None:
        """Discover and register plugins from the given directory."""
        from .plugin_loader import PluginLoader

        loader = PluginLoader(plugin_dir)
        for result in loader.load_all():
            if result.success and result.algorithm:
                cls.register(result.algorithm)

    @classmethod
    def load_saved_dags(cls, dags_dir: Path | str) -> None:
        """Register all saved DAG compositions as composite algorithms.

        Reads every ``*.yaml`` file in ``dags_dir``, builds a
        :class:`CompositeAlgorithm` from it, and registers the result.
        Failed loads are skipped silently so one bad file can't break
        startup.
        """
        from .dag.composite import CompositeAlgorithm
        from .dag.serialization import load_dag

        dags_dir = Path(dags_dir)
        if not dags_dir.exists():
            return
        for yaml_file in sorted(dags_dir.glob("*.yaml")):
            try:
                dag = load_dag(yaml_file)
                composite = CompositeAlgorithm(dag)
                cls.register(composite)
            except Exception:
                # A single corrupt DAG shouldn't prevent startup.
                pass

    @classmethod
    def clear(cls) -> None:
        cls._algorithms.clear()
