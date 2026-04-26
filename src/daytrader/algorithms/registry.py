"""Algorithm registry — discover and retrieve algorithms by id.

Two layers:

* The **global** layer holds built-in algorithms; every tenant sees the
  same instances. ``register`` / ``get`` without a ``tenant_id`` operate
  here.

* A **per-tenant overlay** holds sandboxed plugins uploaded by a
  particular tenant's owner. ``get(algo_id, tenant_id=…)`` checks the
  overlay first, falls back to the global layer. Tenant A's plugins
  are invisible to tenant B by construction — there is no path that
  joins the two without an explicit ``tenant_id`` argument.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar
from uuid import UUID

from .base import Algorithm


class AlgorithmRegistry:
    """Registry of available algorithms (built-in + tenant plugins)."""

    _algorithms: ClassVar[dict[str, Algorithm]] = {}
    _per_tenant: ClassVar[dict[UUID, dict[str, Algorithm]]] = {}

    # ---- global layer (built-ins) ---------------------------------------

    @classmethod
    def register(cls, algo: Algorithm) -> None:
        cls._algorithms[algo.manifest.id] = algo

    @classmethod
    def get(cls, algo_id: str, tenant_id: UUID | None = None) -> Algorithm:
        if tenant_id is not None:
            overlay = cls._per_tenant.get(tenant_id)
            if overlay is not None and algo_id in overlay:
                return overlay[algo_id]
        if algo_id not in cls._algorithms:
            available = sorted(cls._algorithms)
            if tenant_id is not None and cls._per_tenant.get(tenant_id):
                available = sorted(set(available) | set(cls._per_tenant[tenant_id]))
            raise KeyError(
                f"Algorithm {algo_id!r} not registered. Available: {available}"
            )
        return cls._algorithms[algo_id]

    @classmethod
    def available(cls, tenant_id: UUID | None = None) -> list[str]:
        ids = set(cls._algorithms)
        if tenant_id is not None:
            ids |= set(cls._per_tenant.get(tenant_id, {}))
        return sorted(ids)

    @classmethod
    def all(cls, tenant_id: UUID | None = None) -> dict[str, Algorithm]:
        merged: dict[str, Algorithm] = dict(cls._algorithms)
        if tenant_id is not None:
            # Tenant overlay wins on collision — but we don't allow tenants
            # to shadow built-in ids in the installer, so collisions should
            # not happen in practice.
            merged.update(cls._per_tenant.get(tenant_id, {}))
        return merged

    # ---- per-tenant overlay --------------------------------------------

    @classmethod
    def register_for_tenant(cls, tenant_id: UUID, algo: Algorithm) -> None:
        """Add an algorithm to a tenant's overlay. Replaces any existing entry."""
        cls._per_tenant.setdefault(tenant_id, {})[algo.manifest.id] = algo

    @classmethod
    def unregister_for_tenant(cls, tenant_id: UUID, algo_id: str) -> bool:
        """Remove an entry from a tenant's overlay. Returns True if removed."""
        overlay = cls._per_tenant.get(tenant_id)
        if not overlay or algo_id not in overlay:
            return False
        del overlay[algo_id]
        if not overlay:
            cls._per_tenant.pop(tenant_id, None)
        return True

    @classmethod
    def clear_tenant(cls, tenant_id: UUID) -> None:
        """Drop all overlay entries for a tenant (used on tenant teardown)."""
        cls._per_tenant.pop(tenant_id, None)

    @classmethod
    def tenant_overlay(cls, tenant_id: UUID) -> dict[str, Algorithm]:
        """Return a copy of just the tenant's overlay, no built-ins."""
        return dict(cls._per_tenant.get(tenant_id, {}))

    @classmethod
    def auto_register(cls) -> None:
        """Register all built-in algorithms."""
        from .builtin.buy_hold import BuyHoldAlgorithm
        from .builtin.feature_threshold import FeatureThresholdAlgorithm

        if "buy_hold" not in cls._algorithms:
            cls.register(BuyHoldAlgorithm())

        if "feature_threshold" not in cls._algorithms:
            cls.register(FeatureThresholdAlgorithm())

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

        # Phase 7 reinforcement-learning agents (optional 'rl' extra)
        if "ppo_agent" not in cls._algorithms:
            try:
                from .rl.ppo_agent import PPOAgent

                cls.register(PPOAgent())
            except Exception:
                pass  # stable-baselines3 / gymnasium not installed

        if "sac_agent" not in cls._algorithms:
            try:
                from .rl.sac_agent import SACAgent

                cls.register(SACAgent())
            except Exception:
                pass

        # BanditAllocator is unconditionally available — it has no external
        # dependency and is only useful when children are injected (e.g.
        # through a DAG). Register a no-children template.
        if "bandit_allocator" not in cls._algorithms:
            try:
                from .rl.bandit_allocator import BanditAllocator

                cls.register(BanditAllocator())
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
        cls._per_tenant.clear()
