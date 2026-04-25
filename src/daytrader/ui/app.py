"""NiceGUI app factory — lifecycle hooks and page registration."""

from __future__ import annotations

from nicegui import app

from ..core.settings import get_settings


async def _startup() -> None:
    settings = get_settings()

    from ..storage.database import create_tables, init_db
    from ..storage.database import get_session
    from ..storage.seed import seed_default_tenant

    await init_db(settings.database_url)

    # Dev mode: auto-create tables (production uses alembic upgrade head).
    if settings.app_env == "dev":
        await create_tables()

    async with get_session() as session:
        await seed_default_tenant(session, settings)

    # Phase 3 — fail loudly if encrypted credentials exist but no key is set.
    from ..core.crypto import assert_encryption_key_for_existing_secrets

    await assert_encryption_key_for_existing_secrets()

    # Register built-in algorithms, data adapters, and execution adapters.
    from ..algorithms.registry import AlgorithmRegistry
    from ..data.adapters.registry import AdapterRegistry
    from ..data.macro.base import MacroAdapterRegistry
    from ..data.sentiment.base import SentimentAdapterRegistry
    from ..execution.registry import ExecutionRegistry

    AlgorithmRegistry.auto_register()
    AdapterRegistry.auto_register()
    MacroAdapterRegistry.auto_register()
    SentimentAdapterRegistry.auto_register()
    ExecutionRegistry.auto_register()

    # Register saved DAG compositions so they appear in Strategy Lab.
    from pathlib import Path
    _dags_dir = Path(__file__).resolve().parents[3] / "data" / "dags"
    AlgorithmRegistry.load_saved_dags(_dags_dir)

    # Register saved bandit allocator configurations.
    from ..algorithms.rl.bandit_serialization import load_all_bandits

    _bandits_dir = Path(__file__).resolve().parents[3] / "data" / "bandits"
    load_all_bandits(_bandits_dir)

    # Phase 6 — start the plugin sandbox manager and rebuild every tenant's
    # overlay from DB rows. Workers stay un-spawned until first use; the
    # adapter caches each plugin's manifest on disk so we don't pay process
    # startup just to populate the algo picker.
    from ..algorithms.sandbox import (
        PluginWorkerManager,
        default_uploads_dir,
        set_active_manager,
    )
    from ..algorithms.sandbox.installer import restore_all_at_startup

    plugin_manager = PluginWorkerManager(base_dir=default_uploads_dir())
    set_active_manager(plugin_manager)
    await restore_all_at_startup(plugin_manager)
    app.state.plugin_manager = plugin_manager

    # Phase 9 — install the shared market-data bus so concurrent
    # personas trading the same symbol/timeframe collapse to one fetch
    # per (~30s) cache window.
    from ..data.marketdata_bus import MarketDataBus, set_active_bus

    market_data_bus = MarketDataBus()
    set_active_bus(market_data_bus)
    app.state.market_data_bus = market_data_bus

    # Initialise the journal writer, kill switch, and global risk monitor.
    from ..journal.writer import JournalWriter
    from ..execution.kill_switch import init_kill_switch
    from ..risk.global_risk import GlobalRiskConfig, GlobalRiskMonitor

    journal = JournalWriter()
    kill_switch = init_kill_switch(journal=journal, plugin_manager=plugin_manager)
    global_risk = GlobalRiskMonitor(GlobalRiskConfig.from_yaml())

    # Per-tenant supervisors — one TradingLoop / ExplorationScheduler /
    # ShadowScheduler per active tenant. Each polls every 60s and spins
    # workers up or down as personas / tenants change.
    from ..core.supervisor import SupervisorManager
    from ..execution.supervisor import TradingLoopSupervisor
    from ..research.supervisor import ExplorationSupervisor, ShadowSupervisor

    manager = SupervisorManager([
        TradingLoopSupervisor(
            journal=journal,
            kill_switch=kill_switch,
            global_risk=global_risk,
        ),
        ExplorationSupervisor(),
        ShadowSupervisor(),
    ])
    await manager.start_all()
    app.state.supervisor_manager = manager

    # Start the regime watcher (keeps Regime Badge fresh + fires alerts).
    # Tenant-agnostic: the HMM is fit on a single pulse symbol shared by all
    # tenants, so a per-tenant supervisor would just duplicate work.
    from .regime_watcher import RegimeWatcher

    watcher = RegimeWatcher(
        interval_minutes=settings.regime_refresh_minutes,
        pulse_symbol=settings.regime_pulse_symbol,
        pulse_timeframe=settings.regime_pulse_timeframe,
    )
    await watcher.start()
    app.state.regime_watcher = watcher


async def _shutdown() -> None:
    # Stop the regime watcher.
    watcher = getattr(app.state, "regime_watcher", None)
    if watcher:
        try:
            await watcher.stop()
        except Exception:
            pass

    # Stop all per-tenant supervisors (and their workers).
    manager = getattr(app.state, "supervisor_manager", None)
    if manager:
        try:
            await manager.stop_all()
        except Exception:
            pass

    # Shut down all plugin worker subprocesses.
    plugin_manager = getattr(app.state, "plugin_manager", None)
    if plugin_manager:
        try:
            await plugin_manager.shutdown_all()
        except Exception:
            pass
        from ..algorithms.sandbox import set_active_manager
        set_active_manager(None)

    # Clear the market-data bus singleton.
    from ..data.marketdata_bus import set_active_bus

    set_active_bus(None)

    # Close any live broker connections (global adapters + per-tenant cache).
    from ..execution.registry import ExecutionRegistry

    for name in ExecutionRegistry.available():
        adapter = ExecutionRegistry.get(name)
        if hasattr(adapter, "close"):
            try:
                await adapter.close()
            except Exception:
                pass
    for adapter in ExecutionRegistry.cached_tenant_adapters().values():
        if hasattr(adapter, "close"):
            try:
                await adapter.close()
            except Exception:
                pass

    from ..storage.database import close_db

    await close_db()


def create_app() -> None:
    """Wire lifecycle hooks and import pages to register their routes."""
    app.on_startup(_startup)
    app.on_shutdown(_shutdown)

    # Importing page modules triggers their @ui.page() decorators.
    from .pages import (  # noqa: F401
        admin_audit,
        admin_tenants,
        admin_users,
        auth,
        bandit_builder,
        broker_credentials,
        cache,
        charts,
        dag_composer,
        home,
        journal,
        persona_detail,
        personas,
        plugins,
        research_lab,
        risk_center,
        signal_feed,
        strategies,
        strategy_lab,
        universes,
    )
