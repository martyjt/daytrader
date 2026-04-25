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

    # Initialise the journal writer, kill switch, and global risk monitor.
    from ..journal.writer import JournalWriter
    from ..execution.kill_switch import init_kill_switch
    from ..risk.global_risk import GlobalRiskConfig, GlobalRiskMonitor
    from ..execution.loop import TradingLoop

    journal = JournalWriter()
    kill_switch = init_kill_switch(journal=journal)

    global_risk = GlobalRiskMonitor(GlobalRiskConfig.from_yaml())

    # Start the live trading loop.
    _loop = TradingLoop(
        journal=journal,
        kill_switch=kill_switch,
        global_risk=global_risk,
        tenant_id=settings.default_tenant_id,
    )
    await _loop.start()
    app.state.trading_loop = _loop

    # Optional: start the Exploration Agent scheduler (off by default).
    from ..research.scheduler import ExplorationScheduler

    symbols = [
        s.strip() for s in settings.exploration_schedule_symbols.split(",")
        if s.strip()
    ]
    exploration = ExplorationScheduler(
        tenant_id=settings.default_tenant_id,
        interval_hours=settings.exploration_schedule_hours,
        symbols=symbols,
        timeframe=settings.exploration_schedule_timeframe,
        lookback_days=settings.exploration_schedule_lookback_days,
    )
    await exploration.start()
    app.state.exploration_scheduler = exploration

    # Start the regime watcher (keeps Regime Badge fresh + fires alerts).
    from .regime_watcher import RegimeWatcher

    watcher = RegimeWatcher(
        interval_minutes=settings.regime_refresh_minutes,
        pulse_symbol=settings.regime_pulse_symbol,
        pulse_timeframe=settings.regime_pulse_timeframe,
    )
    await watcher.start()
    app.state.regime_watcher = watcher

    # Optional: scheduled shadow tournaments (off by default — heavy).
    from ..research.shadow_scheduler import ShadowScheduler

    shadow_candidates = [
        c.strip() for c in settings.shadow_schedule_candidates.split(",")
        if c.strip()
    ]
    shadow = ShadowScheduler(
        tenant_id=settings.default_tenant_id,
        interval_hours=settings.shadow_schedule_hours,
        primary_algo_id=settings.shadow_schedule_primary,
        candidate_algo_ids=shadow_candidates,
        symbol=settings.shadow_schedule_symbol,
        timeframe=settings.shadow_schedule_timeframe,
        lookback_days=settings.shadow_schedule_lookback_days,
    )
    await shadow.start()
    app.state.shadow_scheduler = shadow


async def _shutdown() -> None:
    # Stop the shadow scheduler.
    shadow = getattr(app.state, "shadow_scheduler", None)
    if shadow:
        try:
            await shadow.stop()
        except Exception:
            pass

    # Stop the regime watcher.
    watcher = getattr(app.state, "regime_watcher", None)
    if watcher:
        try:
            await watcher.stop()
        except Exception:
            pass

    # Stop the exploration scheduler.
    scheduler = getattr(app.state, "exploration_scheduler", None)
    if scheduler:
        try:
            await scheduler.stop()
        except Exception:
            pass

    # Stop the trading loop.
    loop = getattr(app.state, "trading_loop", None)
    if loop:
        await loop.stop()

    # Close any live broker connections.
    from ..execution.registry import ExecutionRegistry

    for name in ExecutionRegistry.available():
        adapter = ExecutionRegistry.get(name)
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
        admin_users,
        auth,
        bandit_builder,
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
