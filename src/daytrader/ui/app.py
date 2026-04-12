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
    from ..execution.registry import ExecutionRegistry

    AlgorithmRegistry.auto_register()
    AdapterRegistry.auto_register()
    ExecutionRegistry.auto_register()

    # Register saved DAG compositions so they appear in Strategy Lab.
    from pathlib import Path
    _dags_dir = Path(__file__).resolve().parents[3] / "data" / "dags"
    AlgorithmRegistry.load_saved_dags(_dags_dir)

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


async def _shutdown() -> None:
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
        dag_composer,
        home,
        journal,
        personas,
        plugins,
        research_lab,
        risk_center,
        strategy_lab,
    )
