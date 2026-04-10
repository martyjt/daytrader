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

    # Register built-in algorithms and data adapters.
    from ..algorithms.registry import AlgorithmRegistry
    from ..data.adapters.registry import AdapterRegistry

    AlgorithmRegistry.auto_register()
    AdapterRegistry.auto_register()


async def _shutdown() -> None:
    from ..storage.database import close_db

    await close_db()


def create_app() -> None:
    """Wire lifecycle hooks and import pages to register their routes."""
    app.on_startup(_startup)
    app.on_shutdown(_shutdown)

    # Importing page modules triggers their @ui.page() decorators.
    from .pages import (  # noqa: F401
        home,
        personas,
        plugins,
        risk_center,
        strategy_lab,
    )
