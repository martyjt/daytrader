"""Entry point: ``python -m daytrader``.

Launches the NiceGUI application server. For local development without
Docker, set ``DATABASE_URL=sqlite+aiosqlite:///daytrader.db`` in a
``.env`` file at the project root.
"""

from __future__ import annotations


def main() -> None:
    from nicegui import ui

    from .core.settings import get_settings
    from .ui.app import create_app

    settings = get_settings()
    create_app()

    ui.run(
        host=settings.app_host,
        port=settings.app_port,
        title="Daytrader",
        dark=True,
        reload=False,
        storage_secret=settings.app_secret_key.get_secret_value(),
    )


if __name__ == "__main__":
    main()
