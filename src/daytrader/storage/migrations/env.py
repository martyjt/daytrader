"""Alembic environment configuration for async SQLAlchemy."""

import asyncio
import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import create_async_engine

from daytrader.storage.database import Base
from daytrader.storage import models as _models  # noqa: F401 — side-effect import

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Production deploys set DATABASE_URL via env so we don't ship secrets in
# alembic.ini. The .ini value is the dev fallback.
_env_url = os.environ.get("DATABASE_URL")
if _env_url:
    config.set_main_option("sqlalchemy.url", _env_url)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection) -> None:  # type: ignore[no-untyped-def]
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    connectable = create_async_engine(
        config.get_main_option("sqlalchemy.url"),
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
