"""Async SQLAlchemy engine, session factory, and declarative base."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """SQLAlchemy declarative base for all Daytrader models."""


# Module-level state, initialized by init_db().
_engine = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


async def init_db(database_url: str, *, echo: bool = False) -> None:
    """Create the async engine and session factory.

    Call once at startup before any DB operations.
    """
    global _engine, _session_factory
    _engine = create_async_engine(database_url, echo=echo)
    _session_factory = async_sessionmaker(_engine, expire_on_commit=False)


async def close_db() -> None:
    """Dispose the engine. Call on shutdown."""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None


async def create_tables() -> None:
    """Create all tables via ``Base.metadata``.

    For dev / test only. Production uses ``alembic upgrade head``.
    """
    if _engine is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


@asynccontextmanager
async def get_session() -> AsyncIterator[AsyncSession]:
    """Yield an async session. Tenant scoping is the caller's responsibility."""
    if _session_factory is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    async with _session_factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
