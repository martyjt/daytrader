"""Shared pytest fixtures for the daytrader test suite.

The ``engine`` fixture honours ``TEST_DATABASE_URL`` so CI can run the
full suite against Postgres alongside the default in-memory SQLite. With
a shared Postgres database, schema is dropped + recreated around each
test so tests stay isolated even when they reuse hard-coded UUIDs.
"""

from __future__ import annotations

import os

import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine

from daytrader.storage.database import Base


def _test_db_url() -> str:
    return os.environ.get("TEST_DATABASE_URL", "sqlite+aiosqlite://")


@pytest_asyncio.fixture
async def engine():
    e = create_async_engine(_test_db_url(), echo=False)
    async with e.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    try:
        yield e
    finally:
        async with e.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        await e.dispose()
