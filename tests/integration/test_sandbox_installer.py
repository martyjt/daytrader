"""Installer tests — full upload flow against a real worker + SQLite."""

from __future__ import annotations

from contextlib import asynccontextmanager
from uuid import UUID

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker

from daytrader.algorithms.registry import AlgorithmRegistry
from daytrader.algorithms.sandbox import (
    PluginWorkerManager,
    SandboxedAlgorithm,
)
from daytrader.algorithms.sandbox.installer import (
    InstallError,
    install_plugin,
    list_for_tenant,
    record_plugin_error,
    restore_for_tenant,
    set_plugin_enabled,
    uninstall_plugin,
)
from daytrader.storage.models import TenantModel, UserModel

TENANT_A = UUID("00000000-0000-0000-0000-0000000000aa")
TENANT_B = UUID("00000000-0000-0000-0000-0000000000bb")
USER_A = UUID("00000000-0000-0000-0000-0000000000a1")


_GOOD_PLUGIN = b"""
from daytrader.algorithms.base import Algorithm, AlgorithmManifest


class HelloAlgo(Algorithm):
    @property
    def manifest(self):
        return AlgorithmManifest(id="hello_algo", name="Hello")

    def on_bar(self, ctx):
        return None
"""


_PLUGIN_WITH_WRONG_ID = b"""
from daytrader.algorithms.base import Algorithm, AlgorithmManifest


class Bad(Algorithm):
    @property
    def manifest(self):
        return AlgorithmManifest(id="some_other_id", name="Bad")

    def on_bar(self, ctx):
        return None
"""


@pytest_asyncio.fixture
async def db(engine, monkeypatch):
    factory = async_sessionmaker(engine, expire_on_commit=False)

    @asynccontextmanager
    async def _get_session():
        async with factory() as s:
            try:
                yield s
            except Exception:
                await s.rollback()
                raise

    monkeypatch.setattr("daytrader.storage.database.get_session", _get_session)
    monkeypatch.setattr(
        "daytrader.algorithms.sandbox.installer.get_session", _get_session
    )

    async with factory() as s:
        s.add(TenantModel(id=TENANT_A, name="a"))
        s.add(TenantModel(id=TENANT_B, name="b"))
        s.add(UserModel(
            id=USER_A, tenant_id=TENANT_A, email="a@x", role="owner",
        ))
        await s.commit()
    return factory


@pytest.fixture
def manager(tmp_path):
    yield PluginWorkerManager(base_dir=tmp_path / "uploads")


@pytest.fixture(autouse=True)
def _clean_registry():
    AlgorithmRegistry.clear()
    yield
    AlgorithmRegistry.clear()


@pytest_asyncio.fixture(autouse=True)
async def _shutdown(manager):
    yield
    await manager.shutdown_all()


# ---------------------------------------------------------------------------
# Install path
# ---------------------------------------------------------------------------


async def test_install_success_writes_file_and_db_row(db, manager):
    result = await install_plugin(
        manager=manager,
        tenant_id=TENANT_A,
        user_id=USER_A,
        filename="hello_algo.py",
        algorithm_id="hello_algo",
        payload=_GOOD_PLUGIN,
    )
    assert result.algorithm_id == "hello_algo"
    assert result.name == "Hello"
    assert len(result.sha256) == 64
    # File on disk
    assert (manager.tenant_dir(TENANT_A) / "hello_algo.py").is_file()
    # Registry overlay populated with a SandboxedAlgorithm
    algo = AlgorithmRegistry.get("hello_algo", tenant_id=TENANT_A)
    assert isinstance(algo, SandboxedAlgorithm)
    # DB row visible via list
    listed = await list_for_tenant(TENANT_A)
    assert len(listed) == 1 and listed[0].algorithm_id == "hello_algo"


async def test_install_rejects_invalid_filename(db, manager):
    with pytest.raises(InstallError):
        await install_plugin(
            manager=manager,
            tenant_id=TENANT_A,
            user_id=USER_A,
            filename="../escape.py",
            algorithm_id="hello_algo",
            payload=_GOOD_PLUGIN,
        )


async def test_install_rejects_syntax_error(db, manager):
    with pytest.raises(InstallError) as exc:
        await install_plugin(
            manager=manager,
            tenant_id=TENANT_A,
            user_id=USER_A,
            filename="broken.py",
            algorithm_id="broken",
            payload=b"def(",
        )
    assert "syntax" in str(exc.value).lower()


async def test_install_rejects_manifest_id_mismatch(db, manager):
    with pytest.raises(InstallError):
        await install_plugin(
            manager=manager,
            tenant_id=TENANT_A,
            user_id=USER_A,
            filename="bad.py",
            algorithm_id="declared_id",
            payload=_PLUGIN_WITH_WRONG_ID,
        )
    # Failed install must not leave the file on disk.
    assert not (manager.tenant_dir(TENANT_A) / "bad.py").exists()


async def test_install_rejects_built_in_id_collision(db, manager):
    AlgorithmRegistry._algorithms["hello_algo"] = object()  # dummy global
    try:
        with pytest.raises(InstallError) as exc:
            await install_plugin(
                manager=manager,
                tenant_id=TENANT_A,
                user_id=USER_A,
                filename="hello_algo.py",
                algorithm_id="hello_algo",
                payload=_GOOD_PLUGIN,
            )
        assert "reserved" in str(exc.value).lower()
    finally:
        AlgorithmRegistry._algorithms.pop("hello_algo", None)


async def test_reinstall_replaces_existing(db, manager):
    await install_plugin(
        manager=manager, tenant_id=TENANT_A, user_id=USER_A,
        filename="hello_algo.py", algorithm_id="hello_algo",
        payload=_GOOD_PLUGIN,
    )
    rows1 = await list_for_tenant(TENANT_A)
    # Re-upload — should update the same row, not create a second.
    await install_plugin(
        manager=manager, tenant_id=TENANT_A, user_id=USER_A,
        filename="hello_algo.py", algorithm_id="hello_algo",
        payload=_GOOD_PLUGIN + b"\n# changed\n",
    )
    rows2 = await list_for_tenant(TENANT_A)
    assert len(rows1) == len(rows2) == 1
    assert rows1[0].id == rows2[0].id
    assert rows1[0].sha256 != rows2[0].sha256


# ---------------------------------------------------------------------------
# Cross-tenant isolation
# ---------------------------------------------------------------------------


async def test_tenant_b_cannot_see_tenant_a_plugin(db, manager):
    await install_plugin(
        manager=manager, tenant_id=TENANT_A, user_id=USER_A,
        filename="hello_algo.py", algorithm_id="hello_algo",
        payload=_GOOD_PLUGIN,
    )
    # Tenant B's overlay does NOT have it.
    assert "hello_algo" not in AlgorithmRegistry.available(tenant_id=TENANT_B)
    # Tenant B's listing is empty.
    assert await list_for_tenant(TENANT_B) == []


# ---------------------------------------------------------------------------
# Uninstall
# ---------------------------------------------------------------------------


async def test_uninstall_removes_overlay_file_and_row(db, manager):
    await install_plugin(
        manager=manager, tenant_id=TENANT_A, user_id=USER_A,
        filename="hello_algo.py", algorithm_id="hello_algo",
        payload=_GOOD_PLUGIN,
    )
    file_path = manager.tenant_dir(TENANT_A) / "hello_algo.py"
    assert file_path.is_file()

    ok = await uninstall_plugin(
        manager=manager, tenant_id=TENANT_A, algorithm_id="hello_algo",
    )
    assert ok is True
    assert not file_path.exists()
    assert "hello_algo" not in AlgorithmRegistry.available(tenant_id=TENANT_A)
    assert await list_for_tenant(TENANT_A) == []


async def test_uninstall_unknown_returns_false(db, manager):
    ok = await uninstall_plugin(
        manager=manager, tenant_id=TENANT_A, algorithm_id="never_existed",
    )
    assert ok is False


# ---------------------------------------------------------------------------
# Startup hydration
# ---------------------------------------------------------------------------


async def test_disable_unregisters_overlay_keeps_file(db, manager):
    await install_plugin(
        manager=manager, tenant_id=TENANT_A, user_id=USER_A,
        filename="hello_algo.py", algorithm_id="hello_algo",
        payload=_GOOD_PLUGIN,
    )
    file_path = manager.tenant_dir(TENANT_A) / "hello_algo.py"

    changed = await set_plugin_enabled(
        manager=manager, tenant_id=TENANT_A,
        algorithm_id="hello_algo", enabled=False,
    )
    assert changed is True
    # Disabled plugins disappear from the overlay but stay on disk + DB.
    assert "hello_algo" not in AlgorithmRegistry.available(tenant_id=TENANT_A)
    assert file_path.is_file()
    rows = await list_for_tenant(TENANT_A)
    assert len(rows) == 1 and rows[0].is_enabled is False

    # Re-enable repopulates the overlay.
    changed_back = await set_plugin_enabled(
        manager=manager, tenant_id=TENANT_A,
        algorithm_id="hello_algo", enabled=True,
    )
    assert changed_back is True
    assert "hello_algo" in AlgorithmRegistry.available(tenant_id=TENANT_A)


async def test_set_enabled_idempotent(db, manager):
    await install_plugin(
        manager=manager, tenant_id=TENANT_A, user_id=USER_A,
        filename="hello_algo.py", algorithm_id="hello_algo",
        payload=_GOOD_PLUGIN,
    )
    # Already enabled — second call is a no-op (returns False).
    assert await set_plugin_enabled(
        manager=manager, tenant_id=TENANT_A,
        algorithm_id="hello_algo", enabled=True,
    ) is False


async def test_set_enabled_unknown_returns_false(db, manager):
    assert await set_plugin_enabled(
        manager=manager, tenant_id=TENANT_A,
        algorithm_id="never_existed", enabled=False,
    ) is False


async def test_record_plugin_error_persists_to_row(db, manager):
    await install_plugin(
        manager=manager, tenant_id=TENANT_A, user_id=USER_A,
        filename="hello_algo.py", algorithm_id="hello_algo",
        payload=_GOOD_PLUGIN,
    )
    await record_plugin_error(
        tenant_id=TENANT_A, algorithm_id="hello_algo",
        message="ValueError: something blew up",
    )
    rows = await list_for_tenant(TENANT_A)
    assert rows[0].last_error == "ValueError: something blew up"


async def test_record_plugin_error_truncates_long_messages(db, manager):
    await install_plugin(
        manager=manager, tenant_id=TENANT_A, user_id=USER_A,
        filename="hello_algo.py", algorithm_id="hello_algo",
        payload=_GOOD_PLUGIN,
    )
    huge = "x" * 5000
    await record_plugin_error(
        tenant_id=TENANT_A, algorithm_id="hello_algo", message=huge,
    )
    rows = await list_for_tenant(TENANT_A)
    assert len(rows[0].last_error) <= 2000


async def test_record_plugin_error_unknown_is_silent(db, manager):
    # No row exists → no-op, no exception.
    await record_plugin_error(
        tenant_id=TENANT_A, algorithm_id="no_such_thing",
        message="anything",
    )


async def test_restore_for_tenant_rebuilds_overlay_without_spawning(db, manager):
    await install_plugin(
        manager=manager, tenant_id=TENANT_A, user_id=USER_A,
        filename="hello_algo.py", algorithm_id="hello_algo",
        payload=_GOOD_PLUGIN,
    )
    # Simulate a fresh process boot: clear overlays and worker handles.
    AlgorithmRegistry.clear()
    await manager.shutdown_all()

    n = await restore_for_tenant(manager=manager, tenant_id=TENANT_A)
    assert n == 1
    assert "hello_algo" in AlgorithmRegistry.available(tenant_id=TENANT_A)
    # No worker should have been spawned just to populate the overlay —
    # the manager hasn't been asked for a handle yet.
    assert not manager.has_handle(TENANT_A)
