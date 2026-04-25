"""Plugin install / uninstall orchestration.

The installer is the only module that writes uploaded plugin files to
disk. The flow:

1. Validate filename and size.
2. Compute sha256.
3. Pre-flight: ``compile()`` the source in the parent so syntax errors
   surface before we ever spawn a worker. (Compilation does not execute
   the module — it just builds the AST. Safe.)
4. Write to ``plugins/uploads/<tenant_id>/<filename>``.
5. Spawn the tenant's worker (if not running), call ``load_plugin``.
6. Build a ``SandboxedAlgorithm`` adapter, register it in the per-tenant
   overlay of ``AlgorithmRegistry``.
7. Upsert the ``TenantPluginModel`` row with the cached manifest so a
   future restart can rebuild the overlay without spawning workers.

Re-uploads of the same algorithm id replace the previous version
in-place — same file path, same DB row, manager is told to ``unload``
the old version first, then ``load`` the new one.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import select

from ...core import audit
from ...storage.database import get_session
from ...storage.models import TenantPluginModel
from ..registry import AlgorithmRegistry
from .adapter import SandboxedAlgorithm, _manifest_from_payload
from .manager import PluginRuntimeError, PluginWorkerManager

logger = logging.getLogger(__name__)


# Filenames are constrained tightly to defang path traversal and to keep
# downstream tools (importlib, the file system, the UI) well-behaved.
VALID_FILENAME = re.compile(r"^[A-Za-z0-9_\-]{1,80}\.py$")
MAX_PLUGIN_BYTES = 1 * 1024 * 1024  # 1 MB — algorithms shouldn't be bigger


class InstallError(Exception):
    """Raised for install-time failures the UI should surface to the user."""


@dataclass(frozen=True)
class InstallResult:
    plugin_id: UUID
    algorithm_id: str
    name: str
    filename: str
    sha256: str
    warmup_bars: int


def _validate_filename(filename: str) -> str:
    name = (filename or "").strip()
    if not VALID_FILENAME.match(name):
        raise InstallError(
            f"Invalid filename {filename!r}. Must match {VALID_FILENAME.pattern} "
            "(letters, digits, underscore, dash; ending in .py)."
        )
    return name


def _validate_payload(payload: bytes) -> None:
    if not payload:
        raise InstallError("Plugin file is empty")
    if len(payload) > MAX_PLUGIN_BYTES:
        raise InstallError(
            f"Plugin file too large: {len(payload)} bytes, max {MAX_PLUGIN_BYTES}"
        )
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise InstallError(f"Plugin file must be UTF-8: {exc}") from exc
    try:
        compile(text, "<plugin>", "exec")
    except SyntaxError as exc:
        raise InstallError(f"Plugin has a syntax error: {exc}") from exc


def _tenant_dir(manager: PluginWorkerManager, tenant_id: UUID) -> Path:
    d = manager.tenant_dir(tenant_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------


async def install_plugin(
    *,
    manager: PluginWorkerManager,
    tenant_id: UUID,
    user_id: UUID | None,
    filename: str,
    algorithm_id: str,
    payload: bytes,
) -> InstallResult:
    """Validate, write, load, register, and persist a tenant plugin."""
    name = _validate_filename(filename)
    _validate_payload(payload)
    algo_id = (algorithm_id or "").strip()
    if not algo_id:
        raise InstallError("algorithm_id is required")
    if algo_id in AlgorithmRegistry._algorithms:
        # Hard rule: tenants can't shadow built-ins. Forces them to choose a
        # unique id, which keeps the algo picker unambiguous.
        raise InstallError(
            f"Algorithm id {algo_id!r} is reserved by a built-in. "
            "Pick a different id in your plugin's manifest."
        )

    sha256 = hashlib.sha256(payload).hexdigest()
    target_dir = _tenant_dir(manager, tenant_id)
    target_path = target_dir / name

    # Tell the worker to unload any previous version under the same algo_id
    # before we overwrite the file. Best-effort — a fresh worker won't have
    # it loaded and that's fine.
    handle = await manager.get_handle(tenant_id)
    try:
        await handle.unload(algo_id)
    except PluginRuntimeError:
        pass  # not loaded — expected on first install

    target_path.write_bytes(payload)

    try:
        info = await handle.load_plugin(target_path, algo_id)
    except PluginRuntimeError as exc:
        # Roll back the file write — we don't want a half-installed plugin
        # in the directory the next startup will discover.
        try:
            target_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise InstallError(f"Plugin failed to load: {exc.error_message}") from exc

    manifest = _manifest_from_payload(info["manifest"])
    if manifest.id != algo_id:
        # Worker validates this too, but we re-check defensively.
        target_path.unlink(missing_ok=True)
        raise InstallError(
            f"Plugin's manifest.id {manifest.id!r} does not match "
            f"declared algorithm_id {algo_id!r}"
        )

    warmup = int(info.get("warmup_bars", 0))
    adapter = SandboxedAlgorithm(
        manager=manager,
        tenant_id=tenant_id,
        plugin_path=target_path,
        algo_id=algo_id,
        manifest=manifest,
        warmup_bars=warmup,
    )
    AlgorithmRegistry.register_for_tenant(tenant_id, adapter)

    plugin_id = await _upsert_db_row(
        tenant_id=tenant_id,
        user_id=user_id,
        algorithm_id=algo_id,
        name=manifest.name,
        filename=name,
        sha256=sha256,
        manifest_payload=info["manifest"],
        warmup_bars=warmup,
    )

    logger.info(
        "Installed plugin %s for tenant %s (sha256=%s)",
        algo_id, tenant_id, sha256[:12],
    )
    await audit.record(
        "plugin.install",
        resource_type="plugin",
        resource_id=plugin_id,
        tenant_id=tenant_id,
        user_id=user_id,
        extra={
            "algorithm_id": algo_id,
            "name": manifest.name,
            "filename": name,
            "sha256": sha256,
        },
    )
    return InstallResult(
        plugin_id=plugin_id,
        algorithm_id=algo_id,
        name=manifest.name,
        filename=name,
        sha256=sha256,
        warmup_bars=warmup,
    )


async def _upsert_db_row(
    *,
    tenant_id: UUID,
    user_id: UUID | None,
    algorithm_id: str,
    name: str,
    filename: str,
    sha256: str,
    manifest_payload: dict[str, Any],
    warmup_bars: int,
) -> UUID:
    manifest_json = json.dumps(manifest_payload, separators=(",", ":"))
    async with get_session() as session:
        existing = (
            await session.execute(
                select(TenantPluginModel).where(
                    TenantPluginModel.tenant_id == tenant_id,
                    TenantPluginModel.algorithm_id == algorithm_id,
                )
            )
        ).scalar_one_or_none()
        if existing is not None:
            existing.filename = filename
            existing.sha256 = sha256
            existing.manifest_json = manifest_json
            existing.warmup_bars = warmup_bars
            existing.name = name
            existing.is_enabled = True
            existing.last_error = None
            if user_id is not None:
                existing.uploaded_by = user_id
            await session.commit()
            return existing.id

        row = TenantPluginModel(
            id=uuid4(),
            tenant_id=tenant_id,
            algorithm_id=algorithm_id,
            name=name,
            filename=filename,
            sha256=sha256,
            manifest_json=manifest_json,
            warmup_bars=warmup_bars,
            uploaded_by=user_id,
            is_enabled=True,
            last_error=None,
        )
        session.add(row)
        await session.commit()
        return row.id


# ---------------------------------------------------------------------------
# Uninstall
# ---------------------------------------------------------------------------


async def set_plugin_enabled(
    *,
    manager: PluginWorkerManager,
    tenant_id: UUID,
    algorithm_id: str,
    enabled: bool,
) -> bool:
    """Toggle a plugin's enabled state.

    Disabling unregisters from the per-tenant overlay and unloads from the
    worker (best-effort) but keeps the file and DB row so re-enabling is
    cheap. Returns True if the row existed and the state changed.
    """
    async with get_session() as session:
        row = (
            await session.execute(
                select(TenantPluginModel).where(
                    TenantPluginModel.tenant_id == tenant_id,
                    TenantPluginModel.algorithm_id == algorithm_id,
                )
            )
        ).scalar_one_or_none()
        if row is None:
            return False
        if row.is_enabled == enabled:
            return False
        row.is_enabled = enabled
        if enabled:
            row.last_error = None
        manifest_payload = json.loads(row.manifest_json)
        manifest = _manifest_from_payload(manifest_payload)
        warmup = row.warmup_bars
        filename = row.filename
        await session.commit()

    if enabled:
        plugin_path = manager.tenant_dir(tenant_id) / filename
        adapter = SandboxedAlgorithm(
            manager=manager,
            tenant_id=tenant_id,
            plugin_path=plugin_path,
            algo_id=algorithm_id,
            manifest=manifest,
            warmup_bars=warmup,
        )
        AlgorithmRegistry.register_for_tenant(tenant_id, adapter)
    else:
        AlgorithmRegistry.unregister_for_tenant(tenant_id, algorithm_id)
        if manager.has_handle(tenant_id):
            try:
                handle = await manager.get_handle(tenant_id)
                await handle.unload(algorithm_id)
            except PluginRuntimeError:
                pass

    await audit.record(
        "plugin.enable" if enabled else "plugin.disable",
        resource_type="plugin",
        resource_id=algorithm_id,
        tenant_id=tenant_id,
    )
    return True


async def record_plugin_error(
    *,
    tenant_id: UUID,
    algorithm_id: str,
    message: str,
) -> None:
    """Persist a plugin's most recent runtime error to the DB row.

    Called by the adapter when ``on_bar_async`` or ``replay_bars`` raises so
    the Plugins page can show the user what went wrong without them having
    to grep server logs.
    """
    truncated = (message or "").strip()[:2000] or "(empty)"
    async with get_session() as session:
        row = (
            await session.execute(
                select(TenantPluginModel).where(
                    TenantPluginModel.tenant_id == tenant_id,
                    TenantPluginModel.algorithm_id == algorithm_id,
                )
            )
        ).scalar_one_or_none()
        if row is None:
            return
        row.last_error = truncated
        await session.commit()


async def uninstall_plugin(
    *,
    manager: PluginWorkerManager,
    tenant_id: UUID,
    algorithm_id: str,
) -> bool:
    """Remove a tenant plugin from the overlay, worker, disk, and DB."""
    AlgorithmRegistry.unregister_for_tenant(tenant_id, algorithm_id)

    if manager.has_handle(tenant_id):
        try:
            handle = await manager.get_handle(tenant_id)
            await handle.unload(algorithm_id)
        except PluginRuntimeError:
            pass  # already gone — same outcome

    async with get_session() as session:
        row = (
            await session.execute(
                select(TenantPluginModel).where(
                    TenantPluginModel.tenant_id == tenant_id,
                    TenantPluginModel.algorithm_id == algorithm_id,
                )
            )
        ).scalar_one_or_none()
        if row is None:
            return False
        filename = row.filename
        await session.delete(row)
        await session.commit()

    plugin_path = manager.tenant_dir(tenant_id) / filename
    try:
        plugin_path.unlink(missing_ok=True)
    except OSError:
        logger.warning("Could not delete plugin file %s", plugin_path)

    await audit.record(
        "plugin.uninstall",
        resource_type="plugin",
        resource_id=algorithm_id,
        tenant_id=tenant_id,
        extra={"filename": filename},
    )
    return True


# ---------------------------------------------------------------------------
# Listing & startup hydration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InstalledPlugin:
    """UI-safe view of a tenant's installed plugin."""

    id: UUID
    algorithm_id: str
    name: str
    filename: str
    sha256: str
    warmup_bars: int
    is_enabled: bool
    last_error: str | None


async def list_for_tenant(tenant_id: UUID) -> list[InstalledPlugin]:
    async with get_session() as session:
        rows = (
            await session.execute(
                select(TenantPluginModel)
                .where(TenantPluginModel.tenant_id == tenant_id)
                .order_by(TenantPluginModel.name)
            )
        ).scalars().all()
    return [
        InstalledPlugin(
            id=r.id,
            algorithm_id=r.algorithm_id,
            name=r.name,
            filename=r.filename,
            sha256=r.sha256,
            warmup_bars=r.warmup_bars,
            is_enabled=r.is_enabled,
            last_error=r.last_error,
        )
        for r in rows
    ]


async def restore_for_tenant(
    *,
    manager: PluginWorkerManager,
    tenant_id: UUID,
) -> int:
    """Rebuild a tenant's overlay from DB rows. Does NOT spawn workers.

    Workers stay lazy — the SandboxedAlgorithm adapter spawns one on its
    first ``on_bar_async`` call. Returns the count of plugins restored.
    """
    async with get_session() as session:
        rows = (
            await session.execute(
                select(TenantPluginModel).where(
                    TenantPluginModel.tenant_id == tenant_id,
                    TenantPluginModel.is_enabled.is_(True),
                )
            )
        ).scalars().all()

    count = 0
    tenant_dir = manager.tenant_dir(tenant_id)
    for row in rows:
        plugin_path = tenant_dir / row.filename
        if not plugin_path.is_file():
            logger.warning(
                "Plugin file missing for tenant %s: %s", tenant_id, plugin_path,
            )
            continue
        try:
            manifest_payload = json.loads(row.manifest_json)
            manifest = _manifest_from_payload(manifest_payload)
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning(
                "Plugin %s for tenant %s has corrupt manifest: %s",
                row.algorithm_id, tenant_id, exc,
            )
            continue
        adapter = SandboxedAlgorithm(
            manager=manager,
            tenant_id=tenant_id,
            plugin_path=plugin_path,
            algo_id=row.algorithm_id,
            manifest=manifest,
            warmup_bars=row.warmup_bars,
        )
        AlgorithmRegistry.register_for_tenant(tenant_id, adapter)
        count += 1

    if count:
        logger.info("Restored %d plugin(s) for tenant %s", count, tenant_id)
    return count


async def restore_all_at_startup(manager: PluginWorkerManager) -> int:
    """Walk every tenant with at least one enabled plugin row and restore.

    Called once during ``ui/app.py:_startup``. Workers stay un-spawned;
    they boot lazily on first use.
    """
    async with get_session() as session:
        result = await session.execute(
            select(TenantPluginModel.tenant_id)
            .where(TenantPluginModel.is_enabled.is_(True))
            .distinct()
        )
        tenant_ids = [tid for (tid,) in result.all()]

    total = 0
    for tid in tenant_ids:
        total += await restore_for_tenant(manager=manager, tenant_id=tid)
    return total
