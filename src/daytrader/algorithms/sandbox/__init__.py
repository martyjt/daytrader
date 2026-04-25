"""Plugin sandbox — runs tenant-uploaded algorithms in subprocess workers.

The trust boundary is the subprocess pipe. Tenants upload Python files; we
load them in a child process with a blank environment (no DB URL, no
encryption key, no broker secrets). The child speaks a length-prefixed
JSON protocol back to the parent. Built-in algorithms are unaffected —
they keep their fast in-process path.

The active ``PluginWorkerManager`` is process-global: ``ui/app.py`` sets
it at startup, the installer and Plugins page read it via
``get_active_manager``. This avoids the alternative (NiceGUI ``app.state``)
which would couple every consumer to the UI layer.
"""

from __future__ import annotations

from pathlib import Path

from .adapter import SandboxedAlgorithm
from .manager import (
    PluginRuntimeError,
    PluginWorkerCrashed,
    PluginWorkerHandle,
    PluginWorkerManager,
    PluginWorkerTimeout,
)

_active_manager: PluginWorkerManager | None = None


def set_active_manager(manager: PluginWorkerManager | None) -> None:
    """Install the process-global plugin worker manager.

    Called once during ``ui/app.py:_startup``. Idempotent; passing ``None``
    clears it (used by tests).
    """
    global _active_manager
    _active_manager = manager


def get_active_manager() -> PluginWorkerManager:
    """Return the active manager. Raises if startup has not run."""
    if _active_manager is None:
        raise RuntimeError(
            "Plugin sandbox manager is not initialized. "
            "Did the app's startup hook run?"
        )
    return _active_manager


def default_uploads_dir() -> Path:
    """Repo-relative directory where tenant plugin files live."""
    return Path(__file__).resolve().parents[4] / "plugins" / "uploads"


__all__ = [
    "PluginRuntimeError",
    "PluginWorkerCrashed",
    "PluginWorkerHandle",
    "PluginWorkerManager",
    "PluginWorkerTimeout",
    "SandboxedAlgorithm",
    "default_uploads_dir",
    "get_active_manager",
    "set_active_manager",
]
