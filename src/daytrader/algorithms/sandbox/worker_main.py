"""Plugin worker subprocess entry point.

Run as ``python -m daytrader.algorithms.sandbox.worker_main``. Reads
length-prefixed JSON frames from stdin, dispatches operations against a
local in-memory plugin registry, writes responses back on stdout.

The worker is the second half of the trust boundary. It cannot import
the storage layer, execution layer, or crypto module — an import hook
installed before any plugin loads blocks those modules. Even if a plugin
tries to ``import daytrader.storage``, it gets ``ImportError`` instead of
a working session factory.

The parent process spawns this with a blank environment (no DATABASE_URL,
no APP_ENCRYPTION_KEY, no broker keys) so even bypassing the import hook
gets a plugin nothing of value.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import importlib.util
import inspect
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any

# These imports happen BEFORE the import hook is installed — they're part of
# the worker's trusted bootstrap. Everything they pull into sys.modules is
# baseline. The hook only blocks NEW imports of forbidden modules.
from ..base import Algorithm
from ...core.context import AlgorithmContext
from ...core.types.signals import Signal
from . import protocol

# Modules a tenant plugin must never import. Even if a plugin reaches the
# storage layer, the env scrub means there's no DATABASE_URL to connect with —
# but we still block the import as defense-in-depth.
_FORBIDDEN_PREFIXES = (
    "daytrader.storage",
    "daytrader.execution",
    "daytrader.core.crypto",
    "daytrader.core.settings",
    "daytrader.auth",
    "daytrader.research",  # promotion / shadow trading touch DB
    "daytrader.ui",
)

logger = logging.getLogger("daytrader.plugin_worker")


class _ForbiddenImportFinder(importlib.abc.MetaPathFinder):
    """Refuses imports of modules a plugin must never reach."""

    def find_spec(
        self,
        fullname: str,
        path: Any = None,
        target: Any = None,
    ) -> importlib.machinery.ModuleSpec | None:
        for prefix in _FORBIDDEN_PREFIXES:
            if fullname == prefix or fullname.startswith(prefix + "."):
                raise ImportError(
                    f"Plugin imports of {fullname!r} are blocked by the "
                    "sandbox. Algorithms can only read features and emit "
                    "signals — they cannot touch the storage, execution, "
                    "or crypto layers."
                )
        return None  # let the rest of the import machinery handle it


def _install_import_hook() -> None:
    sys.meta_path.insert(0, _ForbiddenImportFinder())


# ---------------------------------------------------------------------------
# Plugin loading
# ---------------------------------------------------------------------------


class PluginError(Exception):
    """Raised for plugin-level failures (missing class, manifest mismatch)."""


def _load_plugin_module(plugin_path: Path, algo_id: str) -> Algorithm:
    """Import a plugin file and return an Algorithm instance.

    The module name is namespaced by ``algo_id`` to keep two plugins from
    colliding in this worker's ``sys.modules``. Per-tenant namespacing
    happens at the parent level (one worker per tenant) — this guarantees
    plugin isolation within a tenant.
    """
    if not plugin_path.is_file():
        raise PluginError(f"Plugin file not found: {plugin_path}")

    module_name = f"daytrader_plugin_{algo_id}"
    spec = importlib.util.spec_from_file_location(module_name, plugin_path)
    if spec is None or spec.loader is None:
        raise PluginError(f"Could not build module spec for {plugin_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise

    algo_class: type[Algorithm] | None = None
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if obj is Algorithm or not issubclass(obj, Algorithm):
            continue
        if inspect.isabstract(obj):
            continue
        # Pick the first concrete Algorithm subclass defined in the plugin
        # module itself — skip anything imported from elsewhere.
        if obj.__module__ != module_name:
            continue
        algo_class = obj
        break

    if algo_class is None:
        raise PluginError(
            f"No concrete Algorithm subclass found in {plugin_path.name}"
        )

    instance = algo_class()
    if instance.manifest.id != algo_id:
        raise PluginError(
            f"Plugin manifest id {instance.manifest.id!r} does not match "
            f"declared algo_id {algo_id!r}"
        )
    return instance


# ---------------------------------------------------------------------------
# Manifest serialization
# ---------------------------------------------------------------------------


def _serialize_manifest(algo: Algorithm) -> dict[str, Any]:
    m = algo.manifest
    return {
        "id": m.id,
        "name": m.name,
        "version": m.version,
        "description": m.description,
        "asset_classes": list(m.asset_classes),
        "timeframes": list(m.timeframes),
        "params": [
            {
                "name": p.name,
                "type": p.type,
                "default": p.default,
                "min": p.min,
                "max": p.max,
                "step": p.step,
                "description": p.description,
                "choices": list(p.choices) if p.choices else None,
            }
            for p in m.params
        ],
        "author": m.author,
        "suitable_regimes": list(m.suitable_regimes) if m.suitable_regimes else None,
    }


# ---------------------------------------------------------------------------
# Op handlers
# ---------------------------------------------------------------------------


def _run_on_bar(algo: Algorithm, ctx_payload: dict[str, Any]) -> dict[str, Any]:
    captured_signals: list[Signal] = []
    captured_logs: list[dict[str, Any]] = []

    def emit(sig: Signal) -> None:
        captured_signals.append(sig)

    def log(msg: str, fields: dict[str, Any]) -> None:
        # Serializable fields only — drop anything we can't JSON-encode.
        safe: dict[str, Any] = {"message": msg}
        for k, v in fields.items():
            if v is None or isinstance(v, (bool, int, float, str)):
                safe[k] = v
        captured_logs.append(safe)

    ctx = protocol.deserialize_context(ctx_payload, emit_fn=emit, log_fn=log)
    result = algo.on_bar(ctx)

    # If on_bar returned a Signal that wasn't already captured via emit_fn,
    # add it. This matches the in-process semantics where a returned-but-not-
    # emitted Signal is still the algorithm's intent.
    if result is not None and result not in captured_signals:
        captured_signals.append(result)

    return {
        "signals": [protocol.serialize_signal(s) for s in captured_signals],
        "logs": captured_logs,
    }


def _dispatch(req: dict[str, Any], plugins: dict[str, Algorithm]) -> Any:
    op = req.get("op")
    if not isinstance(op, str):
        raise protocol.ProtocolError("Request missing 'op'")

    if op == "ping":
        return {"pong": True}

    if op == "load_plugin":
        path = req.get("path")
        algo_id = req.get("algo_id")
        if not isinstance(path, str) or not isinstance(algo_id, str):
            raise protocol.ProtocolError("load_plugin requires path + algo_id strings")
        algo = _load_plugin_module(Path(path), algo_id)
        plugins[algo_id] = algo
        return {
            "manifest": _serialize_manifest(algo),
            "warmup_bars": int(algo.warmup_bars()),
        }

    if op == "unload":
        algo_id = req.get("algo_id")
        if not isinstance(algo_id, str):
            raise protocol.ProtocolError("unload requires algo_id")
        plugins.pop(algo_id, None)
        # Also drop from sys.modules so a re-upload picks up a fresh module.
        sys.modules.pop(f"daytrader_plugin_{algo_id}", None)
        return {"unloaded": True}

    # The remaining ops all need an algo_id that's loaded.
    algo_id = req.get("algo_id")
    if not isinstance(algo_id, str):
        raise protocol.ProtocolError(f"{op} requires algo_id")
    algo = plugins.get(algo_id)
    if algo is None:
        raise PluginError(f"Plugin {algo_id!r} not loaded")

    if op == "manifest":
        return _serialize_manifest(algo)

    if op == "warmup_bars":
        return {"warmup_bars": int(algo.warmup_bars())}

    if op == "on_bar":
        ctx_payload = req.get("ctx")
        if not isinstance(ctx_payload, dict):
            raise protocol.ProtocolError("on_bar requires ctx object")
        return _run_on_bar(algo, ctx_payload)

    if op == "replay_bars":
        ctx_list = req.get("contexts")
        if not isinstance(ctx_list, list):
            raise protocol.ProtocolError("replay_bars requires contexts array")
        results: list[dict[str, Any]] = []
        for i, c in enumerate(ctx_list):
            if not isinstance(c, dict):
                raise protocol.ProtocolError(f"contexts[{i}] must be object")
            try:
                results.append(_run_on_bar(algo, c))
            except Exception as exc:
                # Per-bar errors don't abort the whole replay — record and
                # carry on. The algorithm's instance state is already
                # whatever the failed bar left it as.
                results.append({
                    "signals": [],
                    "logs": [],
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                })
        return {"results": results}

    raise protocol.ProtocolError(f"Unknown op {op!r}")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def _serialize_exc(exc: BaseException) -> dict[str, Any]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": "".join(traceback.format_exception(exc)),
    }


def main() -> int:
    # Strip secrets defensively — even though the parent already passes a
    # blank env, a misconfigured caller (test harness, dev runner) might
    # leak them. Belt and braces.
    for key in list(os.environ):
        if any(needle in key.upper() for needle in (
            "DATABASE", "APP_ENCRYPTION", "BINANCE", "ALPACA",
            "SECRET", "API_KEY", "PASSWORD", "TOKEN",
        )):
            os.environ.pop(key, None)

    _install_import_hook()

    plugins: dict[str, Algorithm] = {}
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer

    while True:
        try:
            req = protocol.read_frame(stdin)
        except protocol.ProtocolError as exc:
            # Protocol-level corruption is unrecoverable — there's no rid to
            # respond to. Log to stderr (parent captures it) and exit.
            sys.stderr.write(f"protocol error on read: {exc}\n")
            return 2
        if req is None:
            return 0  # clean EOF — parent closed stdin

        rid = req.get("rid")

        try:
            result = _dispatch(req, plugins)
            protocol.write_frame(stdout, {"rid": rid, "ok": True, "result": result})
        except (protocol.ProtocolError, PluginError, ImportError) as exc:
            protocol.write_frame(stdout, {"rid": rid, "ok": False, "error": _serialize_exc(exc)})
        except Exception as exc:
            # Plugin runtime errors travel back as error frames; the worker
            # stays alive for the next request.
            protocol.write_frame(stdout, {"rid": rid, "ok": False, "error": _serialize_exc(exc)})


if __name__ == "__main__":
    sys.exit(main())
