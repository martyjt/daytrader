"""SandboxedAlgorithm — the in-process façade for a subprocess-backed plugin.

This object satisfies the ``Algorithm`` ABC enough to live in
``AlgorithmRegistry``, but the work happens over a pipe to a worker
subprocess. The trading loop calls ``on_bar_async`` (Phase 6 widens the
runtime contract slightly so async algorithms are first-class). The
synchronous ``on_bar`` raises — the only valid sync path for a sandboxed
plugin is the backtest engine's ``replay_bars`` batch call.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from uuid import UUID

from ..base import Algorithm, AlgorithmManifest, AlgorithmParam
from ...core.context import AlgorithmContext
from ...core.types.signals import Signal
from ...core.types.visualize import PlotTrace, VisualizeContext
from . import protocol
from .manager import PluginRuntimeError, PluginWorkerManager

logger = logging.getLogger(__name__)


def _manifest_from_payload(payload: dict[str, Any]) -> AlgorithmManifest:
    """Reconstruct an AlgorithmManifest from a worker's serialized form."""
    params = [
        AlgorithmParam(
            name=str(p["name"]),
            type=str(p.get("type", "float")),
            default=p.get("default", 0),
            min=p.get("min"),
            max=p.get("max"),
            step=p.get("step"),
            description=str(p.get("description", "")),
            choices=list(p["choices"]) if p.get("choices") else None,
        )
        for p in payload.get("params", [])
    ]
    return AlgorithmManifest(
        id=str(payload["id"]),
        name=str(payload["name"]),
        version=str(payload.get("version", "0.1.0")),
        description=str(payload.get("description", "")),
        asset_classes=list(payload.get("asset_classes", ["crypto", "equities"])),
        timeframes=list(payload.get("timeframes", ["1d"])),
        params=params,
        author=str(payload.get("author", "")),
        suitable_regimes=(
            list(payload["suitable_regimes"])
            if payload.get("suitable_regimes")
            else None
        ),
    )


class SandboxedAlgorithm(Algorithm):
    """Runs in a subprocess. Looks like an Algorithm to the rest of the system.

    Manifest and warmup_bars are cached from the ``load_plugin`` response —
    they don't change after upload, so paying for an RPC every time would
    be wasteful. The sandbox is a leaf algorithm only: it cannot be a DAG
    or Bandit child (deferred to a later phase, see roadmap).
    """

    def __init__(
        self,
        *,
        manager: PluginWorkerManager,
        tenant_id: UUID,
        plugin_path: Path,
        algo_id: str,
        manifest: AlgorithmManifest,
        warmup_bars: int,
    ) -> None:
        self._manager = manager
        self._tenant_id = tenant_id
        self._plugin_path = plugin_path
        self._algo_id = algo_id
        self._manifest = manifest
        self._warmup = int(warmup_bars)

    @classmethod
    async def load(
        cls,
        *,
        manager: PluginWorkerManager,
        tenant_id: UUID,
        plugin_path: Path,
        algo_id: str,
    ) -> SandboxedAlgorithm:
        """Spawn the worker (if needed), load the plugin, return the adapter."""
        handle = await manager.get_handle(tenant_id)
        info = await handle.load_plugin(plugin_path, algo_id)
        manifest = _manifest_from_payload(info["manifest"])
        return cls(
            manager=manager,
            tenant_id=tenant_id,
            plugin_path=plugin_path,
            algo_id=algo_id,
            manifest=manifest,
            warmup_bars=int(info.get("warmup_bars", 0)),
        )

    # ---- Algorithm ABC ---------------------------------------------------

    @property
    def manifest(self) -> AlgorithmManifest:
        return self._manifest

    def warmup_bars(self) -> int:
        return self._warmup

    def on_bar(self, ctx: AlgorithmContext) -> Signal | None:
        # Sandboxed plugins can't run on the synchronous code path. The
        # trading loop calls ``on_bar_async``; the backtest engine calls
        # ``replay_bars``. Reaching this method means a caller didn't get
        # the dispatch branch — fail loudly so we catch it in tests.
        raise NotImplementedError(
            f"Sandboxed plugin {self._algo_id!r} cannot run on the sync "
            "on_bar() path. Use on_bar_async() (live trading) or "
            "replay_bars() (backtest)."
        )

    async def _ensure_loaded(self) -> None:
        """Load the plugin into the current worker, idempotently.

        After a worker crash, the manager respawns a fresh process that knows
        nothing about previously-loaded plugins. The first call from the
        adapter then needs to re-issue ``load_plugin`` before any ``on_bar``.
        Calling load_plugin on a worker that already has the plugin loaded
        replaces it cleanly, so this is safe to invoke any time.
        """
        handle = await self._manager.get_handle(self._tenant_id)
        info = await handle.load_plugin(self._plugin_path, self._algo_id)
        # Refresh cached manifest from the worker — if the file changed since
        # the last cache write the manifest may have shifted too.
        self._manifest = _manifest_from_payload(info["manifest"])
        self._warmup = int(info.get("warmup_bars", self._warmup))

    def _is_not_loaded_error(self, exc: PluginRuntimeError) -> bool:
        return exc.error_type == "PluginError" and "not loaded" in exc.error_message

    async def _record_error(self, message: str) -> None:
        """Persist the plugin's last error to the DB row, best-effort.

        Late-imported to avoid the adapter ↔ installer module cycle. DB
        write failures are swallowed — operational visibility is nice to
        have but mustn't bring the adapter down.
        """
        try:
            from .installer import record_plugin_error
            await record_plugin_error(
                tenant_id=self._tenant_id,
                algorithm_id=self._algo_id,
                message=message,
            )
        except Exception:  # pragma: no cover — defensive
            logger.exception("Failed to record plugin error for %s", self._algo_id)

    async def on_bar_async(self, ctx: AlgorithmContext) -> Signal | None:
        """Run one bar in the worker. Pushes captured signals to ctx.emit_fn."""
        handle = await self._manager.get_handle(self._tenant_id)
        ctx_payload = protocol.serialize_context(ctx)
        try:
            result = await handle.on_bar(self._algo_id, ctx_payload)
        except PluginRuntimeError as exc:
            if self._is_not_loaded_error(exc):
                # Worker was respawned (after crash or first contact) and
                # has no plugin state. Reload and retry once.
                await self._ensure_loaded()
                try:
                    result = await handle.on_bar(self._algo_id, ctx_payload)
                except PluginRuntimeError as exc2:
                    await self._record_error(f"{exc2.error_type}: {exc2.error_message}")
                    return None
            else:
                logger.warning(
                    "Sandboxed plugin %s on_bar failed for tenant %s: %s",
                    self._algo_id, self._tenant_id, exc.error_message,
                )
                await self._record_error(f"{exc.error_type}: {exc.error_message}")
                return None

        signals = [protocol.deserialize_signal(s) for s in result.get("signals", [])]
        for sig in signals:
            ctx.emit_fn(sig)
        for entry in result.get("logs", []):
            msg = entry.pop("message", "")
            ctx.log_fn(msg, entry)
        return signals[0] if signals else None

    async def replay_bars(
        self,
        contexts: list[AlgorithmContext],
        timeout: float | None = None,
    ) -> list[Signal | None]:
        """Run a batch of bars in one round-trip — the backtest hot path.

        Each context is serialized and shipped together; the worker iterates
        ``on_bar`` over them, preserving instance state between bars (which
        lives entirely in the worker). Per-bar errors are returned as
        ``None`` for that slot — they don't abort the batch.
        """
        if not contexts:
            return []
        handle = await self._manager.get_handle(self._tenant_id)
        payloads = [protocol.serialize_context(c) for c in contexts]
        kwargs: dict[str, Any] = {}
        if timeout is not None:
            kwargs["timeout"] = timeout
        try:
            results = await handle.replay_bars(self._algo_id, payloads, **kwargs)
        except PluginRuntimeError as exc:
            if self._is_not_loaded_error(exc):
                await self._ensure_loaded()
                results = await handle.replay_bars(self._algo_id, payloads, **kwargs)
            else:
                await self._record_error(f"{exc.error_type}: {exc.error_message}")
                raise

        out: list[Signal | None] = []
        for ctx, r in zip(contexts, results, strict=True):
            sigs = [protocol.deserialize_signal(s) for s in r.get("signals", [])]
            for s in sigs:
                ctx.emit_fn(s)
            for entry in r.get("logs", []):
                msg = entry.pop("message", "")
                ctx.log_fn(msg, entry)
            out.append(sigs[0] if sigs else None)
        return out

    def visualize(self, vctx: VisualizeContext) -> list[PlotTrace]:
        # Default empty — visualizing through the pipe is feasible (add a
        # ``visualize`` op) but not worth the complexity for Phase 6. The
        # Charts Workbench falls back to plotting score/confidence.
        return []

    # ---- introspection -------------------------------------------------

    @property
    def algo_id(self) -> str:
        return self._algo_id

    @property
    def tenant_id(self) -> UUID:
        return self._tenant_id

    @property
    def plugin_path(self) -> Path:
        return self._plugin_path

    def __repr__(self) -> str:
        return (
            f"SandboxedAlgorithm(algo_id={self._algo_id!r}, "
            f"tenant={self._tenant_id})"
        )

    # Deepcopy guard. CompositeAlgorithm and BanditAllocator deepcopy registry
    # entries to give each node independent state. A SandboxedAlgorithm holding
    # a worker handle reference can't be meaningfully deepcopied — and we've
    # decided not to support DAG composition for sandboxed plugins in Phase 6
    # (see roadmap). Raise here so the failure is loud at compose time.
    def __deepcopy__(self, memo: dict) -> SandboxedAlgorithm:
        raise TypeError(
            f"Sandboxed plugin {self._algo_id!r} cannot be used as a child "
            "of a DAG or Bandit composition. (Sandbox composition is not "
            "supported in Phase 6.)"
        )
