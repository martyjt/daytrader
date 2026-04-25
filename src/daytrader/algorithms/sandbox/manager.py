"""Plugin worker lifecycle: one subprocess per tenant.

The manager owns ``PluginWorkerHandle`` objects keyed by tenant id. Each
handle wraps an ``asyncio.subprocess.Process`` and serializes calls to
the worker through a single asyncio.Lock — one request in flight at a
time per worker. The trading loop calls into a handle through the
``SandboxedAlgorithm`` adapter; the adapter does not know about the
manager.

Crash and timeout policy:

* Worker process exits unexpectedly → mark the handle dead, drop it from
  the manager. Next call respawns lazily.
* Per-call timeout (default 5s for ``on_bar``, configurable for
  ``replay_bars``) → SIGTERM, then SIGKILL after 1s; mark dead.
* Worker raises an exception → returned to the caller as
  ``PluginRuntimeError``; the worker stays alive for the next call.

The handle is the single chokepoint — once a request enters it, exactly
one of {response frame, timeout, crash} happens.
"""

from __future__ import annotations

import asyncio
import logging
import os
import struct
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any
from uuid import UUID

from . import protocol

logger = logging.getLogger(__name__)


# Stdlib subprocess flags differ by platform; collect them here.
def _platform_spawn_kwargs() -> dict[str, Any]:
    if sys.platform == "win32":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP
        if hasattr(subprocess, "CREATE_NO_WINDOW"):
            flags |= subprocess.CREATE_NO_WINDOW
        return {"creationflags": flags}
    return {"start_new_session": True}


# Environment passed to the worker. Anything secret-shaped is stripped.
def _scrubbed_env() -> dict[str, str]:
    """Build a minimal env for the worker — PATH and Python knobs only."""
    keep_keys = {
        "PATH",
        "PYTHONPATH",
        "PYTHONHOME",
        "SYSTEMROOT",  # Windows needs this to bootstrap Python
        "TEMP",
        "TMP",
        "LANG",
        "LC_ALL",
    }
    out: dict[str, str] = {}
    for key in keep_keys:
        v = os.environ.get(key)
        if v is not None:
            out[key] = v
    out["PYTHONDONTWRITEBYTECODE"] = "1"
    # Block parent's site-packages user-overrides from sneaking in.
    out["PYTHONNOUSERSITE"] = "1"
    return out


class PluginRuntimeError(Exception):
    """Raised when a plugin's call returned an error frame."""

    def __init__(self, error_type: str, message: str, traceback: str = "") -> None:
        super().__init__(f"{error_type}: {message}")
        self.error_type = error_type
        self.error_message = message
        self.traceback = traceback


class PluginWorkerCrashed(Exception):
    """Raised when the worker process died mid-call or before a response."""


class PluginWorkerTimeout(Exception):
    """Raised when a call exceeded its deadline."""


# ---------------------------------------------------------------------------
# Async framing
# ---------------------------------------------------------------------------


async def _read_frame_async(reader: asyncio.StreamReader) -> dict[str, Any] | None:
    """Async equivalent of protocol.read_frame, against an asyncio reader."""
    try:
        header = await reader.readexactly(4)
    except asyncio.IncompleteReadError as exc:
        if exc.partial:
            raise protocol.ProtocolError(
                f"Truncated frame header: got {len(exc.partial)} of 4 bytes"
            ) from exc
        return None
    (length,) = struct.unpack(">I", header)
    if length == 0:
        raise protocol.ProtocolError("Zero-length frame")
    if length > protocol.MAX_FRAME_SIZE:
        raise protocol.ProtocolError(f"Frame too large: {length}")
    try:
        body = await reader.readexactly(length)
    except asyncio.IncompleteReadError as exc:
        raise protocol.ProtocolError(
            f"Truncated frame body: wanted {length}, got {len(exc.partial)}"
        ) from exc
    import json as _json
    try:
        obj = _json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, _json.JSONDecodeError) as exc:
        raise protocol.ProtocolError(f"Frame is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise protocol.ProtocolError("Frame must be an object")
    return obj


def _write_frame_to_writer(writer: asyncio.StreamWriter, obj: dict[str, Any]) -> None:
    import json as _json
    body = _json.dumps(obj, separators=(",", ":")).encode("utf-8")
    if len(body) > protocol.MAX_FRAME_SIZE:
        raise protocol.ProtocolError(f"Frame too large to send: {len(body)}")
    writer.write(struct.pack(">I", len(body)))
    writer.write(body)


# ---------------------------------------------------------------------------
# Worker handle
# ---------------------------------------------------------------------------


DEFAULT_CALL_TIMEOUT = 5.0
DEFAULT_REPLAY_TIMEOUT = 60.0
TERMINATE_GRACE = 1.0


class PluginWorkerHandle:
    """One subprocess. One asyncio.Lock around it. Single-flight."""

    def __init__(
        self,
        tenant_id: UUID,
        cwd: Path,
        worker_argv: list[str] | None = None,
    ) -> None:
        self.tenant_id = tenant_id
        self._cwd = cwd
        self._argv = worker_argv or [
            sys.executable,
            "-m",
            "daytrader.algorithms.sandbox.worker_main",
        ]
        self._proc: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()
        self._stderr_drainer: asyncio.Task[None] | None = None
        self._dead = False
        self._stderr_tail: list[str] = []  # last few stderr lines for diagnostics

    # ---- lifecycle ------------------------------------------------------

    async def start(self) -> None:
        """Spawn the subprocess. No-op if already running."""
        if self._proc is not None:
            return
        self._cwd.mkdir(parents=True, exist_ok=True)
        self._proc = await asyncio.create_subprocess_exec(
            *self._argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self._cwd),
            env=_scrubbed_env(),
            close_fds=True,
            **_platform_spawn_kwargs(),
        )
        self._dead = False
        self._stderr_tail = []
        self._stderr_drainer = asyncio.create_task(
            self._drain_stderr(), name=f"plugin-worker-stderr-{self.tenant_id}"
        )
        logger.info(
            "Started plugin worker for tenant %s (pid=%s)",
            self.tenant_id, self._proc.pid,
        )

    async def _drain_stderr(self) -> None:
        """Pump stderr → log + bounded tail buffer. Never blocks the request loop."""
        if self._proc is None or self._proc.stderr is None:
            return
        try:
            async for line in self._proc.stderr:
                msg = line.decode("utf-8", errors="replace").rstrip()
                if msg:
                    logger.warning("[plugin %s] %s", self.tenant_id, msg)
                    self._stderr_tail.append(msg)
                    # Bound the tail — no need to keep more than the last 20 lines.
                    if len(self._stderr_tail) > 20:
                        del self._stderr_tail[: len(self._stderr_tail) - 20]
        except Exception:  # pragma: no cover — drainer is best-effort
            logger.exception("stderr drain loop crashed for tenant %s", self.tenant_id)

    @property
    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.returncode is None and not self._dead

    @property
    def stderr_tail(self) -> list[str]:
        return list(self._stderr_tail)

    async def shutdown(self, timeout: float = 2.0) -> None:
        """Terminate the worker. Best-effort — kill if it hangs."""
        if self._proc is None:
            return
        proc = self._proc
        self._proc = None
        self._dead = True
        if proc.returncode is None:
            try:
                proc.terminate()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(proc.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                await proc.wait()
        if self._stderr_drainer is not None:
            self._stderr_drainer.cancel()
            try:
                await self._stderr_drainer
            except (asyncio.CancelledError, Exception):
                pass
            self._stderr_drainer = None
        logger.info("Stopped plugin worker for tenant %s", self.tenant_id)

    # ---- request/response ------------------------------------------------

    async def call(
        self,
        op: str,
        timeout: float = DEFAULT_CALL_TIMEOUT,
        **kwargs: Any,
    ) -> Any:
        """Send a request, await its response, return the ``result`` field.

        Auto-spawns the worker if needed. Restarts once on PluginWorkerCrashed.
        Raises PluginRuntimeError for plugin-side errors, PluginWorkerTimeout
        for deadlines, PluginWorkerCrashed if the worker died and could not
        be restarted.
        """
        try:
            return await self._call_once(op, timeout=timeout, **kwargs)
        except PluginWorkerCrashed:
            logger.warning(
                "Plugin worker for tenant %s crashed; respawning once",
                self.tenant_id,
            )
            await self.shutdown()
            return await self._call_once(op, timeout=timeout, **kwargs)

    async def _call_once(self, op: str, *, timeout: float, **kwargs: Any) -> Any:
        async with self._lock:
            if not self.is_alive:
                await self.start()
            assert self._proc is not None
            assert self._proc.stdin is not None
            assert self._proc.stdout is not None

            rid = uuid.uuid4().hex
            request = {"op": op, "rid": rid, **kwargs}

            try:
                _write_frame_to_writer(self._proc.stdin, request)
                await self._proc.stdin.drain()
            except (BrokenPipeError, ConnectionResetError) as exc:
                raise PluginWorkerCrashed(
                    f"Worker stdin closed before request: {exc}"
                ) from exc

            try:
                response = await asyncio.wait_for(
                    _read_frame_async(self._proc.stdout), timeout=timeout
                )
            except asyncio.TimeoutError as exc:
                # Kill the worker — we can't trust its state after a hang.
                await self._terminate_now()
                raise PluginWorkerTimeout(
                    f"Worker {op} exceeded {timeout}s for tenant {self.tenant_id}"
                ) from exc
            except protocol.ProtocolError as exc:
                await self._terminate_now()
                raise PluginWorkerCrashed(f"Protocol error: {exc}") from exc

            if response is None:
                raise PluginWorkerCrashed("Worker closed stdout before responding")
            if response.get("rid") != rid:
                # rid mismatch means request/response desync — treat as crash.
                await self._terminate_now()
                raise PluginWorkerCrashed(
                    f"Response rid mismatch: got {response.get('rid')}, expected {rid}"
                )

            if not response.get("ok"):
                err = response.get("error") or {}
                raise PluginRuntimeError(
                    error_type=str(err.get("type", "Error")),
                    message=str(err.get("message", "")),
                    traceback=str(err.get("traceback", "")),
                )
            return response.get("result")

    async def _terminate_now(self) -> None:
        """Force-kill the worker after a fatal error. Releases the handle."""
        if self._proc is None:
            return
        proc = self._proc
        try:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=TERMINATE_GRACE)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
        except ProcessLookupError:
            pass
        self._proc = None
        self._dead = True
        if self._stderr_drainer is not None:
            self._stderr_drainer.cancel()
            self._stderr_drainer = None

    # ---- typed convenience --------------------------------------------

    async def load_plugin(self, plugin_path: Path, algo_id: str) -> dict[str, Any]:
        """Load a plugin file. Returns its serialized manifest + warmup_bars."""
        return await self.call(
            "load_plugin",
            timeout=DEFAULT_CALL_TIMEOUT,
            path=str(plugin_path),
            algo_id=algo_id,
        )

    async def unload(self, algo_id: str) -> None:
        await self.call("unload", algo_id=algo_id)

    async def on_bar(self, algo_id: str, ctx_payload: dict[str, Any]) -> dict[str, Any]:
        return await self.call(
            "on_bar",
            timeout=DEFAULT_CALL_TIMEOUT,
            algo_id=algo_id,
            ctx=ctx_payload,
        )

    async def replay_bars(
        self,
        algo_id: str,
        contexts: list[dict[str, Any]],
        timeout: float = DEFAULT_REPLAY_TIMEOUT,
    ) -> list[dict[str, Any]]:
        result = await self.call(
            "replay_bars",
            timeout=timeout,
            algo_id=algo_id,
            contexts=contexts,
        )
        return result.get("results", [])

    async def ping(self) -> bool:
        try:
            r = await self.call("ping", timeout=2.0)
            return bool(r and r.get("pong"))
        except (PluginWorkerCrashed, PluginWorkerTimeout):
            return False


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class PluginWorkerManager:
    """Owns one ``PluginWorkerHandle`` per tenant.

    Workers are spawned lazily on first ``get_handle`` call. The manager is
    a process-lifetime singleton — the app instantiates it at startup and
    calls ``shutdown_all()`` on shutdown.
    """

    def __init__(self, base_dir: Path) -> None:
        self._base_dir = base_dir
        self._handles: dict[UUID, PluginWorkerHandle] = {}
        self._lock = asyncio.Lock()

    def tenant_dir(self, tenant_id: UUID) -> Path:
        return self._base_dir / str(tenant_id)

    async def get_handle(self, tenant_id: UUID) -> PluginWorkerHandle:
        async with self._lock:
            handle = self._handles.get(tenant_id)
            if handle is None or not handle.is_alive:
                if handle is not None:
                    # Existing handle is dead — replace it.
                    await handle.shutdown()
                handle = PluginWorkerHandle(
                    tenant_id=tenant_id, cwd=self.tenant_dir(tenant_id)
                )
                await handle.start()
                self._handles[tenant_id] = handle
            return handle

    async def shutdown_tenant(self, tenant_id: UUID) -> None:
        async with self._lock:
            handle = self._handles.pop(tenant_id, None)
        if handle is not None:
            await handle.shutdown()

    async def shutdown_all(self) -> None:
        async with self._lock:
            handles = list(self._handles.values())
            self._handles.clear()
        for h in handles:
            await h.shutdown()

    def has_handle(self, tenant_id: UUID) -> bool:
        return tenant_id in self._handles
