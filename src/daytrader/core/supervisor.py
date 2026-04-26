"""Per-tenant background-worker supervisors.

A ``BackgroundSupervisor`` keeps one worker per active tenant alive. It
polls the database periodically; when a new tenant becomes active, the
supervisor calls ``_make_worker(tenant_id)`` and starts the result. When
a tenant goes inactive, the supervisor stops and forgets its worker.

Concrete subclasses define:
- ``_active_tenants()`` — returns the set of tenant_ids that should have
  a running worker right now.
- ``_make_worker(tenant_id)`` — instantiates a fresh worker. The worker
  must expose ``async start()`` and ``async stop()``.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from abc import ABC, abstractmethod
from typing import Protocol
from uuid import UUID

logger = logging.getLogger(__name__)


class _Worker(Protocol):
    async def start(self) -> None: ...
    async def stop(self) -> None: ...


class BackgroundSupervisor(ABC):
    """Base class — keeps one worker per active tenant."""

    name: str = "supervisor"

    def __init__(self, *, supervisor_poll_seconds: float = 60.0) -> None:
        self._poll_seconds = supervisor_poll_seconds
        self._workers: dict[UUID, _Worker] = {}
        self._task: asyncio.Task | None = None
        self._refresh_lock = asyncio.Lock()
        self._stopped = asyncio.Event()

    @abstractmethod
    async def _active_tenants(self) -> set[UUID]:
        """Return the set of tenant_ids that should currently have a worker."""

    @abstractmethod
    def _make_worker(self, tenant_id: UUID) -> _Worker:
        """Construct (but don't start) a fresh worker for the given tenant."""

    @property
    def workers(self) -> dict[UUID, _Worker]:
        """Snapshot of currently running workers (read-only access)."""
        return dict(self._workers)

    async def start(self) -> None:
        """Refresh once immediately, then poll on an interval."""
        if self._task is not None:
            return
        self._stopped.clear()
        await self._refresh()  # synchronous first pass so callers see workers up
        self._task = asyncio.create_task(self._poll_loop(), name=f"{self.name}-poll")
        logger.info("%s started (poll every %.0fs)", self.name, self._poll_seconds)

    async def stop(self) -> None:
        """Stop the poller and every running worker."""
        self._stopped.set()
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

        for tid, worker in list(self._workers.items()):
            try:
                await worker.stop()
            except Exception:
                logger.exception("%s: error stopping worker for tenant %s", self.name, tid)
        self._workers.clear()
        logger.info("%s stopped", self.name)

    async def _poll_loop(self) -> None:
        while not self._stopped.is_set():
            try:
                await asyncio.wait_for(
                    self._stopped.wait(), timeout=self._poll_seconds
                )
                return  # stop requested
            except TimeoutError:
                pass
            try:
                await self._refresh()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("%s refresh failed", self.name)

    async def _refresh(self) -> None:
        """Diff the active set against running workers; spin up / shut down."""
        async with self._refresh_lock:
            try:
                active = await self._active_tenants()
            except Exception:
                logger.exception("%s: _active_tenants failed", self.name)
                return

            running = set(self._workers.keys())
            to_start = active - running
            to_stop = running - active

            for tid in to_start:
                try:
                    worker = self._make_worker(tid)
                    await worker.start()
                    self._workers[tid] = worker
                    logger.info("%s: started worker for tenant %s", self.name, tid)
                except Exception:
                    logger.exception(
                        "%s: failed to start worker for tenant %s", self.name, tid
                    )

            for tid in to_stop:
                if tid not in self._workers:
                    continue
                worker = self._workers.pop(tid)
                try:
                    await worker.stop()
                    logger.info("%s: stopped worker for tenant %s", self.name, tid)
                except Exception:
                    logger.exception(
                        "%s: error stopping worker for tenant %s", self.name, tid
                    )


class SupervisorManager:
    """Group of supervisors started together and stopped together."""

    def __init__(self, supervisors: list[BackgroundSupervisor]) -> None:
        self._supervisors = supervisors

    async def start_all(self) -> None:
        for s in self._supervisors:
            try:
                await s.start()
            except Exception:
                logger.exception("Failed to start supervisor %s", s.name)

    async def stop_all(self) -> None:
        # Reverse so supervisors can have implicit ordering dependencies.
        for s in reversed(self._supervisors):
            try:
                await s.stop()
            except Exception:
                logger.exception("Failed to stop supervisor %s", s.name)

    def by_name(self, name: str) -> BackgroundSupervisor | None:
        return next((s for s in self._supervisors if s.name == name), None)
