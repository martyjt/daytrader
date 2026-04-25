"""Execution adapter registry — paper is global; live brokers are per-tenant.

* ``get("paper")`` — returns the single in-process PaperExecutor.
* ``get_for_tenant(tenant_id, broker)`` — returns a per-tenant adapter built
  from that tenant's encrypted credentials. Cached for the process lifetime
  (broker connections are expensive); ``invalidate_tenant`` clears the cache
  when credentials are added/rotated/removed.

Mirrors the data adapter registry in ``data/adapters/registry.py``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from uuid import UUID

from .base import ExecutionAdapter

logger = logging.getLogger(__name__)


class ExecutionRegistry:
    """Registry of execution adapters."""

    _adapters: dict[str, ExecutionAdapter] = {}  # noqa: RUF012
    _tenant_adapters: dict[tuple[UUID, str], ExecutionAdapter] = {}  # noqa: RUF012
    _tenant_lock = asyncio.Lock()
    _close_tasks: set[asyncio.Task[None]] = set()  # noqa: RUF012

    # ---- global (paper) registry ----------------------------------------

    @classmethod
    def register(cls, adapter: ExecutionAdapter) -> None:
        cls._adapters[adapter.name] = adapter

    @classmethod
    def get(cls, name: str) -> ExecutionAdapter:
        if name not in cls._adapters:
            raise KeyError(
                f"Execution adapter {name!r} not registered. "
                f"Available: {sorted(cls._adapters)}"
            )
        return cls._adapters[name]

    @classmethod
    def available(cls) -> list[str]:
        return sorted(cls._adapters)

    @classmethod
    def auto_register(cls) -> None:
        """Register the global paper executor.

        Live brokers (Binance, Alpaca) are per-tenant — see ``get_for_tenant``.
        """
        from .paper import PaperExecutor

        if "paper" not in cls._adapters:
            cls.register(PaperExecutor())

    @classmethod
    def clear(cls) -> None:
        """Remove all registered adapters (for testing)."""
        cls._adapters.clear()
        cls._tenant_adapters.clear()

    # ---- per-tenant live-broker cache -----------------------------------

    @classmethod
    async def get_for_tenant(
        cls, tenant_id: UUID, broker_name: str
    ) -> ExecutionAdapter | None:
        """Return the per-tenant adapter for ``broker_name`` or ``None`` if unset.

        Builds the adapter lazily from the tenant's encrypted credentials and
        caches it. Returns ``None`` (rather than raising) when the tenant has
        no credentials on file, so trading-loop callers can decide whether to
        fall back to paper or skip.
        """
        name = broker_name.strip().lower()
        if name == "paper":
            return cls.get("paper")

        cache_key = (tenant_id, name)
        cached = cls._tenant_adapters.get(cache_key)
        if cached is not None:
            return cached

        async with cls._tenant_lock:
            cached = cls._tenant_adapters.get(cache_key)
            if cached is not None:
                return cached

            from .credentials import CredentialError, build_executor, get_decrypted

            try:
                result = await get_decrypted(tenant_id, name)
            except CredentialError:
                logger.exception(
                    "Failed to decrypt %s credentials for tenant %s", name, tenant_id
                )
                return None
            if result is None:
                return None

            payload, is_testnet = result
            adapter = build_executor(name, payload, is_testnet=is_testnet)
            cls._tenant_adapters[cache_key] = adapter
            return adapter

    @classmethod
    def invalidate_tenant(cls, tenant_id: UUID, broker_name: str | None = None) -> None:
        """Drop cached adapters for a tenant (call after credential mutations).

        If ``broker_name`` is given, only that one is invalidated; otherwise
        every adapter cached for ``tenant_id`` is dropped.
        """
        if broker_name is None:
            keys = [k for k in cls._tenant_adapters if k[0] == tenant_id]
        else:
            keys = [(tenant_id, broker_name.strip().lower())]
        for key in keys:
            adapter = cls._tenant_adapters.pop(key, None)
            close = getattr(adapter, "close", None)
            if close is None:
                continue
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                continue
            if loop.is_running():
                task = loop.create_task(_safe_close(close))
                cls._close_tasks.add(task)
                task.add_done_callback(cls._close_tasks.discard)

    @classmethod
    def cached_tenant_adapters(
        cls,
    ) -> dict[tuple[UUID, str], ExecutionAdapter]:
        """Snapshot of the per-tenant adapter cache (read-only — for shutdown)."""
        return dict(cls._tenant_adapters)


async def _safe_close(close_fn: Any) -> None:
    try:
        await close_fn()
    except Exception:
        logger.exception("Error closing execution adapter")
