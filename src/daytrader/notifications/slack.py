"""SlackNotifier — POST to a tenant's encrypted incoming-webhook URL."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any
from uuid import UUID

import httpx

from .base import Notifier
from .service import resolve_webhook_url

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 5.0


class SlackNotifier(Notifier):
    """Send notifications to a tenant's Slack incoming-webhook URL.

    The URL is stored encrypted on ``TenantModel.notification_webhook_url``
    and decrypted on demand via :func:`resolve_webhook_url`. Tenants
    without a webhook configured silently drop their notifications — the
    UI is what tells them they have nothing wired up.

    HTTP failures are logged and swallowed; a flaky Slack must never
    propagate up into the trading loop.
    """

    def __init__(
        self,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        url_resolver: Callable[[UUID], Awaitable[str | None]] | None = None,
        client_factory: Callable[[], httpx.AsyncClient] | None = None,
    ) -> None:
        self._timeout = timeout
        self._url_resolver = url_resolver or resolve_webhook_url
        self._client_factory = client_factory

    async def notify(
        self,
        tenant_id: UUID,
        message: str,
        *,
        dedupe_key: str | None = None,
    ) -> None:
        try:
            url = await self._url_resolver(tenant_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to resolve webhook URL for tenant %s: %s", tenant_id, exc
            )
            return
        if not url:
            return
        await self._post(url, {"text": message})

    async def _post(self, url: str, payload: dict[str, Any]) -> None:
        try:
            if self._client_factory is not None:
                client = self._client_factory()
            else:
                client = httpx.AsyncClient(timeout=self._timeout)
            async with client as c:
                response = await c.post(url, json=payload)
                if response.status_code >= 400:
                    logger.warning(
                        "Slack webhook returned HTTP %s: %s",
                        response.status_code,
                        response.text[:200],
                    )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Slack webhook POST failed: %s", exc)
