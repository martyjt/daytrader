"""Process-global notifier singleton + tenant webhook resolver + admin helpers."""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

import httpx
from sqlalchemy import select, update

from ..core import audit
from ..core.crypto import get_codec
from ..storage.database import get_session
from ..storage.models import TenantModel
from .base import Notifier

logger = logging.getLogger(__name__)


class WebhookError(Exception):
    """Raised by :func:`test_webhook` for bad URLs or HTTP failures."""


_active_notifier: Notifier | None = None


def set_active_notifier(notifier: Notifier | None) -> None:
    """Install the process-global notifier (called once at startup).

    Idempotent; passing ``None`` clears it (used by tests + shutdown).
    """
    global _active_notifier
    _active_notifier = notifier


def get_active_notifier() -> Notifier | None:
    """Return the active notifier or ``None`` if startup hasn't run."""
    return _active_notifier


async def notify_active(
    tenant_id: UUID,
    message: str,
    *,
    dedupe_key: str | None = None,
) -> None:
    """Convenience wrapper — fire on the active notifier if one is installed.

    Callers in code paths that may run without startup (tests, scripts)
    should use this instead of asserting :func:`get_active_notifier`
    returned a notifier. Failures are swallowed.
    """
    notifier = _active_notifier
    if notifier is None:
        return
    try:
        await notifier.notify(tenant_id, message, dedupe_key=dedupe_key)
    except Exception as exc:  # noqa: BLE001 — notifications must never raise
        logger.warning("Notifier failed for tenant %s: %s", tenant_id, exc)


async def resolve_webhook_url(tenant_id: UUID) -> str | None:
    """Read + decrypt a tenant's notification webhook URL.

    Returns ``None`` if the tenant row is missing, the field is unset,
    or the ciphertext can't be decrypted. The "wrong key" case is
    logged but not raised — a misconfigured webhook should not bring
    down trading.
    """
    async with get_session() as session:
        row = (
            await session.execute(
                select(TenantModel).where(TenantModel.id == tenant_id)
            )
        ).scalar_one_or_none()
    if row is None:
        return None
    encrypted = row.notification_webhook_url
    if not encrypted:
        return None
    try:
        return get_codec().decrypt(encrypted)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Failed to decrypt webhook URL for tenant %s: %s", tenant_id, exc
        )
        return None


async def has_webhook(tenant_id: UUID) -> bool:
    """Return True iff the tenant row has a non-empty webhook URL field."""
    async with get_session() as session:
        row = (
            await session.execute(
                select(TenantModel.notification_webhook_url).where(
                    TenantModel.id == tenant_id
                )
            )
        ).scalar_one_or_none()
    return bool(row)


def _validate_webhook_url(url: str) -> str:
    cleaned = (url or "").strip()
    if not cleaned:
        raise WebhookError("Webhook URL is empty")
    if not cleaned.lower().startswith("https://"):
        raise WebhookError("Webhook URL must start with https://")
    return cleaned


async def save_webhook_url(tenant_id: UUID, url: str) -> None:
    """Encrypt and persist ``url`` on the tenant row. Audited."""
    cleaned = _validate_webhook_url(url)
    encrypted = get_codec().encrypt(cleaned)
    async with get_session() as session:
        await session.execute(
            update(TenantModel)
            .where(TenantModel.id == tenant_id)
            .values(notification_webhook_url=encrypted)
        )
        await session.commit()
    await audit.record(
        "notification.webhook.save",
        resource_type="tenant",
        resource_id=tenant_id,
        tenant_id=tenant_id,
    )


async def clear_webhook_url(tenant_id: UUID) -> None:
    """Wipe the tenant's webhook URL. Audited."""
    async with get_session() as session:
        await session.execute(
            update(TenantModel)
            .where(TenantModel.id == tenant_id)
            .values(notification_webhook_url=None)
        )
        await session.commit()
    await audit.record(
        "notification.webhook.clear",
        resource_type="tenant",
        resource_id=tenant_id,
        tenant_id=tenant_id,
    )


async def send_test_message(
    url: str,
    *,
    message: str = "Daytrader: notification webhook test",
    timeout: float = 5.0,
    client: httpx.AsyncClient | None = None,
) -> None:
    """Validate + POST a probe message. Raises :class:`WebhookError` on failure.

    Used by the admin UI's "Test" button so the user gets immediate
    feedback before they leave the page assuming alerts will fire. The
    name avoids the ``test_`` prefix because pytest auto-collects any
    top-level callable starting with ``test_``.
    """
    cleaned = _validate_webhook_url(url)
    payload: dict[str, Any] = {"text": message}
    try:
        if client is None:
            async with httpx.AsyncClient(timeout=timeout) as c:
                response = await c.post(cleaned, json=payload)
        else:
            response = await client.post(cleaned, json=payload)
    except httpx.HTTPError as exc:
        raise WebhookError(f"Could not reach webhook: {exc}") from exc
    if response.status_code >= 400:
        raise WebhookError(
            f"Webhook returned HTTP {response.status_code}: "
            f"{response.text[:200]}"
        )
