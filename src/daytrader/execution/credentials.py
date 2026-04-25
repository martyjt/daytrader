"""Per-tenant broker credentials — encrypted storage + adapter factory.

The CRUD surface is intentionally small: list / add / delete. Updates are
implemented as delete-then-add so we never leave a half-encrypted row.

All credential blobs are JSON-encoded then Fernet-encrypted via
``SecretCodec``. Plaintext only crosses three boundaries:

* :func:`save_credential` — accepts a dict from the UI and encrypts it.
* :func:`build_executor` — decrypts to instantiate a per-tenant adapter.
* :func:`test_connection` — same path, then calls ``get_balance``.

The ``BrokerCredentialModel`` rows are tenant-scoped via ``TenantMixin``,
so the repository layer enforces isolation automatically.
"""

from __future__ import annotations

import contextlib
import json
import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import select

from ..core import audit
from ..core.context import tenant_scope
from ..core.crypto import get_codec
from ..storage.database import get_session
from ..storage.models import BrokerCredentialModel
from .base import ExecutionAdapter

logger = logging.getLogger(__name__)


SUPPORTED_BROKERS: tuple[str, ...] = ("binance", "alpaca")


@dataclass(frozen=True)
class CredentialSummary:
    """Masked, UI-safe view of a stored credential."""

    id: UUID
    broker_name: str
    is_testnet: bool
    api_key_masked: str
    created_at: Any


class CredentialError(Exception):
    """Raised for validation / decryption failures the UI should surface."""


def _mask(value: str) -> str:
    """Return a masked preview, e.g. ``XXXX…1234`` for a 16-char key."""
    if not value:
        return ""
    if len(value) <= 6:
        return "•" * len(value)
    return f"{value[:2]}…{value[-4:]}"


def _payload_for_broker(broker_name: str, fields: dict[str, str]) -> dict[str, str]:
    """Validate + normalise the input fields for a given broker.

    Keeps a tiny per-broker schema so we don't store junk that ``build_executor``
    can't consume.
    """
    name = broker_name.strip().lower()
    if name not in SUPPORTED_BROKERS:
        raise CredentialError(
            f"Unsupported broker {broker_name!r}. "
            f"Supported: {', '.join(SUPPORTED_BROKERS)}"
        )

    api_key = (fields.get("api_key") or "").strip()
    api_secret = (fields.get("api_secret") or "").strip()
    if not api_key or not api_secret:
        raise CredentialError("Both API key and API secret are required")

    return {"api_key": api_key, "api_secret": api_secret}


async def list_credentials(tenant_id: UUID) -> list[CredentialSummary]:
    """Return UI-safe summaries (api_key masked, secret never exposed)."""
    async with get_session() as session:
        rows = (
            await session.execute(
                select(BrokerCredentialModel)
                .where(BrokerCredentialModel.tenant_id == tenant_id)
                .order_by(BrokerCredentialModel.created_at)
            )
        ).scalars().all()

    codec = get_codec()
    summaries: list[CredentialSummary] = []
    for row in rows:
        try:
            payload = json.loads(codec.decrypt(row.credential_data))
            api_key = payload.get("api_key", "")
        except Exception:
            api_key = ""
        summaries.append(
            CredentialSummary(
                id=row.id,
                broker_name=row.broker_name,
                is_testnet=row.is_testnet,
                api_key_masked=_mask(api_key) if api_key else "(unreadable)",
                created_at=row.created_at,
            )
        )
    return summaries


async def save_credential(
    *,
    tenant_id: UUID,
    broker_name: str,
    fields: dict[str, str],
    is_testnet: bool,
) -> UUID:
    """Replace any existing credential for ``(tenant, broker)`` with a fresh row."""
    payload = _payload_for_broker(broker_name, fields)
    encrypted = get_codec().encrypt(json.dumps(payload))
    name = broker_name.strip().lower()

    async with get_session() as session:
        existing = (
            await session.execute(
                select(BrokerCredentialModel).where(
                    BrokerCredentialModel.tenant_id == tenant_id,
                    BrokerCredentialModel.broker_name == name,
                )
            )
        ).scalars().all()
        for row in existing:
            await session.delete(row)

        new_row = BrokerCredentialModel(
            id=uuid4(),
            tenant_id=tenant_id,
            broker_name=name,
            credential_data=encrypted,
            is_testnet=is_testnet,
        )
        session.add(new_row)
        await session.commit()
        new_id = new_row.id
        # Force the per-tenant adapter cache to rebuild on the next call.
        from .registry import ExecutionRegistry
        ExecutionRegistry.invalidate_tenant(tenant_id)
    await audit.record(
        "broker_creds.save",
        resource_type="broker_credential",
        resource_id=new_id,
        tenant_id=tenant_id,
        extra={"broker_name": name, "is_testnet": is_testnet},
    )
    return new_id


async def delete_credential(*, tenant_id: UUID, credential_id: UUID) -> bool:
    async with get_session() as session:
        with tenant_scope(tenant_id):
            row = (
                await session.execute(
                    select(BrokerCredentialModel).where(
                        BrokerCredentialModel.id == credential_id,
                        BrokerCredentialModel.tenant_id == tenant_id,
                    )
                )
            ).scalar_one_or_none()
            if row is None:
                return False
            broker_name = row.broker_name
            await session.delete(row)
            await session.commit()

    from .registry import ExecutionRegistry
    ExecutionRegistry.invalidate_tenant(tenant_id)
    await audit.record(
        "broker_creds.delete",
        resource_type="broker_credential",
        resource_id=credential_id,
        tenant_id=tenant_id,
        extra={"broker_name": broker_name},
    )
    return True


async def get_decrypted(
    tenant_id: UUID, broker_name: str
) -> tuple[dict[str, str], bool] | None:
    """Return ``(payload, is_testnet)`` or ``None`` if no credential is on file."""
    name = broker_name.strip().lower()
    async with get_session() as session:
        row = (
            await session.execute(
                select(BrokerCredentialModel).where(
                    BrokerCredentialModel.tenant_id == tenant_id,
                    BrokerCredentialModel.broker_name == name,
                )
            )
        ).scalar_one_or_none()
    if row is None:
        return None
    try:
        payload = json.loads(get_codec().decrypt(row.credential_data))
    except Exception as exc:
        raise CredentialError(
            f"Failed to decrypt {name} credentials — wrong APP_ENCRYPTION_KEY?"
        ) from exc
    return payload, row.is_testnet


def build_executor(
    broker_name: str, payload: dict[str, str], *, is_testnet: bool
) -> ExecutionAdapter:
    """Instantiate the broker-specific adapter from a decrypted payload."""
    name = broker_name.strip().lower()
    if name == "binance":
        from .binance import BinanceExecutor
        return BinanceExecutor(
            api_key=payload["api_key"],
            api_secret=payload["api_secret"],
            testnet=is_testnet,
        )
    if name == "alpaca":
        from .alpaca import AlpacaExecutor
        return AlpacaExecutor(
            api_key=payload["api_key"],
            api_secret=payload["api_secret"],
            paper=is_testnet,
        )
    raise CredentialError(f"Unsupported broker {broker_name!r}")


async def test_connection(
    *, broker_name: str, fields: dict[str, str], is_testnet: bool
) -> Decimal:
    """Validate connectivity by calling ``get_balance`` with a synthetic persona id.

    Builds a throwaway adapter (does NOT persist or cache it) and returns the
    fetched balance so the UI can show ``Connected · $123.45``.
    """
    payload = _payload_for_broker(broker_name, fields)
    adapter = build_executor(broker_name, payload, is_testnet=is_testnet)
    try:
        return await adapter.get_balance(uuid4())
    finally:
        close = getattr(adapter, "close", None)
        if close is not None:
            with contextlib.suppress(Exception):
                await close()
