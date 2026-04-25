"""UI service for live broker balance read-out (Risk Center widget)."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Any
from uuid import uuid4

from ..execution.credentials import list_credentials
from ..execution.registry import ExecutionRegistry
from .services import _tenant_id

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LiveBalance:
    broker_name: str
    is_testnet: bool
    balance: Decimal
    ok: bool
    error: str | None = None


_BALANCE_TIMEOUT_SECONDS = 8.0


async def live_broker_balances() -> list[LiveBalance]:
    """Fetch the current cash balance for every connected broker.

    Errors per-broker are caught so one misconfigured key doesn't blank out the
    whole widget. A short per-call timeout keeps a slow exchange from hanging
    the page.
    """
    tenant_id = _tenant_id()
    creds = await list_credentials(tenant_id)
    if not creds:
        return []

    async def _one(cred: Any) -> LiveBalance:
        adapter = await ExecutionRegistry.get_for_tenant(
            tenant_id, cred.broker_name
        )
        if adapter is None:
            return LiveBalance(
                broker_name=cred.broker_name,
                is_testnet=cred.is_testnet,
                balance=Decimal(0),
                ok=False,
                error="adapter unavailable",
            )
        try:
            balance = await asyncio.wait_for(
                adapter.get_balance(uuid4()), timeout=_BALANCE_TIMEOUT_SECONDS
            )
        except TimeoutError:
            return LiveBalance(
                broker_name=cred.broker_name,
                is_testnet=cred.is_testnet,
                balance=Decimal(0),
                ok=False,
                error="timeout",
            )
        except Exception as exc:
            logger.exception("get_balance failed for %s", cred.broker_name)
            return LiveBalance(
                broker_name=cred.broker_name,
                is_testnet=cred.is_testnet,
                balance=Decimal(0),
                ok=False,
                error=str(exc)[:80],
            )
        return LiveBalance(
            broker_name=cred.broker_name,
            is_testnet=cred.is_testnet,
            balance=balance,
            ok=True,
        )

    return list(await asyncio.gather(*[_one(c) for c in creds]))
