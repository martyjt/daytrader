"""Audit log helper — Phase 7 productionization.

Single entry point :func:`record` writes one ``AuditLogModel`` row per
meaningful tenant-affecting action. The caller can pass ``tenant_id`` /
``user_id`` explicitly (e.g. login flows where no session is established
yet, or super-admin actions on a tenant the actor doesn't belong to);
otherwise we fall back to the active session via
:func:`current_tenant_id` / :func:`current_user_id`.

Audit failures are deliberately swallowed and logged — a write to the
audit table must never abort the user's primary action. Losing one audit
row on a transient DB blip is recoverable; losing the user's persona
because the audit table was unavailable is not.
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID, uuid4

from ..auth.session import current_tenant_id, current_user_id
from ..storage.database import get_session
from ..storage.models import AuditLogModel

logger = logging.getLogger(__name__)


def _safe_tenant_id() -> UUID | None:
    """Read tenant from session, returning None outside a request context.

    ``current_tenant_id()`` accesses NiceGUI's per-request user storage,
    which raises if no request context is active (background tasks,
    tests). The audit module treats "no context" as "no fallback".
    """
    try:
        return current_tenant_id()
    except Exception:
        return None


def _safe_user_id() -> UUID | None:
    try:
        return current_user_id()
    except Exception:
        return None


async def record(
    action: str,
    *,
    resource_type: str | None = None,
    resource_id: Any = None,
    extra: dict[str, Any] | None = None,
    tenant_id: UUID | None = None,
    user_id: UUID | None = None,
) -> None:
    """Insert one audit row.

    ``action``   — short dotted identifier, e.g. ``"persona.create"``.
    ``resource_type``/``resource_id`` — the entity touched (UUID stringified).
    ``extra``    — small dict of additional context; serialised as JSON.
    ``tenant_id``/``user_id`` — explicit overrides; default to session.
    """
    tid = tenant_id if tenant_id is not None else _safe_tenant_id()
    uid = user_id if user_id is not None else _safe_user_id()
    rid_str = str(resource_id) if resource_id is not None else None

    try:
        async with get_session() as session:
            session.add(
                AuditLogModel(
                    id=uuid4(),
                    tenant_id=tid,
                    user_id=uid,
                    action=action,
                    resource_type=resource_type,
                    resource_id=rid_str,
                    extra=dict(extra or {}),
                )
            )
            await session.commit()
    except Exception as exc:
        logger.warning("Failed to write audit row for %s: %s", action, exc)
