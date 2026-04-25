"""Invite tokens for invite-only registration.

Super admins (cross-tenant) and tenant owners (within-tenant) generate
invites; ``redeem_invite`` consumes one and creates the user account.
"""

from __future__ import annotations

import secrets
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

from sqlalchemy import select

from ..core.settings import get_settings
from ..storage.database import get_session
from ..storage.models import UserInviteModel, UserModel
from .password import hash_password
from .roles import ROLES
from .service import AuthError


def _new_token() -> str:
    return secrets.token_urlsafe(32)


def _as_utc(dt: datetime) -> datetime:
    """SQLite returns naive datetimes; treat them as UTC for comparisons."""
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def is_expired(invite: Any) -> bool:
    return _as_utc(invite.expires_at) < datetime.now(timezone.utc)


async def create_invite(
    *,
    email: str,
    role: str,
    tenant_id: UUID | None,
    invited_by: UUID,
    ttl_hours: int | None = None,
) -> str:
    if role not in ROLES:
        raise ValueError(f"Unknown role {role!r}")

    settings = get_settings()
    ttl = ttl_hours if ttl_hours is not None else settings.invite_token_ttl_hours
    token = _new_token()
    async with get_session() as session:
        invite = UserInviteModel(
            token=token,
            email=email.strip().lower(),
            role=role,
            tenant_id=tenant_id,
            invited_by=invited_by,
            expires_at=datetime.now(timezone.utc) + timedelta(hours=ttl),
        )
        session.add(invite)
        await session.commit()
    return token


async def list_pending_invites() -> list[Any]:
    async with get_session() as session:
        rows = (
            await session.execute(
                select(UserInviteModel)
                .where(UserInviteModel.used_at.is_(None))
                .order_by(UserInviteModel.created_at.desc())
            )
        ).scalars().all()
        return list(rows)


async def get_invite(token: str) -> Any | None:
    async with get_session() as session:
        return (
            await session.execute(
                select(UserInviteModel).where(UserInviteModel.token == token)
            )
        ).scalar_one_or_none()


async def redeem_invite(
    *,
    token: str,
    password: str,
    display_name: str | None = None,
) -> UUID:
    """Consume an invite and create the user. Returns the new user_id."""
    from ..storage.models import TenantModel  # local to avoid cycles

    async with get_session() as session:
        invite = (
            await session.execute(
                select(UserInviteModel).where(UserInviteModel.token == token)
            )
        ).scalar_one_or_none()
        if invite is None:
            raise AuthError("Invite not found")
        if invite.used_at is not None:
            raise AuthError("Invite already used")
        if is_expired(invite):
            raise AuthError("Invite has expired")

        # New tenants: invite.tenant_id is None → create one named after the email
        tenant_id = invite.tenant_id
        if tenant_id is None:
            tenant = TenantModel(name=invite.email)
            session.add(tenant)
            await session.flush()
            tenant_id = tenant.id

        existing = (
            await session.execute(
                select(UserModel).where(UserModel.email == invite.email)
            )
        ).scalar_one_or_none()
        if existing is not None:
            raise AuthError(f"User {invite.email} already exists")

        user = UserModel(
            email=invite.email,
            display_name=display_name,
            password_hash=hash_password(password),
            role=invite.role,
            tenant_id=tenant_id,
            is_active=True,
        )
        session.add(user)
        await session.flush()

        invite.used_at = datetime.now(timezone.utc)
        await session.commit()
        return user.id
