"""User account service: authenticate, create, list, deactivate."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from ..core import audit
from ..storage.database import get_session
from ..storage.models import UserModel
from .password import hash_password, verify_password
from .roles import ROLES
from .session import SessionUser


class AuthError(Exception):
    """Raised for auth failures the UI should surface."""


async def authenticate(email: str, password: str) -> SessionUser:
    """Look up the user by email and verify the password.

    Raises ``AuthError`` on missing user, wrong password, or deactivated account.
    Updates ``last_login_at`` on success.
    """
    email_norm = email.strip().lower()
    async with get_session() as session:
        user = (
            await session.execute(
                select(UserModel).where(UserModel.email == email_norm)
            )
        ).scalar_one_or_none()
        if user is None:
            await audit.record(
                "login.failure",
                extra={"email": email_norm, "reason": "no_user"},
            )
            raise AuthError("Invalid email or password")
        if not user.is_active:
            await audit.record(
                "login.failure",
                resource_type="user",
                resource_id=user.id,
                tenant_id=user.tenant_id,
                user_id=user.id,
                extra={"email": email_norm, "reason": "deactivated"},
            )
            raise AuthError("Account is deactivated — contact an administrator")
        if not verify_password(password, user.password_hash):
            await audit.record(
                "login.failure",
                resource_type="user",
                resource_id=user.id,
                tenant_id=user.tenant_id,
                user_id=user.id,
                extra={"email": email_norm, "reason": "bad_password"},
            )
            raise AuthError("Invalid email or password")

        user.last_login_at = datetime.now(UTC)
        await session.commit()

        await audit.record(
            "login.success",
            resource_type="user",
            resource_id=user.id,
            tenant_id=user.tenant_id,
            user_id=user.id,
        )

        return SessionUser(
            user_id=user.id,
            tenant_id=user.tenant_id,
            email=user.email,
            role=user.role,
            display_name=user.display_name,
        )


async def list_users(session: AsyncSession | None = None) -> list[Any]:
    """Cross-tenant user list — only super admins should call this."""
    if session is not None:
        rows = (
            await session.execute(select(UserModel).order_by(UserModel.created_at))
        ).scalars().all()
        return list(rows)
    async with get_session() as s:
        rows = (
            await s.execute(select(UserModel).order_by(UserModel.created_at))
        ).scalars().all()
        return list(rows)


async def create_user(
    *,
    email: str,
    password: str,
    tenant_id: UUID,
    role: str,
    display_name: str | None = None,
) -> UUID:
    if role not in ROLES:
        raise ValueError(f"Unknown role {role!r}")
    email_norm = email.strip().lower()
    async with get_session() as session:
        existing = (
            await session.execute(select(UserModel).where(UserModel.email == email_norm))
        ).scalar_one_or_none()
        if existing is not None:
            raise AuthError(f"User {email_norm} already exists")
        user = UserModel(
            email=email_norm,
            display_name=display_name,
            password_hash=hash_password(password),
            role=role,
            tenant_id=tenant_id,
            is_active=True,
        )
        session.add(user)
        await session.flush()
        new_id = user.id
        await session.commit()
        return new_id


async def set_active(user_id: UUID, active: bool) -> bool:
    async with get_session() as session:
        result = await session.execute(
            update(UserModel).where(UserModel.id == user_id).values(is_active=active)
        )
        await session.commit()
        return result.rowcount > 0  # type: ignore[attr-defined]


async def change_password(user_id: UUID, new_password: str) -> bool:
    async with get_session() as session:
        result = await session.execute(
            update(UserModel)
            .where(UserModel.id == user_id)
            .values(password_hash=hash_password(new_password))
        )
        await session.commit()
        return result.rowcount > 0  # type: ignore[attr-defined]
