"""Seed the default tenant and super-admin user on first boot."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..auth.password import hash_password
from ..auth.roles import ROLE_SUPER_ADMIN
from ..core.settings import AppSettings
from .models import TenantModel, UserModel


async def seed_default_tenant(session: AsyncSession, settings: AppSettings) -> None:
    """Create the default tenant + bootstrap super admin if missing.

    Idempotent. Existing super-admin users keep their current password — only
    a brand-new install (or an empty ``password_hash`` left over from the
    Phase-0 stub) gets the bootstrap password applied.
    """
    tenant = (
        await session.execute(
            select(TenantModel).where(TenantModel.id == settings.default_tenant_id)
        )
    ).scalar_one_or_none()
    if tenant is None:
        tenant = TenantModel(
            id=settings.default_tenant_id,
            name=settings.default_tenant_name,
        )
        session.add(tenant)
        await session.flush()

    bootstrap_email = settings.bootstrap_admin_email.strip().lower()
    user = (
        await session.execute(
            select(UserModel).where(UserModel.email == bootstrap_email)
        )
    ).scalar_one_or_none()

    if user is None:
        user = UserModel(
            tenant_id=settings.default_tenant_id,
            email=bootstrap_email,
            display_name="Super Admin",
            password_hash=hash_password(
                settings.bootstrap_admin_password.get_secret_value()
            ),
            role=ROLE_SUPER_ADMIN,
            is_active=True,
        )
        session.add(user)
    elif not user.password_hash:
        # Phase-0 row carried over without a password — set the bootstrap.
        user.password_hash = hash_password(
            settings.bootstrap_admin_password.get_secret_value()
        )
        if user.role != ROLE_SUPER_ADMIN:
            user.role = ROLE_SUPER_ADMIN

    await session.commit()
