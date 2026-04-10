"""Seed the default tenant and user on first boot (Phase 0 auth stub)."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.settings import AppSettings
from .models import TenantModel, UserModel


async def seed_default_tenant(session: AsyncSession, settings: AppSettings) -> None:
    """Create the default tenant and operator user if they don't exist.

    Idempotent — safe to call on every boot.
    """
    result = await session.execute(
        select(TenantModel).where(TenantModel.id == settings.default_tenant_id)
    )
    if result.scalar_one_or_none() is not None:
        return

    tenant = TenantModel(
        id=settings.default_tenant_id,
        name=settings.default_tenant_name,
    )
    session.add(tenant)

    user = UserModel(
        tenant_id=settings.default_tenant_id,
        email=settings.default_user_email,
        display_name="Operator",
    )
    session.add(user)

    await session.commit()
