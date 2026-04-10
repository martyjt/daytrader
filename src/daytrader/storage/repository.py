"""Tenant-scoped generic CRUD repository.

Every query automatically injects ``WHERE tenant_id = <current_tenant>``.
If no tenant is in scope, ``current_tenant()`` raises — which means you
structurally cannot forget to scope a query. Cross-tenant leaks become
impossible rather than merely unlikely.
"""

from __future__ import annotations

from typing import Any, Sequence
from uuid import UUID

from sqlalchemy import func as sa_func
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.context import current_tenant


class TenantRepository:
    """Generic CRUD repository scoped to the current tenant.

    Usage::

        repo = TenantRepository(session, PersonaModel)
        with tenant_scope(some_tenant_id):
            persona = await repo.create(name="TestBot", ...)
            all_mine = await repo.get_all()
    """

    def __init__(self, session: AsyncSession, model_class: type) -> None:
        self._session = session
        self._model = model_class

    def _base_query(self) -> Any:
        return select(self._model).where(
            self._model.tenant_id == current_tenant()
        )

    async def get(self, id: UUID) -> Any | None:
        result = await self._session.execute(
            self._base_query().where(self._model.id == id)
        )
        return result.scalar_one_or_none()

    async def get_all(self, **filters: Any) -> Sequence[Any]:
        q = self._base_query()
        for key, value in filters.items():
            q = q.where(getattr(self._model, key) == value)
        result = await self._session.execute(q)
        return result.scalars().all()

    async def create(self, **kwargs: Any) -> Any:
        obj = self._model(tenant_id=current_tenant(), **kwargs)
        self._session.add(obj)
        await self._session.flush()
        return obj

    async def update(self, id: UUID, **kwargs: Any) -> Any:
        obj = await self.get(id)
        if obj is None:
            raise ValueError(
                f"{self._model.__name__} {id} not found in tenant "
                f"{current_tenant()}"
            )
        for key, value in kwargs.items():
            setattr(obj, key, value)
        await self._session.flush()
        return obj

    async def delete(self, id: UUID) -> bool:
        obj = await self.get(id)
        if obj is None:
            return False
        await self._session.delete(obj)
        await self._session.flush()
        return True

    async def count(self, **filters: Any) -> int:
        q = (
            select(sa_func.count())
            .select_from(self._model)
            .where(self._model.tenant_id == current_tenant())
        )
        for key, value in filters.items():
            q = q.where(getattr(self._model, key) == value)
        result = await self._session.execute(q)
        return result.scalar_one()
