"""Admin Tenants page — super admins manage tenants and pause workers."""

from __future__ import annotations

from uuid import UUID

from nicegui import app, ui
from sqlalchemy import func, select, update

from ...auth.roles import ROLE_SUPER_ADMIN
from ...storage.database import get_session
from ...storage.models import PersonaModel, TenantModel, UserModel
from ..middleware import ensure_role
from ..shell import page_layout


@ui.page("/admin/tenants")
async def admin_tenants_page() -> None:
    if not ensure_role(ROLE_SUPER_ADMIN):
        return
    if not page_layout("Admin · Tenants"):
        return

    ui.label("Tenants").classes("text-h5 q-mb-md")

    container = ui.column().classes("w-full")

    async def refresh() -> None:
        rows = await _load_tenants()
        worker_counts = _supervisor_worker_counts()
        container.clear()
        with container:
            with ui.row().classes(
                "w-full text-caption text-grey-6 q-px-sm q-py-xs"
            ):
                ui.label("Name").classes("col-3")
                ui.label("Users").classes("col-1")
                ui.label("Active personas").classes("col-2")
                ui.label("Workers").classes("col-2")
                ui.label("Status").classes("col-2")
                ui.label("").classes("col-2")
            ui.separator()
            for row in rows:
                with ui.row().classes("w-full items-center q-px-sm q-py-xs"):
                    ui.label(row["name"]).classes("col-3")
                    ui.label(str(row["user_count"])).classes("col-1")
                    ui.label(str(row["active_personas"])).classes("col-2")
                    running = sum(
                        1 for c in worker_counts.values() if row["id"] in c
                    )
                    ui.label(f"{running} running").classes("col-2 text-caption")
                    ui.label(
                        "enabled" if row["enabled"] else "paused"
                    ).classes(
                        "col-2 text-"
                        + ("positive" if row["enabled"] else "warning")
                    )
                    with ui.row().classes("col-2 gap-1"):
                        await _toggle_button(row, refresh)

    await refresh()


async def _load_tenants() -> list[dict]:
    async with get_session() as session:
        users_count = (
            select(UserModel.tenant_id, func.count().label("c"))
            .group_by(UserModel.tenant_id)
            .subquery()
        )
        personas_count = (
            select(PersonaModel.tenant_id, func.count().label("c"))
            .where(PersonaModel.mode.in_(("paper", "live")))
            .group_by(PersonaModel.tenant_id)
            .subquery()
        )
        stmt = (
            select(
                TenantModel.id,
                TenantModel.name,
                TenantModel.background_workers_enabled,
                func.coalesce(users_count.c.c, 0),
                func.coalesce(personas_count.c.c, 0),
            )
            .outerjoin(users_count, users_count.c.tenant_id == TenantModel.id)
            .outerjoin(personas_count, personas_count.c.tenant_id == TenantModel.id)
            .order_by(TenantModel.created_at)
        )
        rows = (await session.execute(stmt)).all()
        return [
            {
                "id": tid,
                "name": name,
                "enabled": bool(enabled),
                "user_count": int(uc),
                "active_personas": int(pc),
            }
            for tid, name, enabled, uc, pc in rows
        ]


def _supervisor_worker_counts() -> dict[str, set[UUID]]:
    """Snapshot which supervisors currently have a worker per tenant."""
    manager = getattr(app.state, "supervisor_manager", None)
    if manager is None:
        return {}
    return {s.name: set(s.workers.keys()) for s in manager._supervisors}  # noqa: SLF001


async def _toggle_button(row: dict, refresh) -> None:
    label = "Pause workers" if row["enabled"] else "Resume workers"

    async def _toggle() -> None:
        async with get_session() as session:
            await session.execute(
                update(TenantModel)
                .where(TenantModel.id == row["id"])
                .values(background_workers_enabled=not row["enabled"])
            )
            await session.commit()
        ui.notify(
            f"{row['name']}: workers {'paused' if row['enabled'] else 'resumed'}",
            type="positive",
        )
        # Kick supervisors so the change takes effect within seconds rather
        # than waiting for the next 60s poll.
        manager = getattr(app.state, "supervisor_manager", None)
        if manager is not None:
            for s in manager._supervisors:  # noqa: SLF001
                try:
                    await s._refresh()  # noqa: SLF001
                except Exception:
                    pass
        await refresh()

    ui.button(label, on_click=_toggle).props(
        "flat dense color=" + ("warning" if row["enabled"] else "positive")
    )
