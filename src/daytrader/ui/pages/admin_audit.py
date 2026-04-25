"""Admin Audit page — super admins inspect the audit log (Phase 7).

Cross-tenant by default since super-admins operate above the tenant
boundary; the tenant filter narrows the view to a single tenant when
diagnosing a specific incident. Action and resource filters help
isolate "all logins this hour" or "every plugin install ever".
"""

from __future__ import annotations

import json
from typing import Any
from uuid import UUID

from nicegui import ui
from sqlalchemy import select

from ...auth.roles import ROLE_SUPER_ADMIN
from ...storage.database import get_session
from ...storage.models import AuditLogModel, TenantModel, UserModel
from ..middleware import ensure_role
from ..shell import page_layout


_PAGE_LIMIT = 100


@ui.page("/admin/audit")
async def admin_audit_page() -> None:
    if not ensure_role(ROLE_SUPER_ADMIN):
        return
    if not page_layout("Admin · Audit"):
        return

    ui.label("Audit log").classes("text-h5 q-mb-sm")
    ui.label(
        f"Showing the last {_PAGE_LIMIT} events. Use the filters to narrow "
        "the view."
    ).classes("text-caption text-grey-6 q-mb-md")

    # Filter state lives in plain refs so the closures can mutate without
    # ceremony. NiceGUI rebuilds the table every time refresh() runs.
    state: dict[str, Any] = {
        "tenant_id": None,
        "action": None,
        "resource_type": None,
    }

    tenants = await _load_tenants()
    actions, resource_types = await _load_filter_options()

    tenant_options = {None: "All tenants", **{t["id"]: t["name"] for t in tenants}}
    action_options = {None: "All actions", **{a: a for a in actions}}
    rt_options = {None: "All resource types", **{r: r for r in resource_types}}

    table_container = ui.column().classes("w-full")

    async def refresh() -> None:
        rows = await _load_events(
            tenant_id=state["tenant_id"],
            action=state["action"],
            resource_type=state["resource_type"],
        )
        table_container.clear()
        with table_container:
            with ui.row().classes(
                "w-full text-caption text-grey-6 q-px-sm q-py-xs"
            ):
                ui.label("Time").classes("col-2")
                ui.label("Tenant").classes("col-2")
                ui.label("User").classes("col-2")
                ui.label("Action").classes("col-2")
                ui.label("Resource").classes("col-2")
                ui.label("Extra").classes("col-2")
            ui.separator()
            if not rows:
                ui.label("No matching events.").classes(
                    "text-caption text-grey-6 q-pa-md"
                )
                return
            for r in rows:
                with ui.row().classes(
                    "w-full items-start q-px-sm q-py-xs"
                ).style("border-bottom: 1px solid #2a2b3e"):
                    ui.label(_fmt_ts(r["created_at"])).classes(
                        "col-2 text-caption"
                    )
                    ui.label(r["tenant_name"] or "—").classes("col-2 text-caption")
                    ui.label(r["user_email"] or "—").classes("col-2 text-caption")
                    ui.label(r["action"]).classes(
                        "col-2 text-caption text-mono"
                    )
                    ui.label(_fmt_resource(r)).classes(
                        "col-2 text-caption text-mono"
                    )
                    ui.label(_fmt_extra(r["extra"])).classes(
                        "col-2 text-caption text-mono"
                    ).style("white-space: pre-wrap; word-break: break-all")

    with ui.row().classes("w-full items-center q-mb-md gap-2"):
        tenant_select = ui.select(
            tenant_options, value=None, label="Tenant"
        ).props("outlined dense").classes("w-48")
        action_select = ui.select(
            action_options, value=None, label="Action"
        ).props("outlined dense").classes("w-48")
        rt_select = ui.select(
            rt_options, value=None, label="Resource type"
        ).props("outlined dense").classes("w-48")

        async def _on_tenant(e) -> None:
            state["tenant_id"] = e.value
            await refresh()

        async def _on_action(e) -> None:
            state["action"] = e.value
            await refresh()

        async def _on_rt(e) -> None:
            state["resource_type"] = e.value
            await refresh()

        tenant_select.on("update:model-value", _on_tenant)
        action_select.on("update:model-value", _on_action)
        rt_select.on("update:model-value", _on_rt)

        ui.button("Refresh", icon="refresh", on_click=refresh).props(
            "flat dense color=primary"
        )

    await refresh()


async def _load_tenants() -> list[dict[str, Any]]:
    async with get_session() as session:
        rows = (
            await session.execute(
                select(TenantModel.id, TenantModel.name).order_by(TenantModel.name)
            )
        ).all()
    return [{"id": tid, "name": name} for tid, name in rows]


async def _load_filter_options() -> tuple[list[str], list[str]]:
    async with get_session() as session:
        actions = (
            await session.execute(
                select(AuditLogModel.action).distinct().order_by(AuditLogModel.action)
            )
        ).scalars().all()
        rts = (
            await session.execute(
                select(AuditLogModel.resource_type)
                .where(AuditLogModel.resource_type.is_not(None))
                .distinct()
                .order_by(AuditLogModel.resource_type)
            )
        ).scalars().all()
    return list(actions), list(rts)


async def _load_events(
    *,
    tenant_id: UUID | None,
    action: str | None,
    resource_type: str | None,
) -> list[dict[str, Any]]:
    async with get_session() as session:
        stmt = (
            select(
                AuditLogModel.created_at,
                AuditLogModel.action,
                AuditLogModel.resource_type,
                AuditLogModel.resource_id,
                AuditLogModel.extra,
                AuditLogModel.tenant_id,
                AuditLogModel.user_id,
                TenantModel.name,
                UserModel.email,
            )
            .outerjoin(TenantModel, TenantModel.id == AuditLogModel.tenant_id)
            .outerjoin(UserModel, UserModel.id == AuditLogModel.user_id)
            .order_by(AuditLogModel.created_at.desc())
            .limit(_PAGE_LIMIT)
        )
        if tenant_id is not None:
            stmt = stmt.where(AuditLogModel.tenant_id == tenant_id)
        if action:
            stmt = stmt.where(AuditLogModel.action == action)
        if resource_type:
            stmt = stmt.where(AuditLogModel.resource_type == resource_type)

        result = (await session.execute(stmt)).all()
    return [
        {
            "created_at": r[0],
            "action": r[1],
            "resource_type": r[2],
            "resource_id": r[3],
            "extra": r[4] or {},
            "tenant_id": r[5],
            "user_id": r[6],
            "tenant_name": r[7],
            "user_email": r[8],
        }
        for r in result
    ]


def _fmt_ts(ts: Any) -> str:
    try:
        return ts.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:  # noqa: BLE001
        return str(ts)


def _fmt_resource(row: dict[str, Any]) -> str:
    rt = row.get("resource_type") or ""
    rid = row.get("resource_id") or ""
    if not rt and not rid:
        return "—"
    if rid and len(rid) > 14:
        rid = rid[:8] + "…"
    return f"{rt}:{rid}" if rt else rid


def _fmt_extra(extra: dict[str, Any] | str | None) -> str:
    if not extra:
        return ""
    if isinstance(extra, str):
        # Some adapters may surface raw JSON strings
        try:
            extra = json.loads(extra)
        except ValueError:
            return extra
    return ", ".join(f"{k}={v}" for k, v in sorted(extra.items()))
