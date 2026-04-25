"""Admin Users page — list users, send invites, activate/deactivate.

Super-admin only. Cross-tenant view (a super admin can manage everyone).
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from nicegui import ui

from ...auth.invites import create_invite, list_pending_invites
from ...auth.roles import ROLE_MEMBER, ROLE_OWNER, ROLE_SUPER_ADMIN, ROLES
from ...auth.service import AuthError, list_users, set_active
from ...auth.session import current_user_id
from ...storage.database import get_session
from ...storage.models import TenantModel
from sqlalchemy import select

from ..middleware import ensure_role
from ..shell import page_layout


@ui.page("/admin/users")
async def admin_users_page() -> None:
    if not ensure_role(ROLE_SUPER_ADMIN):
        return
    if not page_layout("Admin · Users"):
        return

    with ui.row().classes("w-full items-center justify-between q-mb-md"):
        ui.label("User accounts").classes("text-h5")
        with ui.row().classes("gap-2"):
            invite_btn = ui.button("New invite", icon="mail").props("color=primary")

    users_container = ui.column().classes("w-full")
    invites_container = ui.column().classes("w-full q-mt-lg")

    async def refresh() -> None:
        users_container.clear()
        with users_container:
            ui.label("Users").classes("text-subtitle1 text-weight-medium")
            users = await list_users()
            tenants = await _tenants_by_id()
            with ui.element("div").classes("w-full"):
                with ui.row().classes(
                    "w-full text-caption text-grey-6 q-px-sm q-py-xs"
                ):
                    ui.label("Email").classes("col-3")
                    ui.label("Role").classes("col-2")
                    ui.label("Tenant").classes("col-2")
                    ui.label("Active").classes("col-1")
                    ui.label("Last login").classes("col-2")
                    ui.label("Actions").classes("col-2")
                ui.separator()
                for u in users:
                    tenant_name = tenants.get(u.tenant_id, str(u.tenant_id)[:8])
                    with ui.row().classes("w-full items-center q-px-sm q-py-xs"):
                        ui.label(u.email).classes("col-3")
                        ui.label(u.role).classes("col-2")
                        ui.label(tenant_name).classes("col-2")
                        ui.icon(
                            "check_circle" if u.is_active else "cancel",
                            color="positive" if u.is_active else "negative",
                        ).classes("col-1")
                        ui.label(
                            u.last_login_at.strftime("%Y-%m-%d %H:%M")
                            if u.last_login_at
                            else "—"
                        ).classes("col-2 text-caption")
                        with ui.row().classes("col-2 gap-1"):
                            await _row_actions(u, refresh)

        invites_container.clear()
        with invites_container:
            ui.label("Pending invites").classes("text-subtitle1 text-weight-medium")
            invites = await list_pending_invites()
            if not invites:
                ui.label("No pending invites").classes("text-caption text-grey-6")
            else:
                for inv in invites:
                    with ui.card().classes("w-full q-pa-sm q-mb-xs"):
                        with ui.row().classes("items-center gap-3"):
                            ui.icon("mail").classes("text-primary")
                            ui.label(inv.email).classes("text-body2")
                            ui.badge(inv.role, color="grey-8").props("outline")
                            ui.space()
                            ui.label(
                                f"expires {inv.expires_at:%Y-%m-%d %H:%M}"
                            ).classes("text-caption text-grey-6")
                            invite_url = _invite_url(inv.token)
                            ui.button(
                                "Copy link",
                                on_click=lambda url=invite_url: ui.run_javascript(
                                    f"navigator.clipboard.writeText({url!r}); "
                                    f"window._copied=true"
                                ) and ui.notify("Invite link copied"),
                            ).props("flat dense")

    async def _open_invite_dialog() -> None:
        dlg = ui.dialog()
        with dlg, ui.card().classes("min-w-[420px]"):
            ui.label("Send invite").classes("text-h6")
            email = ui.input("Email").props("outlined dense").classes("w-full")
            role = ui.select(
                list(ROLES),
                value=ROLE_MEMBER,
                label="Role",
            ).props("outlined dense").classes("w-full")

            tenants_map = await _tenants_by_id()
            tenant_options = {"__new__": "(Create new tenant)"} | {
                str(tid): name for tid, name in tenants_map.items()
            }
            tenant_sel = ui.select(
                tenant_options,
                value="__new__",
                label="Tenant",
            ).props("outlined dense").classes("w-full")

            error = ui.label("").classes("text-negative text-caption")

            async def submit() -> None:
                if not email.value or "@" not in email.value:
                    error.text = "Enter a valid email"
                    return
                if role.value == ROLE_SUPER_ADMIN and tenant_sel.value == "__new__":
                    error.text = "Super admin must be attached to an existing tenant"
                    return
                tenant_id = (
                    None if tenant_sel.value == "__new__"
                    else UUID(tenant_sel.value)
                )
                inviter = current_user_id()
                if inviter is None:
                    error.text = "Session expired"
                    return
                try:
                    token = await create_invite(
                        email=email.value,
                        role=role.value,
                        tenant_id=tenant_id,
                        invited_by=inviter,
                    )
                except (AuthError, ValueError) as e:
                    error.text = str(e)
                    return
                dlg.close()
                ui.notify(f"Invite created: {_invite_url(token)}", type="positive")
                await refresh()

            with ui.row().classes("w-full justify-end gap-2 q-mt-md"):
                ui.button("Cancel", on_click=dlg.close).props("flat")
                ui.button("Create invite", on_click=submit).props("color=primary")
        dlg.open()

    invite_btn.on_click(_open_invite_dialog)
    await refresh()


async def _row_actions(user, refresh) -> None:
    me = current_user_id()
    is_self = me is not None and user.id == me

    async def _toggle() -> None:
        if is_self:
            ui.notify("Cannot deactivate your own account", type="warning")
            return
        await set_active(user.id, not user.is_active)
        await refresh()

    label = "Deactivate" if user.is_active else "Reactivate"
    ui.button(label, on_click=_toggle).props(
        "flat dense color=" + ("negative" if user.is_active else "positive")
    )


async def _tenants_by_id() -> dict[UUID, str]:
    async with get_session() as session:
        rows = (await session.execute(select(TenantModel))).scalars().all()
        return {t.id: t.name for t in rows}


def _invite_url(token: str) -> str:
    return f"/invite/{token}"
