"""Plugins page — view built-ins, upload + manage tenant-scoped plugins.

Owners (``role='owner'``) see an upload form and an Uninstall button on
every tenant plugin card. Members see the same listing but read-only.
Built-in algorithms are visible to all roles and aren't tenant-scoped —
they're the shared library every tenant inherits.

The upload path goes through ``algorithms.sandbox.installer.install_plugin``,
which validates the file, writes it to ``plugins/uploads/<tenant_id>/``,
loads it inside the tenant's worker subprocess (where it has no access to
the database, broker keys, or other tenants' code), and registers the
adapter into ``AlgorithmRegistry``'s per-tenant overlay.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

from nicegui import events, ui

from ...algorithms.sandbox import SandboxedAlgorithm, get_active_manager
from ...algorithms.sandbox.installer import (
    InstallError,
    install_plugin,
    list_for_tenant,
    uninstall_plugin,
)
from ...auth.session import current_session, current_tenant_id, current_user_id
from ..shell import page_layout


_ALGO_ID = re.compile(r"^[a-z][a-z0-9_]{2,49}$")


def _infer_algo_id(filename: str) -> str:
    """Default algo_id is the filename stem normalized to lowercase."""
    stem = filename.rsplit(".", 1)[0]
    stem = re.sub(r"[^a-z0-9_]", "_", stem.lower())
    stem = re.sub(r"_+", "_", stem).strip("_")
    return stem


@ui.page("/plugins")
async def plugins_page() -> None:
    if not page_layout("Plugins"):
        return

    sess = current_session()
    if sess is None:
        return  # page_layout normally redirects, this is just defensive
    is_owner = sess.role == "owner" or sess.role == "super_admin"
    tenant_id = current_tenant_id()

    ui.label("Algorithm Library").classes("text-h5 q-pb-sm")
    ui.label(
        "Built-in algorithms are available to every tenant. Plugins you "
        "upload below are visible only to your tenant and run in an "
        "isolated subprocess."
    ).classes("text-body2 text-grey-6 q-pb-md")

    with ui.card().classes("w-full bg-orange-9 text-white q-pa-md q-mb-md"):
        with ui.row().classes("items-center gap-2"):
            ui.icon("warning_amber", size="md")
            ui.label("Sandbox notice").classes("text-h6")
        ui.label(
            "Plugins run in a separate process with no access to broker "
            "credentials, the database, or other tenants' code. They can "
            "still execute Python and make outbound network requests — "
            "only upload code from sources you trust."
        ).classes("text-body2")

    builtin_area = ui.column().classes("w-full")
    plugins_area = ui.column().classes("w-full q-pt-md")

    async def _render() -> None:
        from ...algorithms.registry import AlgorithmRegistry

        builtin_area.clear()
        plugins_area.clear()

        # Built-ins (global, shared)
        with builtin_area:
            with ui.row().classes("items-center gap-2 q-pt-sm q-pb-xs"):
                ui.icon("auto_awesome", color="primary", size="sm")
                ui.label("Built-in algorithms").classes("text-subtitle1")
            ui.label(
                f"{len(AlgorithmRegistry._algorithms)} algorithm(s) "
                "available to every tenant."
            ).classes("text-caption text-grey-6 q-pb-sm")
            with ui.row().classes("w-full gap-3 flex-wrap"):
                for algo_id, algo in sorted(AlgorithmRegistry._algorithms.items()):
                    _render_algo_card(algo_id, algo, removable=False)

        # Tenant overlay
        installed = await list_for_tenant(tenant_id) if tenant_id else []
        with plugins_area:
            with ui.row().classes("items-center gap-2 q-pt-md q-pb-xs"):
                ui.icon("extension", color="accent", size="sm")
                ui.label("Your plugins").classes("text-subtitle1")
            if not installed:
                ui.label(
                    "No plugins uploaded yet. "
                    + ("Use the upload form below." if is_owner else
                       "Ask your tenant owner to upload one.")
                ).classes("text-caption text-grey-6 q-pb-sm")
            else:
                ui.label(
                    f"{len(installed)} plugin(s) installed for this tenant."
                ).classes("text-caption text-grey-6 q-pb-sm")

            with ui.row().classes("w-full gap-3 flex-wrap"):
                for plug in installed:
                    _render_plugin_card(plug, is_owner)

    def _render_algo_card(algo_id: str, algo: Any, *, removable: bool) -> None:
        m = algo.manifest
        with ui.card().classes("w-72"):
            with ui.row().classes("w-full items-center justify-between"):
                ui.label(m.name).classes("text-h6")
                ui.badge("Built-in", color="primary")
            if m.description:
                ui.label(m.description[:140]).classes(
                    "text-body2 text-grey-5 q-py-xs"
                )
            with ui.row().classes("gap-1 q-pb-xs flex-wrap"):
                for ac in m.asset_classes:
                    ui.chip(ac, color="teal").props("dense outline")
            ui.label(f"id: {m.id} · v{m.version}").classes(
                "text-caption text-grey-7"
            )

    def _render_plugin_card(plug: Any, owner: bool) -> None:
        with ui.card().classes("w-72"):
            with ui.row().classes("w-full items-center justify-between"):
                ui.label(plug.name).classes("text-h6")
                if plug.is_enabled:
                    ui.badge("Sandbox", color="accent")
                else:
                    ui.badge("Disabled", color="grey")
            ui.label(f"file: {plug.filename}").classes(
                "text-caption text-grey-7"
            )
            ui.label(f"sha256: {plug.sha256[:12]}…").classes(
                "text-caption text-grey-7"
            )
            ui.label(f"id: {plug.algorithm_id}").classes(
                "text-caption text-grey-7"
            )
            if plug.last_error:
                ui.label(f"⚠ {plug.last_error[:80]}").classes(
                    "text-caption text-negative"
                )
            if owner:
                async def _do_uninstall(p_id: str = plug.algorithm_id) -> None:
                    if tenant_id is None:
                        return
                    try:
                        ok = await uninstall_plugin(
                            manager=get_active_manager(),
                            tenant_id=tenant_id,
                            algorithm_id=p_id,
                        )
                    except Exception as exc:  # noqa: BLE001
                        ui.notify(f"Uninstall failed: {exc}", type="negative")
                        return
                    if ok:
                        ui.notify(f"Uninstalled {p_id}", type="positive")
                    else:
                        ui.notify(f"Plugin {p_id} not found", type="warning")
                    await _render()

                ui.button(
                    "Uninstall",
                    icon="delete_outline",
                    on_click=_do_uninstall,
                ).props("flat dense color=negative").classes("q-mt-xs")

    # ---- upload (owner only) -----------------------------------------------
    if is_owner and tenant_id is not None:
        ui.separator().classes("q-my-md")
        ui.label("Upload single-file plugin (.py)").classes(
            "text-subtitle1 q-pb-xs"
        )
        ui.label(
            "Plugin must define one ``Algorithm`` subclass. The "
            "``manifest.id`` must be globally unique and use only "
            "lowercase letters, digits, and underscores."
        ).classes("text-caption text-grey-6 q-pb-sm")

        upload_status = ui.row().classes("w-full q-pt-xs")

        async def _handle_upload(e: events.UploadEventArguments) -> None:
            name = e.name or ""
            payload = e.content.read()
            algo_id = _infer_algo_id(name)
            if not _ALGO_ID.match(algo_id):
                ui.notify(
                    f"Could not derive a valid algorithm id from {name!r}. "
                    "Rename the file to letters/digits/underscores only.",
                    type="negative",
                )
                return

            upload_status.clear()
            try:
                result = await install_plugin(
                    manager=get_active_manager(),
                    tenant_id=tenant_id,
                    user_id=current_user_id(),
                    filename=name,
                    algorithm_id=algo_id,
                    payload=payload,
                )
            except InstallError as exc:
                with upload_status:
                    ui.icon("error", color="negative")
                    ui.label(str(exc)).classes("text-negative")
                return

            with upload_status:
                ui.icon("check_circle", color="positive")
                ui.label(
                    f"Installed {result.name} (id={result.algorithm_id}, "
                    f"warmup={result.warmup_bars})"
                ).classes("text-positive")
            await _render()

        def _bridge(e: events.UploadEventArguments) -> None:
            asyncio.create_task(_handle_upload(e))

        ui.upload(
            on_upload=_bridge, auto_upload=True, max_files=1,
        ).props("accept=.py color=primary").classes("w-full")
    elif not is_owner:
        ui.separator().classes("q-my-md")
        ui.label(
            "Plugin uploads are restricted to tenant owners. "
            "Ask your owner to upload a plugin on your behalf."
        ).classes("text-caption text-grey-6")

    await _render()
