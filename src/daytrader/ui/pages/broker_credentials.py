"""Broker Credentials page — paste / test / rotate / delete API keys.

Per-tenant: every operator only sees their own tenant's keys (the service
layer scopes by ``current_tenant_id()``).

Adding or replacing a credential invalidates the cached executor for that
``(tenant, broker)`` pair, so the trading loop picks up the new keys on its
next poll without restarting the process.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any
from uuid import UUID

from nicegui import ui

from ...auth.session import current_tenant_id
from ...core.settings import get_settings
from ...execution.credentials import (
    SUPPORTED_BROKERS,
    CredentialError,
    delete_credential,
    list_credentials,
    save_credential,
    test_connection,
)
from ..shell import page_layout

_RefreshFn = Callable[[], Awaitable[None]]


@ui.page("/broker-credentials")
async def broker_credentials_page() -> None:
    if not page_layout("Broker Credentials"):
        return

    tenant_id = current_tenant_id()
    if tenant_id is None:
        ui.label("No active session").classes("text-negative")
        return

    if not get_settings().app_encryption_key.get_secret_value():
        with ui.card().classes("w-full q-pa-md").style(
            "background-color: #2d1111; border-left: 3px solid var(--q-negative)"
        ):
            ui.label("APP_ENCRYPTION_KEY is not set").classes(
                "text-h6 text-white"
            )
            ui.label(
                "You can browse this page, but adding or testing credentials "
                "will fail. Generate a key and set APP_ENCRYPTION_KEY in .env, "
                "then restart the app:"
            ).classes("text-body2 text-grey-4")
            ui.label(
                'python -c "from cryptography.fernet import Fernet; '
                'print(Fernet.generate_key().decode())"'
            ).classes("text-caption text-grey-5 q-mt-sm").style(
                "font-family: monospace"
            )

    with ui.row().classes("w-full items-center justify-between q-mb-md"):
        ui.label("Connected brokers").classes("text-h5")
        with ui.row().classes("gap-2"):
            add_btn = ui.button("Add / replace", icon="vpn_key").props(
                "color=primary"
            )

    list_container = ui.column().classes("w-full")

    async def refresh() -> None:
        list_container.clear()
        creds = await list_credentials(tenant_id)
        with list_container:
            if not creds:
                ui.label(
                    "No broker credentials saved yet. Click 'Add / replace' "
                    "to paste your Binance or Alpaca API keys."
                ).classes("text-caption text-grey-6")
                return

            with ui.row().classes(
                "w-full text-caption text-grey-6 q-px-sm q-py-xs"
            ):
                ui.label("Broker").classes("col-2")
                ui.label("API key").classes("col-3")
                ui.label("Mode").classes("col-2")
                ui.label("Added").classes("col-2")
                ui.label("Actions").classes("col-3")
            ui.separator()
            for c in creds:
                with ui.row().classes("w-full items-center q-px-sm q-py-xs"):
                    ui.label(c.broker_name).classes("col-2 text-weight-medium")
                    ui.label(c.api_key_masked).classes(
                        "col-3"
                    ).style("font-family: monospace")
                    ui.badge(
                        "testnet" if c.is_testnet else "live",
                        color="warning" if c.is_testnet else "negative",
                    ).classes("col-2")
                    ui.label(
                        c.created_at.strftime("%Y-%m-%d %H:%M")
                        if c.created_at
                        else "—"
                    ).classes("col-2 text-caption")
                    with ui.row().classes("col-3 gap-1"):
                        await _row_actions(tenant_id, c, refresh)

    add_btn.on_click(lambda: _open_add_dialog(tenant_id, refresh))
    await refresh()


async def _row_actions(
    tenant_id: UUID, credential: Any, refresh: _RefreshFn
) -> None:
    async def _delete() -> None:
        ok = await delete_credential(
            tenant_id=tenant_id, credential_id=credential.id
        )
        if ok:
            ui.notify(
                f"{credential.broker_name} credentials removed",
                type="positive",
            )
            await refresh()
        else:
            ui.notify("Credential not found", type="warning")

    ui.button("Delete", on_click=_delete).props("flat dense color=negative")


async def _open_add_dialog(tenant_id: UUID, refresh: _RefreshFn) -> None:
    dlg = ui.dialog()
    with dlg, ui.card().classes("min-w-[480px]"):
        ui.label("Add or replace broker credentials").classes("text-h6")
        ui.label(
            "Adding credentials for a broker that already has saved keys "
            "will replace them."
        ).classes("text-caption text-grey-6 q-mb-sm")

        broker = (
            ui.select(list(SUPPORTED_BROKERS), value="binance", label="Broker")
            .props("outlined dense")
            .classes("w-full")
        )
        api_key = (
            ui.input("API key").props("outlined dense").classes("w-full")
        )
        api_secret = (
            ui.input("API secret", password=True, password_toggle_button=True)
            .props("outlined dense")
            .classes("w-full")
        )
        testnet = ui.checkbox("Testnet / paper environment", value=True)

        status = ui.label("").classes("text-caption q-mt-sm")

        async def _do_test() -> None:
            try:
                balance = await test_connection(
                    broker_name=broker.value,
                    fields={
                        "api_key": api_key.value,
                        "api_secret": api_secret.value,
                    },
                    is_testnet=testnet.value,
                )
            except CredentialError as exc:
                status.text = f"✗ {exc}"
                status.classes(replace="text-caption text-negative")
                return
            except Exception as exc:
                status.text = f"✗ Connection failed: {exc}"
                status.classes(replace="text-caption text-negative")
                return
            status.text = f"✓ Connected — balance ${float(balance):,.2f}"
            status.classes(replace="text-caption text-positive")

        async def _do_save() -> None:
            try:
                await save_credential(
                    tenant_id=tenant_id,
                    broker_name=broker.value,
                    fields={
                        "api_key": api_key.value,
                        "api_secret": api_secret.value,
                    },
                    is_testnet=testnet.value,
                )
            except CredentialError as exc:
                status.text = f"✗ {exc}"
                status.classes(replace="text-caption text-negative")
                return
            except RuntimeError as exc:
                status.text = f"✗ {exc}"
                status.classes(replace="text-caption text-negative")
                return
            dlg.close()
            ui.notify(f"{broker.value} credentials saved", type="positive")
            await refresh()

        with ui.row().classes("w-full justify-end gap-2 q-mt-md"):
            ui.button("Cancel", on_click=dlg.close).props("flat")
            ui.button("Test connection", on_click=_do_test).props(
                "flat color=secondary"
            )
            ui.button("Save", on_click=_do_save).props("color=primary")
    dlg.open()
