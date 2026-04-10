"""Personas page — create, manage, and compare bot identities.

This is the user's primary entry point. Every persona is an isolated
bot with its own portfolio, strategy binding, risk profile, and mode
(backtest / paper / live).
"""

from __future__ import annotations

from decimal import Decimal

from nicegui import ui

from ..services import create_persona, delete_persona, list_personas
from ..shell import page_layout, persona_card


@ui.page("/personas")
async def personas_page() -> None:
    page_layout("Personas")

    container = ui.row().classes("w-full gap-4 flex-wrap")

    async def refresh() -> None:
        container.clear()
        personas = await list_personas()
        with container:
            if not personas:
                with ui.column().classes("w-full items-center q-pa-xl"):
                    ui.icon("person_add", size="64px").classes("text-grey-6")
                    ui.label("No personas yet").classes(
                        "text-h6 text-grey-6 q-pb-xs"
                    )
                    ui.label("Click the button above to create one.").classes(
                        "text-body2 text-grey-7"
                    )
            else:
                for p in personas:
                    persona_card(p, on_delete=refresh)

    async def open_create_dialog() -> None:
        with ui.dialog() as dlg, ui.card().classes("w-96"):
            ui.label("Create Persona").classes("text-h6 q-pb-sm")

            name = ui.input(
                "Name", placeholder="e.g. BTC Paper Tester", validation={
                    "Required": lambda v: bool(v and v.strip()),
                }
            )
            asset_class = ui.select(
                ["crypto", "equities", "forex", "commodities"],
                value="crypto",
                label="Asset Class",
            )
            base_currency = ui.select(
                ["USDT", "USD", "EUR", "BTC"],
                value="USDT",
                label="Base Currency",
            )
            capital = ui.number(
                "Initial Capital ($)", value=10000, min=1, format="%.2f"
            )
            risk = ui.select(
                ["conservative", "balanced", "aggressive"],
                value="balanced",
                label="Risk Profile",
            )

            ui.separator().classes("q-my-sm")

            with ui.row().classes("w-full justify-end gap-2"):
                ui.button("Cancel", on_click=dlg.close).props("flat")

                async def _do_create() -> None:
                    if not name.value or not name.value.strip():
                        ui.notify("Name is required", type="negative")
                        return
                    await create_persona(
                        name=name.value.strip(),
                        asset_class=asset_class.value,
                        base_currency=base_currency.value,
                        initial_capital=Decimal(str(capital.value)),
                        risk_profile=risk.value,
                    )
                    dlg.close()
                    ui.notify(
                        f"Persona '{name.value}' created",
                        type="positive",
                    )
                    await refresh()

                ui.button("Create", on_click=_do_create).props(
                    "color=primary unelevated"
                )

        dlg.open()

    # ---- Header row + create button --------------------------------------
    with ui.row().classes("w-full justify-between items-center q-pb-md"):
        ui.label("Your Personas").classes("text-h5")
        ui.button(
            "+ New Persona", icon="add", on_click=open_create_dialog
        ).props("color=primary unelevated")

    await refresh()
