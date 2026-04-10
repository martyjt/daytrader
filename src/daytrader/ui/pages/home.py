"""Home page — 'What's happening today?'

Shows summary stats across all personas, quick-view persona cards,
and a call-to-action if no personas exist yet.
"""

from __future__ import annotations

from nicegui import ui

from ..services import count_personas, list_personas
from ..shell import page_layout, persona_card, stat_card


@ui.page("/")
async def home_page() -> None:
    page_layout("What's happening today?")

    personas = await list_personas()
    total_equity = sum(float(p.current_equity or 0) for p in personas)
    active = sum(1 for p in personas if p.mode in ("paper", "live"))

    # ---- Summary stats ---------------------------------------------------
    with ui.row().classes("w-full gap-4 q-pb-md"):
        stat_card("Personas", len(personas), icon="smart_toy")
        stat_card(
            "Total Equity",
            f"${total_equity:,.2f}",
            icon="account_balance",
        )
        stat_card("Active", active, icon="play_circle", color="positive")

    ui.separator()

    # ---- Persona cards or empty state ------------------------------------
    if personas:
        ui.label("Your Personas").classes("text-h6 q-pt-md q-pb-sm")
        with ui.row().classes("w-full gap-4 flex-wrap"):
            for p in personas:
                persona_card(p)
    else:
        with ui.column().classes("w-full items-center q-pa-xl"):
            ui.icon("rocket_launch", size="64px").classes("text-grey-6")
            ui.label("No personas yet").classes("text-h6 text-grey-6 q-pb-sm")
            ui.label(
                "Create a persona to start testing algorithms"
            ).classes("text-body2 text-grey-7")
            ui.button(
                "Create your first persona",
                icon="add",
                on_click=lambda: ui.navigate.to("/personas"),
            ).props("color=primary unelevated").classes("q-mt-md")
