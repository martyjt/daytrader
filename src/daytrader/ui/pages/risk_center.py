"""Risk Center — 'Am I safe right now?'

Global kill switch, per-persona risk indicators, drawdown tracking.
Phase 0: kill switch + placeholder risk metrics.
"""

from __future__ import annotations

from nicegui import ui

from ..services import list_personas
from ..shell import page_layout, stat_card


@ui.page("/risk")
async def risk_center_page() -> None:
    page_layout("Risk Center")

    # ---- Kill switch banner ----------------------------------------------
    with ui.card().classes("w-full").style("background-color: #2d1111"):
        with ui.row().classes("w-full items-center justify-between"):
            with ui.column().classes("gap-0"):
                ui.label("Emergency Kill Switch").classes(
                    "text-h6 text-white"
                )
                ui.label(
                    "Immediately halt all live and paper trading across "
                    "every persona."
                ).classes("text-body2 text-grey-5")

            async def _on_kill() -> None:
                from ..services import kill_all_trading

                try:
                    count = await kill_all_trading(reason="manual")
                    ui.notify(
                        f"KILL SWITCH ACTIVATED — {count} persona(s) halted",
                        type="warning",
                        position="top",
                        timeout=5000,
                    )
                    ui.navigate.to("/risk")  # Refresh the page
                except RuntimeError:
                    ui.notify(
                        "Kill switch not ready",
                        type="negative",
                        position="top",
                    )

            ui.button("KILL ALL", color="negative", on_click=_on_kill).props(
                "size=lg unelevated"
            )

    # ---- Global risk metrics (placeholder) --------------------------------
    ui.label("Global Risk Metrics").classes("text-h6 q-pt-lg q-pb-sm")

    with ui.row().classes("w-full gap-4"):
        stat_card(
            "Max Drawdown", "0.0%", icon="trending_down", color="positive"
        )
        stat_card("Open Positions", "0", icon="swap_vert")
        stat_card("Daily P&L", "$0.00", icon="today")
        stat_card("Data Staleness", "OK", icon="wifi", color="positive")

    # ---- Per-persona risk status ------------------------------------------
    ui.separator().classes("q-my-md")
    ui.label("Per-Persona Risk Status").classes("text-h6 q-pb-sm")

    personas = await list_personas()
    if not personas:
        ui.label(
            "No personas running. Create one and start paper trading "
            "to see risk indicators here."
        ).classes("text-body2 text-grey-6")
    else:
        columns = [
            {"name": "name", "label": "Persona", "field": "name"},
            {"name": "mode", "label": "Mode", "field": "mode"},
            {"name": "equity", "label": "Equity", "field": "equity"},
            {"name": "risk", "label": "Risk Status", "field": "risk"},
        ]
        rows = [
            {
                "name": p.name,
                "mode": p.mode,
                "equity": f"${float(p.current_equity):,.2f}",
                "risk": "OK",
            }
            for p in personas
        ]
        ui.aggrid(
            {
                "columnDefs": columns,
                "rowData": rows,
                "domLayout": "autoHeight",
            }
        ).classes("w-full")

    # ---- Risk rules reference ---------------------------------------------
    ui.separator().classes("q-my-md")
    ui.label("Risk Rules (from config/default.yaml)").classes(
        "text-h6 q-pb-sm"
    )
    with ui.card().classes("w-full q-pa-md"):
        rules = [
            ("Per-trade stop-loss", "2.0× ATR"),
            ("Per-trade take-profit", "4.0× ATR"),
            ("Max hold duration", "500 bars"),
            ("Daily loss limit", "5.0% per persona"),
            ("Max open positions", "10 per persona"),
            ("Global max drawdown", "20.0%"),
            ("Data staleness alert", "120 seconds"),
        ]
        for rule, val in rules:
            with ui.row().classes("items-center justify-between w-full"):
                ui.label(rule).classes("text-body2")
                ui.badge(val, color="primary").props("outline")
