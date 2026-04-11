"""Journal page — activity log for all trading events.

Displays signals, order fills, risk breaches, kill-switch activations,
and mode changes across all personas.
"""

from __future__ import annotations

from nicegui import ui

from ..services import list_journal_entries, list_personas
from ..shell import page_layout


@ui.page("/journal")
async def journal_page() -> None:
    page_layout("Journal")

    # ---- Filters ------------------------------------------------------------
    ui.label("Activity Journal").classes("text-h5 q-pb-sm")

    personas = await list_personas()
    persona_options = {"": "All Personas"} | {
        str(p.id): p.name for p in personas
    }
    event_options = {
        "": "All Events",
        "signal_emitted": "Signals",
        "order_submitted": "Orders Submitted",
        "order_filled": "Orders Filled",
        "order_cancelled": "Cancelled / Rejected",
        "risk_breach": "Risk Breaches",
        "kill_switch": "Kill Switch",
        "mode_change": "Mode Changes",
        "system": "System",
    }

    selected_persona = {"value": ""}
    selected_event = {"value": ""}

    table_container = ui.column().classes("w-full")

    async def refresh() -> None:
        table_container.clear()

        pid = selected_persona["value"] or None
        etype = selected_event["value"] or None

        from uuid import UUID

        entries = await list_journal_entries(
            persona_id=UUID(pid) if pid else None,
            event_type=etype,
        )

        with table_container:
            if not entries:
                ui.label("No journal entries yet.").classes(
                    "text-body2 text-grey-6 q-pa-lg"
                )
                return

            severity_colors = {
                "info": "primary",
                "warning": "orange",
                "critical": "red",
            }

            columns = [
                {
                    "name": "time",
                    "label": "Time",
                    "field": "time",
                    "sortable": True,
                },
                {
                    "name": "severity",
                    "label": "Severity",
                    "field": "severity",
                    "sortable": True,
                },
                {
                    "name": "event_type",
                    "label": "Event",
                    "field": "event_type",
                    "sortable": True,
                },
                {
                    "name": "persona",
                    "label": "Persona",
                    "field": "persona",
                    "sortable": True,
                },
                {
                    "name": "summary",
                    "label": "Summary",
                    "field": "summary",
                },
            ]

            persona_map = {str(p.id): p.name for p in personas}

            rows = [
                {
                    "time": str(e.created_at)[:19] if e.created_at else "",
                    "severity": e.severity,
                    "event_type": e.event_type,
                    "persona": persona_map.get(str(e.persona_id), "—")
                    if e.persona_id
                    else "Global",
                    "summary": e.summary,
                }
                for e in entries
            ]

            ui.aggrid(
                {
                    "columnDefs": columns,
                    "rowData": rows,
                    "domLayout": "autoHeight",
                    "defaultColDef": {"resizable": True},
                }
            ).classes("w-full")

    with ui.row().classes("w-full gap-4 items-end q-pb-md"):

        def _on_persona_change(e):
            selected_persona["value"] = e.value

        def _on_event_change(e):
            selected_event["value"] = e.value

        ui.select(
            persona_options,
            value="",
            label="Persona",
            on_change=_on_persona_change,
        ).classes("min-w-[200px]")

        ui.select(
            event_options,
            value="",
            label="Event Type",
            on_change=_on_event_change,
        ).classes("min-w-[200px]")

        ui.button("Refresh", icon="refresh", on_click=refresh).props(
            "color=primary unelevated"
        )

    await refresh()
