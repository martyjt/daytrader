"""Persona detail page — one persona's full state + activity timeline.

Shows the persona's metadata, bound strategy (if any), recent signals,
recent orders, and recent journal entries on a single page. Primary
landing when a user clicks "Open" on a persona card.

Route: ``/persona/<id>``
"""

from __future__ import annotations

from uuid import UUID

from nicegui import ui

from ..shell import page_layout


@ui.page("/persona/{persona_id}")
async def persona_detail_page(persona_id: str) -> None:
    if not page_layout("Persona Detail"):
        return

    try:
        pid = UUID(persona_id)
    except (ValueError, TypeError):
        ui.label("Invalid persona id.").classes("text-negative")
        return

    from ..services import get_persona, list_journal_entries

    persona = await get_persona(pid)
    if persona is None:
        ui.label(f"Persona {persona_id} not found.").classes("text-negative")
        return

    # ---- Header --------------------------------------------------------
    with ui.row().classes("w-full items-center q-pb-sm"):
        ui.label(persona.name).classes("text-h4")
        ui.badge(persona.mode, color={
            "paper": "blue", "live": "green",
            "backtest": "grey", "paused": "orange",
        }.get(persona.mode, "grey"))
        ui.space()
        ui.button(
            "Back to Personas",
            icon="arrow_back",
            on_click=lambda: ui.navigate.to("/personas"),
        ).props("flat dense color=primary")

    # ---- Stat cards ----------------------------------------------------
    ic = float(persona.initial_capital or 0)
    ce = float(persona.current_equity or 0)
    pnl = ce - ic
    pnl_pct = (pnl / ic * 100) if ic > 0 else 0.0
    pnl_color = "positive" if pnl >= 0 else "negative"

    with ui.row().classes("w-full gap-4 q-pb-md flex-wrap"):
        with ui.card().classes("q-pa-md min-w-[180px]"):
            ui.label("Equity").classes("text-caption text-grey-5")
            ui.label(f"${ce:,.2f}").classes("text-h5 text-weight-bold")
        with ui.card().classes("q-pa-md min-w-[180px]"):
            ui.label("P&L").classes("text-caption text-grey-5")
            ui.label(f"{pnl:+,.2f} ({pnl_pct:+.2f}%)").classes(
                f"text-h5 text-weight-bold text-{pnl_color}"
            )
        with ui.card().classes("q-pa-md min-w-[180px]"):
            ui.label("Risk Profile").classes("text-caption text-grey-5")
            ui.label(persona.risk_profile.upper()).classes("text-h5 text-weight-bold")
        with ui.card().classes("q-pa-md min-w-[180px]"):
            ui.label("Asset Class").classes("text-caption text-grey-5")
            ui.label(persona.asset_class.upper()).classes("text-h5 text-weight-bold")

    # ---- Bound strategy summary ----------------------------------------
    meta = dict(persona.meta or {})
    strategy_id = meta.get("strategy_config_id")

    with ui.card().classes("w-full q-pa-md q-mb-md"):
        with ui.row().classes("w-full items-center"):
            ui.icon("bookmarks", color="primary")
            ui.label("Strategy").classes("text-h6")
            ui.space()
            if strategy_id:
                ui.button(
                    "Open in Strategy Lab",
                    icon="launch",
                    on_click=lambda: ui.navigate.to(f"/strategy-lab?strategy={strategy_id}"),
                ).props("flat color=primary")

        if strategy_id:
            with ui.row().classes("gap-4 q-pt-xs"):
                _kv("Name", meta.get("strategy_name", "—"))
                _kv("Algorithm", meta.get("algo_id", "—"))
                _kv("Symbol", meta.get("symbol", "—"))
                _kv("Timeframe", meta.get("timeframe", "—"))
                _kv("Venue", meta.get("venue", "—"))
        else:
            ui.label(
                "No strategy bound. Create a saved strategy in the "
                "Strategies page and link it when creating the persona, "
                "or set it here via the meta field."
            ).classes("text-caption text-grey-6 q-pt-xs")

    # ---- Recent journal entries --------------------------------------------
    ui.label("Recent activity").classes("text-h6 q-pb-xs")
    try:
        entries = await list_journal_entries(persona_id=pid, limit=30)
    except Exception as exc:
        ui.label(f"Failed to load journal: {exc}").classes("text-negative")
        entries = []

    if not entries:
        ui.label(
            "No journal activity yet. Signals and orders will appear here "
            "once this persona starts trading."
        ).classes("text-caption text-grey-6")
    else:
        cols = [
            {"name": "when", "label": "When", "field": "when", "sortable": True},
            {"name": "severity", "label": "Severity", "field": "severity"},
            {"name": "type", "label": "Event", "field": "type"},
            {"name": "summary", "label": "Summary", "field": "summary"},
        ]
        rows = [
            {
                "when": str(e.created_at)[:19] if e.created_at else "",
                "severity": e.severity,
                "type": e.event_type,
                "summary": e.summary,
            }
            for e in entries
        ]
        ui.aggrid({
            "columnDefs": cols,
            "rowData": rows,
            "domLayout": "autoHeight",
            "defaultColDef": {"sortable": True, "resizable": True},
        }).classes("w-full")


def _kv(label: str, value: str) -> None:
    with ui.column().classes("gap-0"):
        ui.label(label).classes("text-caption text-grey-5")
        ui.label(str(value)).classes("text-body2 text-weight-medium")
