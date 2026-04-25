"""Journal page — activity log for all trading events.

Displays signals, order fills, risk breaches, kill-switch activations,
and mode changes across all personas. Clicking any row opens an
explainability drawer that shows full detail including the signal's
attribution tree (which algorithm / combinator produced what score at
that moment), closing the loop between "what happened in production"
and "why".

Supports CSV export for audit / offline analysis.
"""

from __future__ import annotations

import csv
import io
import json
from typing import Any

from nicegui import ui

from ..services import list_journal_entries, list_personas
from ..shell import page_layout


def _entries_to_csv(entries, persona_map: dict[str, str]) -> str:
    """Serialize journal entries to CSV bytes (UTF-8, RFC4180)."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "created_at", "event_type", "severity", "persona",
        "summary", "detail_json",
    ])
    for e in entries:
        persona = (
            persona_map.get(str(e.persona_id), "—")
            if e.persona_id else "Global"
        )
        writer.writerow([
            str(e.created_at) if e.created_at else "",
            e.event_type,
            e.severity,
            persona,
            e.summary,
            json.dumps(dict(e.detail or {}), default=str),
        ])
    return buf.getvalue()


@ui.page("/journal")
async def journal_page() -> None:
    if not page_layout("Journal"):
        return

    # ---- Filters ------------------------------------------------------------
    ui.label("Activity Journal").classes("text-h5 q-pb-sm")
    ui.label(
        "Click any row to see the full event detail and (for signals) "
        "the attribution tree — which algorithm or combinator drove "
        "that score at that moment."
    ).classes("text-caption text-grey-6 q-pb-sm")

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
    entries_by_id: dict[str, Any] = {}

    table_container = ui.column().classes("w-full")
    detail_drawer = ui.column().classes("w-full q-pt-md")

    async def refresh() -> None:
        table_container.clear()
        detail_drawer.clear()
        entries_by_id.clear()

        pid = selected_persona["value"] or None
        etype = selected_event["value"] or None

        from uuid import UUID

        entries = await list_journal_entries(
            persona_id=UUID(pid) if pid else None,
            event_type=etype,
        )
        for e in entries:
            entries_by_id[str(e.id)] = e

        with table_container:
            if not entries:
                ui.label("No journal entries yet.").classes(
                    "text-body2 text-grey-6 q-pa-lg"
                )
                return

            columns = [
                {"name": "time", "label": "Time", "field": "time", "sortable": True},
                {"name": "severity", "label": "Severity", "field": "severity", "sortable": True},
                {"name": "event_type", "label": "Event", "field": "event_type", "sortable": True},
                {"name": "persona", "label": "Persona", "field": "persona", "sortable": True},
                {"name": "summary", "label": "Summary", "field": "summary"},
            ]

            persona_map = {str(p.id): p.name for p in personas}

            rows = [
                {
                    "id": str(e.id),
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

            grid = ui.aggrid({
                "columnDefs": columns,
                "rowData": rows,
                "domLayout": "autoHeight",
                "defaultColDef": {"resizable": True},
            }).classes("w-full")

            def on_row_clicked(event) -> None:
                row = event.args.get("data") if hasattr(event, "args") else None
                if not row:
                    return
                entry = entries_by_id.get(row.get("id"))
                if entry is not None:
                    _render_detail_drawer(detail_drawer, entry, persona_map)

            grid.on("cellClicked", on_row_clicked)

    with ui.row().classes("w-full gap-4 items-end q-pb-md"):

        def _on_persona_change(e):
            selected_persona["value"] = e.value

        def _on_event_change(e):
            selected_event["value"] = e.value

        ui.select(
            persona_options, value="", label="Persona",
            on_change=_on_persona_change,
        ).classes("min-w-[200px]")

        ui.select(
            event_options, value="", label="Event Type",
            on_change=_on_event_change,
        ).classes("min-w-[200px]")

        ui.button("Refresh", icon="refresh", on_click=refresh).props(
            "color=primary unelevated"
        )

        async def _export_csv() -> None:
            from uuid import UUID

            pid = selected_persona["value"] or None
            etype = selected_event["value"] or None
            rows = await list_journal_entries(
                persona_id=UUID(pid) if pid else None,
                event_type=etype,
                limit=2000,
            )
            if not rows:
                ui.notify("No entries to export", type="warning")
                return
            pmap = {str(p.id): p.name for p in personas}
            csv_text = _entries_to_csv(rows, pmap)

            # Trigger a client-side download via a data URL.
            from urllib.parse import quote

            filename = f"journal-{len(rows)}-entries.csv"
            data_url = f"data:text/csv;charset=utf-8,{quote(csv_text)}"
            ui.run_javascript(
                "(() => {"
                f"  const a = document.createElement('a');"
                f"  a.href = {data_url!r};"
                f"  a.download = {filename!r};"
                f"  document.body.appendChild(a);"
                f"  a.click();"
                f"  a.remove();"
                "})()"
            )
            ui.notify(f"Exported {len(rows)} entries → {filename}", type="positive")

        ui.button(
            "Export CSV", icon="file_download", on_click=_export_csv,
        ).props("flat color=primary").tooltip(
            "Download the currently-filtered view as CSV"
        )

    await refresh()


# ----------------------------------------------------------------------
# Detail drawer (explainability popover)
# ----------------------------------------------------------------------


def _render_detail_drawer(
    container, entry, persona_map: dict[str, str],
) -> None:
    """Render the full detail of a journal entry, including attribution tree."""
    container.clear()
    with container:
        with ui.card().classes("w-full"):
            with ui.row().classes("w-full items-center justify-between"):
                ui.label("Event Detail").classes("text-h6")
                ui.button(icon="close", on_click=container.clear).props(
                    "flat dense"
                )

            # Header metadata
            persona_label = persona_map.get(str(entry.persona_id), "—") if entry.persona_id else "Global"
            with ui.row().classes("w-full gap-4 q-pt-xs flex-wrap"):
                _chip("Time", str(entry.created_at)[:19] if entry.created_at else "—")
                _chip("Event", entry.event_type)
                _chip("Severity", entry.severity)
                _chip("Persona", persona_label)

            ui.separator().classes("q-my-sm")
            ui.label("Summary").classes("text-subtitle2")
            ui.label(entry.summary).classes("text-body2 q-pb-sm")

            detail = dict(entry.detail or {})

            # If this is a signal-emitted event with attribution, render the tree.
            attribution = detail.get("attribution")
            if attribution:
                ui.label("Attribution tree").classes("text-subtitle2 q-pt-sm")
                _render_attribution(attribution)

            # Structured detail payload (excluding attribution — rendered above)
            rest = {k: v for k, v in detail.items() if k != "attribution"}
            if rest:
                ui.label("Raw detail").classes("text-subtitle2 q-pt-sm")
                with ui.card().classes("w-full bg-dark").style(
                    "font-family: ui-monospace, monospace"
                ):
                    ui.label(json.dumps(rest, indent=2, default=str)).classes(
                        "text-caption"
                    ).style("white-space: pre-wrap")


def _chip(label: str, value: str) -> None:
    with ui.column().classes("gap-0"):
        ui.label(label).classes("text-caption text-grey-6")
        ui.label(value).classes("text-body2")


def _render_attribution(attribution: Any, depth: int = 0) -> None:
    """Recursively render a SignalContribution tree as nested cards."""
    if not isinstance(attribution, dict):
        return

    node_id = attribution.get("node_id", "?")
    node_type = attribution.get("node_type", "")
    score = attribution.get("score")
    confidence = attribution.get("confidence")
    weight = attribution.get("weight")
    reason = attribution.get("reason", "")
    children = attribution.get("children", []) or []

    indent_px = 16 * depth
    score_color = (
        "positive" if isinstance(score, (int, float)) and score > 0
        else "negative" if isinstance(score, (int, float)) and score < 0
        else "grey-5"
    )

    with ui.card().classes("w-full q-pa-sm q-mb-xs").style(
        f"margin-left: {indent_px}px; border-left: 3px solid #5c7cfa"
    ):
        with ui.row().classes("items-center gap-3 flex-wrap"):
            ui.icon(
                "widgets" if node_type == "combinator" else "functions",
                size="sm", color="primary",
            )
            ui.label(node_id).classes("text-body2 text-weight-medium")
            ui.badge(node_type, color="grey-8")
            if isinstance(score, (int, float)):
                ui.badge(f"score {score:+.3f}", color=score_color)
            if isinstance(confidence, (int, float)):
                ui.badge(f"conf {confidence:.2f}", color="info")
            if isinstance(weight, (int, float)) and weight is not None:
                ui.label(f"weight {weight:.2f}").classes("text-caption text-grey-6")
        if reason:
            ui.label(reason).classes("text-caption text-grey-5 q-pt-xs")

    for child in children:
        _render_attribution(child, depth=depth + 1)
