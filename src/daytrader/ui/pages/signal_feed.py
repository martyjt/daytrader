"""Live Signal Feed — auto-refreshing view of recent signal emissions.

Every persona that emits a ``Signal`` writes it to the ``signals`` table.
This page polls that table every 5 seconds so users can watch activity
without manually refreshing.

Route: ``/signals``
"""

from __future__ import annotations

from nicegui import ui

from ..shell import page_layout


@ui.page("/signals")
async def signal_feed_page() -> None:
    page_layout("Live Signal Feed")

    ui.label("Live Signal Feed").classes("text-h5 q-pb-xs")
    ui.label(
        "Most recent 100 signals across all personas. Auto-refreshes every "
        "5 seconds. For a specific persona's attribution tree, open the "
        "Journal page."
    ).classes("text-caption text-grey-5 q-pb-md")

    table_container = ui.column().classes("w-full")

    async def _refresh() -> None:
        from ..services import list_personas, list_recent_signals

        try:
            signals = await list_recent_signals(limit=100)
            personas = await list_personas()
        except Exception as exc:  # noqa: BLE001
            try:
                table_container.clear()
                with table_container:
                    ui.label(f"Failed to load: {exc}").classes("text-negative")
            except RuntimeError:
                return
            return

        persona_map = {str(p.id): p.name for p in personas}

        try:
            table_container.clear()
            with table_container:
                if not signals:
                    ui.label(
                        "No signals recorded yet. Start paper trading a "
                        "persona to see signals appear here."
                    ).classes("text-caption text-grey-6")
                    return

                cols = [
                    {"name": "when", "label": "When", "field": "when", "sortable": True},
                    {"name": "persona", "label": "Persona", "field": "persona"},
                    {"name": "symbol", "label": "Symbol", "field": "symbol"},
                    {"name": "source", "label": "Source", "field": "source"},
                    {"name": "dir", "label": "Dir", "field": "dir"},
                    {"name": "score", "label": "Score", "field": "score", "sortable": True},
                    {"name": "conf", "label": "Confidence", "field": "conf"},
                    {"name": "reason", "label": "Reason", "field": "reason"},
                ]
                rows = []
                for s in signals:
                    direction = (
                        "LONG" if s.score > 0.05
                        else "SHORT" if s.score < -0.05
                        else "FLAT"
                    )
                    rows.append({
                        "when": str(s.created_at)[:19] if s.created_at else "",
                        "persona": persona_map.get(str(s.persona_id), "—"),
                        "symbol": s.symbol_key,
                        "source": s.source,
                        "dir": direction,
                        "score": f"{s.score:+.3f}",
                        "conf": f"{s.confidence:.2f}",
                        "reason": (s.reason or "")[:160],
                    })
                ui.aggrid({
                    "columnDefs": cols,
                    "rowData": rows,
                    "domLayout": "autoHeight",
                    "defaultColDef": {"sortable": True, "resizable": True},
                }).classes("w-full")

                ui.label(
                    f"{len(signals)} signal(s) · auto-refreshes every 5s"
                ).classes("text-caption text-grey-6 q-pt-xs")
        except RuntimeError:
            return

    # Auto-refresh every 5s. Exceptions on dead elements are swallowed.
    ui.timer(5.0, _refresh)
    await _refresh()
