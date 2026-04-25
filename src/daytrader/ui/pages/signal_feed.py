"""Live Signal Feed — pushes new signals as the trading loop emits them.

Loads the most recent 100 rows from ``SignalModel`` once on page mount,
then subscribes to the in-process ``signal_bus`` for the active tenant.
A background task awaits new ``SignalEvent`` payloads and prepends them
to the table — no polling, no re-fetch.

The subscription is torn down on client disconnect. Late events that
arrive while the table is being rebuilt are queued; the per-subscriber
queue drops the oldest on overflow (see ``core.pubsub``).

Route: ``/signals``
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from nicegui import context, ui

from ...auth.session import current_tenant_id
from ...core.pubsub import SignalEvent, signal_bus
from ..shell import page_layout

logger = logging.getLogger(__name__)


_MAX_ROWS = 100


def _direction(score: float) -> str:
    if score > 0.05:
        return "LONG"
    if score < -0.05:
        return "SHORT"
    return "FLAT"


def _row_from_signal(s: Any, persona_map: dict[str, str]) -> dict:
    return {
        "when": str(s.created_at)[:19] if s.created_at else "",
        "persona": persona_map.get(str(s.persona_id), "—"),
        "symbol": s.symbol_key,
        "source": s.source,
        "dir": _direction(s.score),
        "score": f"{s.score:+.3f}",
        "conf": f"{s.confidence:.2f}",
        "reason": (s.reason or "")[:160],
    }


def _row_from_event(ev: SignalEvent, persona_map: dict[str, str]) -> dict:
    return {
        "when": ev.created_at[:19] if ev.created_at else "",
        "persona": persona_map.get(str(ev.persona_id), "—"),
        "symbol": ev.symbol_key,
        "source": ev.source,
        "dir": _direction(ev.score),
        "score": f"{ev.score:+.3f}",
        "conf": f"{ev.confidence:.2f}",
        "reason": (ev.reason or "")[:160],
    }


@ui.page("/signals")
async def signal_feed_page() -> None:
    if not page_layout("Live Signal Feed"):
        return

    ui.label("Live Signal Feed").classes("text-h5 q-pb-xs")
    status_label = ui.label(
        "Most recent 100 signals across all personas. Streams live as the "
        "trading loop emits them — no manual refresh."
    ).classes("text-caption text-grey-5 q-pb-md")

    grid_container = ui.column().classes("w-full")
    footer_row = ui.row().classes("w-full q-pt-xs")

    # ---- Initial load ---------------------------------------------------
    from ..services import list_personas, list_recent_signals

    try:
        signals = await list_recent_signals(limit=_MAX_ROWS)
        personas = await list_personas()
    except Exception as exc:  # noqa: BLE001
        with grid_container:
            ui.label(f"Failed to load: {exc}").classes("text-negative")
        return

    persona_map = {str(p.id): p.name for p in personas}
    rows: list[dict] = [_row_from_signal(s, persona_map) for s in signals]

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

    with grid_container:
        if not rows:
            empty_label = ui.label(
                "No signals recorded yet. Start paper trading a persona to "
                "see signals appear here in real time."
            ).classes("text-caption text-grey-6")
            grid = None
        else:
            empty_label = None
            grid = ui.aggrid({
                "columnDefs": cols,
                "rowData": rows,
                "domLayout": "autoHeight",
                "defaultColDef": {"sortable": True, "resizable": True},
            }).classes("w-full")

    with footer_row:
        count_label = ui.label(
            f"{len(rows)} signal(s) · live"
        ).classes("text-caption text-grey-6")

    # ---- Live subscription ---------------------------------------------
    tenant_id = current_tenant_id()
    if tenant_id is None:
        return  # page_layout already redirected, but guard for safety

    state = {"grid": grid, "empty_label": empty_label, "count_label": count_label}

    def _apply_event(ev: SignalEvent) -> None:
        """Prepend a new row and refresh the grid in-place."""
        new_row = _row_from_event(ev, persona_map)
        rows.insert(0, new_row)
        if len(rows) > _MAX_ROWS:
            del rows[_MAX_ROWS:]

        try:
            if state["grid"] is None:
                # First signal — replace the empty-state label with a real grid.
                if state["empty_label"] is not None:
                    state["empty_label"].delete()
                    state["empty_label"] = None
                with grid_container:
                    state["grid"] = ui.aggrid({
                        "columnDefs": cols,
                        "rowData": rows,
                        "domLayout": "autoHeight",
                        "defaultColDef": {"sortable": True, "resizable": True},
                    }).classes("w-full")
            else:
                state["grid"].options["rowData"] = list(rows)
                state["grid"].update()
            state["count_label"].text = f"{len(rows)} signal(s) · live"
        except RuntimeError:
            # Element torn down by page nav — let the disconnect handler
            # cancel the subscriber. Don't try to mutate further.
            raise

    async def _subscriber() -> None:
        bus = signal_bus()
        with bus.subscribe(tenant_id) as queue:
            while True:
                event = await queue.get()
                if not isinstance(event, SignalEvent):
                    continue
                try:
                    _apply_event(event)
                except RuntimeError:
                    return

    task = asyncio.create_task(_subscriber(), name=f"signal-feed-{tenant_id}")

    def _cancel_task() -> None:
        if not task.done():
            task.cancel()

    # Cancel cleanly on client disconnect (also fires on reconnect — see
    # NiceGUI 3.0 release notes — which is fine: the new connection
    # re-mounts the page and starts a fresh subscriber).
    try:
        client = context.client
        client.on_disconnect(_cancel_task)
    except Exception:
        # Tests / non-page contexts: best-effort cleanup.
        logger.debug("signal_feed: no client context for disconnect cleanup")
