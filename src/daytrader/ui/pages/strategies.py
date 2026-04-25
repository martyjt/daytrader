"""Strategy Library — save and reuse algo + params + market bindings.

A saved strategy is a recipe: algo id, symbol, timeframe, venue, and
any algorithm-specific parameters the user tuned. Promoting one is a
one-click path into Strategy Lab (where the user can backtest / walk-
forward / promote to paper).

Parallel to personas: a persona binds a strategy to capital + mode;
the saved strategy here is just the reusable recipe.
"""

from __future__ import annotations

from nicegui import ui

from ..shell import page_layout


@ui.page("/strategies")
async def strategies_page() -> None:
    if not page_layout("Strategy Library"):
        return

    from ...algorithms.registry import AlgorithmRegistry
    from ..services import (
        delete_strategy_config,
        list_strategies,
        save_strategy_config,
    )

    from ...auth.session import current_tenant_id
    _tid = current_tenant_id()
    algo_ids = AlgorithmRegistry.available(tenant_id=_tid)
    algo_labels = {
        aid: AlgorithmRegistry.get(aid, tenant_id=_tid).manifest.name for aid in algo_ids
    }

    ui.label("Strategy Library").classes("text-h5 q-pb-xs")
    ui.label(
        "Save named configurations (algo + params + symbol + venue) for "
        "quick reuse in Strategy Lab or when spinning up a new persona."
    ).classes("text-caption text-grey-5 q-pb-md")

    container = ui.column().classes("w-full q-pt-md")

    async def _refresh() -> None:
        container.clear()
        try:
            rows = await list_strategies()
        except Exception as exc:  # noqa: BLE001
            with container:
                ui.label(f"Failed to load: {exc}").classes("text-negative")
            return

        with container:
            ui.label("Saved strategies").classes("text-subtitle1 q-pb-xs")
            if not rows:
                ui.label("None yet — save one below.").classes("text-caption text-grey-6")
                return

            cols = [
                {"name": "name", "label": "Name", "field": "name"},
                {"name": "algo", "label": "Algorithm", "field": "algo"},
                {"name": "symbol", "label": "Symbol", "field": "symbol"},
                {"name": "tf", "label": "Timeframe", "field": "tf"},
                {"name": "venue", "label": "Venue", "field": "venue"},
                {"name": "tags", "label": "Tags", "field": "tags"},
                {"name": "when", "label": "Updated", "field": "when"},
            ]
            table_rows = []
            for r in rows:
                algo_name = algo_labels.get(r.algo_id, r.algo_id)
                table_rows.append({
                    "id": str(r.id),
                    "name": r.name,
                    "algo": algo_name,
                    "symbol": r.symbol,
                    "tf": r.timeframe,
                    "venue": r.venue,
                    "tags": ", ".join(r.tags or []) or "—",
                    "when": r.updated_at.strftime("%Y-%m-%d %H:%M") if r.updated_at else "",
                })
            ui.aggrid({
                "columnDefs": cols,
                "rowData": table_rows,
                "domLayout": "autoHeight",
                "defaultColDef": {"sortable": True, "resizable": True},
            }).classes("w-full")

            # Per-strategy action row for load / delete
            for r in rows:
                with ui.row().classes("w-full items-center gap-2 q-py-xs"):
                    ui.badge(r.name, color="primary")
                    ui.label(
                        f"{algo_labels.get(r.algo_id, r.algo_id)} · "
                        f"{r.symbol} · {r.timeframe} · {r.venue}"
                    ).classes("text-caption text-grey-5")
                    ui.space()
                    ui.button(
                        "Open in Strategy Lab",
                        icon="launch",
                        on_click=lambda sid=r.id: ui.navigate.to(f"/strategy-lab?strategy={sid}"),
                    ).props("flat dense color=primary")

                    async def _del(sid=r.id, name=r.name) -> None:
                        await delete_strategy_config(sid)
                        ui.notify(f"Deleted strategy: {name}", type="warning")
                        await _refresh()

                    ui.button(
                        icon="delete", on_click=_del,
                    ).props("flat dense color=negative").tooltip("Delete")

    with ui.card().classes("w-full q-mt-md"):
        ui.label("Save new strategy").classes("text-subtitle1 q-pb-sm")
        with ui.row().classes("w-full gap-4 items-end flex-wrap"):
            name_in = ui.input(
                "Name", placeholder="e.g. BTC trend follower",
            ).classes("min-w-[220px]")
            algo_in = ui.select(
                options=algo_labels,
                value=algo_ids[0] if algo_ids else "",
                label="Algorithm",
            ).classes("min-w-[220px]")
            symbol_in = ui.input(
                "Symbol", value="BTC-USD",
            ).classes("min-w-[140px]")
            tf_in = ui.select(
                ["1d", "4h", "1h", "15m"], value="1d", label="Timeframe",
            ).classes("min-w-[100px]")

        with ui.row().classes("w-full gap-4 items-end q-pt-sm"):
            try:
                from ...backtest.fees import VENUE_PROFILES

                venue_labels = {k: v.venue for k, v in VENUE_PROFILES.items()}
            except Exception:
                venue_labels = {"binance_spot": "Binance Spot"}
            venue_in = ui.select(
                options=venue_labels, value="binance_spot", label="Venue",
            ).classes("min-w-[200px]")
            tags_in = ui.input(
                "Tags (comma-separated)",
                placeholder="momentum, crypto, intraday",
            ).classes("min-w-[260px] flex-grow")

        desc_in = ui.textarea(
            "Description (optional)",
        ).classes("w-full q-pt-sm")

        async def _save() -> None:
            try:
                nm = (name_in.value or "").strip()
                if not nm:
                    ui.notify("Name required", type="warning")
                    return
                tags = [t.strip() for t in (tags_in.value or "").split(",") if t.strip()]
                await save_strategy_config(
                    name=nm,
                    algo_id=algo_in.value,
                    symbol=symbol_in.value,
                    timeframe=tf_in.value,
                    venue=venue_in.value,
                    algo_params={},
                    description=desc_in.value or "",
                    tags=tags,
                )
                ui.notify(f"Saved strategy: {nm}", type="positive")
                name_in.value = ""
                tags_in.value = ""
                desc_in.value = ""
                await _refresh()
            except Exception as exc:  # noqa: BLE001
                ui.notify(f"Save failed: {exc}", type="negative")

        ui.button("Save strategy", icon="save", on_click=_save).props(
            "color=primary unelevated"
        ).classes("q-mt-sm")

    await _refresh()
