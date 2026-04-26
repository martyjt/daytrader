"""Symbol universes — curated watchlists reusable across Portfolio + Shadow.

A universe is just a named list of symbols. Persisting them lets users
build watchlists like "Top-5 crypto", "Mega-cap tech", or "My FX
basket" and pull them into any multi-symbol workflow without re-typing.
"""

from __future__ import annotations

from nicegui import ui

from ..shell import page_layout


@ui.page("/universes")
async def universes_page() -> None:
    if not page_layout("Symbol Universes"):
        return

    ui.label("Symbol Universes").classes("text-h5 q-pb-xs")
    ui.label(
        "Reusable lists of symbols. Pick one from the dropdown in the "
        "Portfolio or Shadow Tournament tabs to populate their symbol "
        "selector in a single click."
    ).classes("text-caption text-grey-5 q-pb-md")

    from ..services import list_universes

    universes_container = ui.column().classes("w-full q-pt-md")

    async def _refresh() -> None:
        universes_container.clear()
        try:
            rows = await list_universes()
        except Exception as exc:
            with universes_container:
                ui.label(f"Failed to load universes: {exc}").classes("text-negative")
            return

        with universes_container:
            ui.label("Saved universes").classes("text-subtitle1 q-pb-xs")
            if not rows:
                ui.label("None yet — create one below.").classes("text-caption text-grey-6")
                return
            for u in rows:
                with ui.card().classes("w-full q-pa-sm q-mb-xs"):
                    with ui.row().classes("w-full items-center"):
                        ui.icon("pie_chart", color="primary")
                        ui.label(u.name).classes("text-body1 text-weight-medium")
                        ui.badge(f"{len(u.symbols)} symbols", color="grey-8").props("outline")
                        ui.space()

                        async def _delete(uid=u.id) -> None:
                            from ..services import delete_universe

                            await delete_universe(uid)
                            ui.notify("Universe deleted", type="warning")
                            await _refresh()

                        ui.button(
                            icon="delete", on_click=_delete,
                        ).props("flat dense color=negative").tooltip("Delete")
                    if u.description:
                        ui.label(u.description).classes(
                            "text-caption text-grey-5 q-pt-xs"
                        )
                    with ui.row().classes("gap-1 q-pt-xs flex-wrap"):
                        for s in u.symbols:
                            ui.badge(s, color="primary").props("outline")

    with ui.card().classes("w-full q-mt-md"):
        ui.label("Create a universe").classes("text-subtitle1 q-pb-sm")
        with ui.row().classes("w-full gap-4 items-end flex-wrap"):
            name_in = ui.input(
                "Name", placeholder="e.g. Top-5 crypto",
            ).classes("min-w-[220px]")
            symbols_in = ui.input(
                "Symbols (comma-separated)",
                placeholder="BTC-USD, ETH-USD, SOL-USD, AAPL, MSFT",
            ).classes("min-w-[360px] flex-grow")
        desc_in = ui.textarea(
            "Description (optional)",
        ).classes("w-full q-pt-sm")

        async def _save() -> None:
            try:
                from ..services import save_universe

                symbols = [s.strip() for s in (symbols_in.value or "").split(",") if s.strip()]
                if not (name_in.value or "").strip() or not symbols:
                    ui.notify("Name + at least one symbol required", type="warning")
                    return
                await save_universe(
                    name=name_in.value,
                    symbols=symbols,
                    description=desc_in.value or "",
                )
                ui.notify(f"Saved universe: {name_in.value}", type="positive")
                name_in.value = ""
                symbols_in.value = ""
                desc_in.value = ""
                await _refresh()
            except Exception as exc:
                ui.notify(f"Save failed: {exc}", type="negative")

        ui.button(
            "Save universe", icon="save", on_click=_save,
        ).props("color=primary unelevated").classes("q-mt-sm")

    await _refresh()
