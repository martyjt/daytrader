"""Plugins page — view and manage algorithm plugins.

Shows all registered algorithms from the AlgorithmRegistry (built-in
and user-installed plugins from the plugins/ directory).
"""

from __future__ import annotations

from nicegui import ui

from ..shell import page_layout


@ui.page("/plugins")
async def plugins_page() -> None:
    page_layout("Plugins")

    ui.label("Algorithm Library").classes("text-h5 q-pb-sm")
    ui.label(
        "These algorithms are available for backtesting in the Strategy Lab. "
        "Drop new plugins into the plugins/ directory to add more."
    ).classes("text-body2 text-grey-6 q-pb-md")

    # ---- Registered algorithms from the real registry ---------------------
    from ...algorithms.registry import AlgorithmRegistry

    registered = AlgorithmRegistry.all()

    with ui.row().classes("w-full gap-4 flex-wrap"):
        for algo_id, algo in registered.items():
            m = algo.manifest
            with ui.card().classes("w-72"):
                with ui.row().classes("w-full items-center justify-between"):
                    ui.label(m.name).classes("text-h6")
                    ui.badge("Active", color="positive")

                ui.label(m.description).classes(
                    "text-body2 text-grey-5 q-py-xs"
                )

                with ui.row().classes("gap-1 q-pb-xs flex-wrap"):
                    for ac in m.asset_classes:
                        ui.chip(ac, color="teal").props("dense outline")
                    for tf in m.timeframes[:4]:
                        ui.chip(tf, color="blue").props("dense outline")
                    if len(m.timeframes) > 4:
                        ui.chip(
                            f"+{len(m.timeframes) - 4} more", color="grey"
                        ).props("dense outline")

                # Show param count
                if m.params:
                    ui.label(
                        f"{len(m.params)} configurable parameters"
                    ).classes("text-caption text-grey-7 q-py-xs")

                with ui.row().classes("items-center gap-2"):
                    ui.label(f"v{m.version}").classes(
                        "text-caption text-grey-7"
                    )
                    if m.author:
                        ui.label(f"· {m.author}").classes(
                            "text-caption text-grey-7"
                        )

                ui.switch("Enabled", value=True)

    if not registered:
        ui.label(
            "No algorithms registered. Check that auto_register() "
            "ran at startup."
        ).classes("text-body2 text-grey-6")
