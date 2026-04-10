"""Shared page shell: header, navigation sidebar, reusable card components.

Every page calls ``page_layout(title)`` first to get a consistent
dark-themed layout with persistent navigation.
"""

from __future__ import annotations

from nicegui import ui

NAV_ITEMS: list[tuple[str, str, str]] = [
    ("Home", "/", "dashboard"),
    ("Personas", "/personas", "smart_toy"),
    ("Strategy Lab", "/strategy-lab", "science"),
    ("Plugins", "/plugins", "extension"),
    ("Risk Center", "/risk", "shield"),
]


def page_layout(title: str) -> None:
    """Build the shared page shell: dark theme, header bar, sidebar nav."""
    ui.dark_mode(True)
    ui.colors(primary="#5c7cfa", secondary="#495057", accent="#22b8cf")

    # ---- Header ----------------------------------------------------------
    with ui.header(elevated=True).classes(
        "items-center justify-between q-px-md"
    ).style("background-color: #1a1b2e"):
        with ui.row().classes("items-center gap-2 no-wrap"):
            ui.icon("show_chart", size="sm").classes("text-primary")
            ui.label("Daytrader").classes("text-h6 text-weight-bold")

        ui.label(title).classes("text-subtitle1 text-grey-5")

        ui.button(
            "KILL ALL",
            color="negative",
            on_click=_kill_switch,
        ).props("flat dense")

    # ---- Sidebar ---------------------------------------------------------
    with ui.left_drawer(value=True, fixed=True, bordered=True).style(
        "background-color: #141522"
    ):
        ui.label("NAVIGATION").classes("text-overline text-grey-7 q-pa-sm")
        for label, path, icon in NAV_ITEMS:
            ui.button(
                label,
                icon=icon,
                on_click=lambda p=path: ui.navigate.to(p),
            ).props("flat align=left no-caps").classes("w-full")


def _kill_switch() -> None:
    ui.notify(
        "Kill switch activated — all live trading halted",
        type="warning",
        position="top",
    )


# ---- Reusable card components -------------------------------------------


def stat_card(
    label: str,
    value: str | int | float,
    *,
    icon: str = "info",
    color: str = "primary",
) -> None:
    """Small summary stat card for dashboards."""
    with ui.card().classes("q-pa-md min-w-[160px]"):
        with ui.row().classes("items-center gap-2"):
            ui.icon(icon, size="sm", color=color)
            ui.label(label).classes("text-caption text-grey-5")
        ui.label(str(value)).classes("text-h5 text-weight-bold q-pt-xs")


def persona_card(
    persona: object,
    *,
    on_delete=None,
) -> None:
    """Card showing a persona's name, mode, equity, P&L, and actions."""
    ic = float(getattr(persona, "initial_capital", 0) or 0)
    ce = float(getattr(persona, "current_equity", 0) or 0)
    pnl_pct = ((ce - ic) / ic * 100) if ic > 0 else 0.0
    pnl_color = "positive" if pnl_pct >= 0 else "negative"
    pnl_icon = "trending_up" if pnl_pct >= 0 else "trending_down"

    mode = getattr(persona, "mode", "paper")
    mode_colors = {
        "paper": "blue",
        "live": "green",
        "backtest": "grey",
        "paused": "orange",
    }

    with ui.card().classes("w-72"):
        # Header row: name + mode badge
        with ui.row().classes("w-full items-center justify-between"):
            ui.label(persona.name).classes("text-h6")
            ui.badge(mode, color=mode_colors.get(mode, "grey"))

        ui.separator()

        # Stats
        with ui.column().classes("gap-1"):
            with ui.row().classes("items-center gap-1"):
                ui.label("Equity:").classes("text-body2 text-grey-6")
                ui.label(f"${ce:,.2f}").classes("text-body1 text-weight-medium")

            with ui.row().classes("items-center gap-1"):
                ui.icon(pnl_icon, size="xs", color=pnl_color)
                ui.label(f"{pnl_pct:+.1f}%").classes(f"text-body2 text-{pnl_color}")

            with ui.row().classes("items-center gap-1"):
                ui.label(
                    getattr(persona, "asset_class", "crypto")
                ).classes("text-caption text-grey-6")
                ui.label(
                    f"· {getattr(persona, 'risk_profile', 'balanced')}"
                ).classes("text-caption text-grey-6")

        ui.separator()

        # Actions
        with ui.row().classes("w-full justify-between"):
            ui.button(
                "Open",
                on_click=lambda p=persona: ui.navigate.to(
                    f"/strategy-lab?persona={p.id}"
                ),
            ).props("flat dense color=primary")

            if on_delete:

                async def _do_delete(pid=persona.id):
                    from .services import delete_persona

                    await delete_persona(pid)
                    if on_delete:
                        await on_delete()

                ui.button("Delete", on_click=_do_delete).props(
                    "flat dense color=negative"
                )
