"""Shared page shell: header, navigation sidebar, reusable card components.

Every page calls ``page_layout(title)`` first to get a consistent
dark-themed layout with persistent navigation.
"""

from __future__ import annotations

from typing import Any

from nicegui import ui

from ..auth.roles import ROLE_SUPER_ADMIN
from ..auth.session import current_session, require_role
from .middleware import ensure_authenticated

NAV_ITEMS: list[tuple[str, str, str]] = [
    ("Home", "/", "dashboard"),
    ("Personas", "/personas", "smart_toy"),
    ("Strategy Lab", "/strategy-lab", "science"),
    ("Strategies", "/strategies", "bookmarks"),
    ("Charts", "/charts", "candlestick_chart"),
    ("DAG Composer", "/dag-composer", "account_tree"),
    ("Bandit Builder", "/bandit-builder", "casino"),
    ("Universes", "/universes", "pie_chart"),
    ("Plugins", "/plugins", "extension"),
    ("Broker Keys", "/broker-credentials", "vpn_key"),
    ("Risk Center", "/risk", "shield"),
    ("Signals", "/signals", "radio"),
    ("Journal", "/journal", "history"),
    ("Research Lab", "/research-lab", "biotech"),
    ("Data Cache", "/cache", "dataset"),
]

ADMIN_NAV_ITEMS: list[tuple[str, str, str]] = [
    ("Users", "/admin/users", "manage_accounts"),
    ("Tenants", "/admin/tenants", "domain"),
    ("Audit", "/admin/audit", "fact_check"),
]


def page_layout(title: str) -> bool:
    """Build the shared page shell. Returns ``False`` when redirected to /login.

    Every page MUST early-return on a False return value:

        async def my_page() -> None:
            if not page_layout("My page"):
                return
            ...
    """
    if not ensure_authenticated():
        return False

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

        with ui.row().classes("items-center gap-3 no-wrap"):
            _render_regime_badge()
            _render_alerts_badge()
            _render_user_menu()
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

        if require_role(ROLE_SUPER_ADMIN):
            ui.separator().classes("q-my-sm")
            ui.label("ADMIN").classes("text-overline text-grey-7 q-pa-sm")
            for label, path, icon in ADMIN_NAV_ITEMS:
                ui.button(
                    label,
                    icon=icon,
                    on_click=lambda p=path: ui.navigate.to(p),
                ).props("flat align=left no-caps").classes("w-full")
    return True


def _render_user_menu() -> None:
    """Email + dropdown with logout."""
    sess = current_session()
    if sess is None:
        return
    label_text = sess.display_name or sess.email
    with ui.button(icon="account_circle").props("flat dense round").classes(
        "text-grey-4"
    ), ui.menu():
        with ui.row().classes("items-center q-pa-sm gap-2"):
            ui.icon("person").classes("text-grey-5")
            with ui.column().classes("gap-0"):
                ui.label(label_text).classes("text-body2")
                ui.label(sess.role).classes("text-caption text-grey-6")
        ui.separator()
        ui.menu_item("Sign out", on_click=lambda: ui.navigate.to("/logout"))


_REGIME_COLORS = {
    "bull": "positive",
    "bear": "negative",
    "sideways": "warning",
    "unknown": "grey",
}


def _render_regime_badge() -> None:
    """Renders the small regime pulse badge in the header.

    Loads lazily via a background timer so the page paints immediately.
    The HMM fit happens in a worker thread and is cached for 5 minutes
    in ``services_regime``, so this is cheap per page nav.
    """
    badge_row = ui.row().classes("items-center gap-1 no-wrap")
    with badge_row:
        ui.icon("insights", size="xs").classes("text-grey-5")
        label = ui.label("Regime: …").classes("text-caption text-grey-5")

    async def refresh() -> None:
        try:
            from .services_regime import get_current_regime

            snapshot = await get_current_regime()
        except Exception:
            return

        # Mutating a label whose parent slot has been torn down (page nav)
        # raises RuntimeError — swallow it, the next page will rebuild.
        try:
            if snapshot.status != "ok":
                label.text = "Regime: —"
                label.classes(replace="text-caption text-grey-6")
                return

            regime_text = snapshot.regime.upper()
            color_cls = _REGIME_COLORS.get(snapshot.regime, "grey")
            top_pct = round(snapshot.probabilities.get(snapshot.regime, 0.0) * 100)
            label.text = f"Regime: {regime_text} {top_pct}%"
            label.classes(replace=f"text-caption text-{color_cls}")
            probs_str = " · ".join(
                f"{k}: {round(v*100)}%" for k, v in snapshot.probabilities.items()
            )
            label.tooltip(
                f"Pulse: {snapshot.pulse_symbol} {snapshot.pulse_timeframe} "
                f"({snapshot.bars_analyzed} bars)\n{probs_str}"
            )
        except RuntimeError:
            return

    ui.timer(0.5, refresh, once=True)


_ALERT_LEVEL_COLORS = {
    "info": "primary",
    "warning": "warning",
    "critical": "negative",
}
_ALERT_LEVEL_ICONS = {
    "info": "info",
    "warning": "warning",
    "critical": "report",
}


def _render_alerts_badge() -> None:
    """Bell icon with unread count. Click opens a scrollable alerts menu.

    Count refreshes on-demand (when the dialog opens) plus a slow 15-second
    background poll — deliberately low-frequency so the UI doesn't churn.
    """
    from .alerts import alerts as _alerts

    label = ui.label("").classes("text-warning text-caption q-mr-xs")
    bell_btn = ui.button(icon="notifications_none", on_click=None).props(
        "flat dense round"
    ).classes("text-grey-4")

    def _refresh() -> None:
        # Swallow dead-element errors when the page has navigated away.
        try:
            n = _alerts().unread_count()
            label.text = (str(n) if n < 100 else "99+") if n > 0 else ""
        except RuntimeError:
            return

    dialog = ui.dialog()

    async def _open_dialog() -> None:
        items = _alerts().list(limit=30)
        _alerts().mark_all_read()
        _refresh()
        dialog.clear()
        with dialog, ui.card().classes("min-w-[520px] max-w-[720px]"):
            with ui.row().classes("w-full items-center justify-between q-pb-sm"):
                ui.label(f"Alerts ({len(items)})").classes("text-h6")
                ui.button(icon="close", on_click=dialog.close).props("flat dense")
            if not items:
                ui.label("No alerts yet.").classes("text-caption text-grey-6")
            for a in items:
                color = _ALERT_LEVEL_COLORS.get(a.level, "primary")
                icon_name = _ALERT_LEVEL_ICONS.get(a.level, "info")
                with ui.card().classes("w-full q-pa-sm q-mb-xs").style(
                    f"border-left: 3px solid var(--q-{color})"
                ):
                    with ui.row().classes("items-center gap-2"):
                        ui.icon(icon_name, color=color)
                        ui.label(a.title).classes("text-body2 text-weight-medium")
                        ui.space()
                        ui.label(a.created_at.strftime("%H:%M:%S")).classes(
                            "text-caption text-grey-6"
                        )
                    if a.body:
                        ui.label(a.body).classes("text-caption text-grey-4 q-pt-xs")
                    if a.source:
                        ui.badge(a.source, color="grey-8").props("outline")
        dialog.open()

    bell_btn.on_click(_open_dialog)
    # Refresh once at page load; users see updates when they click the bell.
    # A persistent background timer would outlive page navigation and spam
    # "dead element" errors — not worth it for a visual count.
    _refresh()


async def _kill_switch() -> None:
    from .services import kill_all_trading

    try:
        count = await kill_all_trading(reason="manual")
        ui.notify(
            f"Kill switch activated — {count} persona(s) paused",
            type="warning",
            position="top",
            timeout=5000,
        )
    except RuntimeError:
        # Kill switch not yet initialised (e.g. during startup)
        ui.notify(
            "Kill switch not ready — try again in a moment",
            type="negative",
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
    persona: Any,
    *,
    on_delete: Any = None,
    on_refresh: Any = None,
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
        with ui.row().classes("w-full justify-between flex-wrap gap-1"):
            ui.button(
                "Open",
                on_click=lambda p=persona: ui.navigate.to(f"/persona/{p.id}"),
            ).props("flat dense color=primary")

            if mode == "paper":
                _persona_asset_class = getattr(persona, "asset_class", "crypto")

                async def _do_promote(pid=persona.id, ac=_persona_asset_class):
                    venue = "binance" if ac == "crypto" else "alpaca"
                    try:
                        from .services import promote_to_live

                        _, gate_result = await promote_to_live(pid, venue)
                        if gate_result.overall_pass:
                            ui.notify(
                                "Promoted to LIVE trading!",
                                type="positive",
                            )
                        else:
                            failed = ", ".join(
                                c.gate_name for c in gate_result.failed_checks
                            )
                            ui.notify(
                                f"Gates not passed: {failed}",
                                type="warning",
                                timeout=8000,
                            )
                        if on_refresh:
                            await on_refresh()
                    except ValueError as e:
                        ui.notify(str(e), type="negative")

                ui.button("Go Live", on_click=_do_promote).props(
                    "flat dense color=green"
                )

            if on_delete:

                async def _do_delete(pid=persona.id):
                    from .services import delete_persona

                    await delete_persona(pid)
                    if on_delete:
                        await on_delete()

                ui.button("Delete", on_click=_do_delete).props(
                    "flat dense color=negative"
                )
