"""Home page — 'What's happening today?'

Summary dashboard: portfolio stats, current regime, recent discoveries,
most recent alerts, and a persona grid. Designed so a user landing cold
in the morning can see whether anything important changed overnight.
"""

from __future__ import annotations

from nicegui import ui

from ...auth.session import current_tenant_id
from ..services import list_discoveries, list_personas
from ..services_onboarding import get_onboarding_status
from ..shell import page_layout, persona_card, stat_card


@ui.page("/")
async def home_page(welcome: str | None = None) -> None:
    # Phase 13 — bounce fresh tenants to /welcome unless they explicitly
    # asked to skip via the welcome page's "Skip for now" link. The check
    # runs before page_layout() so the welcome page paints first instead
    # of flashing the empty home dashboard.
    if welcome is None:
        tenant_id = current_tenant_id()
        if tenant_id is not None:
            status = await get_onboarding_status(tenant_id)
            if status.is_fresh:
                ui.navigate.to("/welcome")
                return

    if not page_layout("What's happening today?"):
        return

    personas = await list_personas()
    total_equity = sum(float(p.current_equity or 0) for p in personas)
    active = sum(1 for p in personas if p.mode in ("paper", "live"))

    # ---- Summary stats ---------------------------------------------------
    with ui.row().classes("w-full gap-4 q-pb-md flex-wrap"):
        stat_card("Personas", len(personas), icon="smart_toy")
        stat_card(
            "Total Equity",
            f"${total_equity:,.2f}",
            icon="account_balance",
        )
        stat_card("Active", active, icon="play_circle", color="positive")
        await _render_regime_summary_card()

    ui.separator()

    # ---- Recent alerts preview + discoveries side-by-side ---------------
    with ui.row().classes("w-full gap-4 q-pt-md items-stretch").style(
        "flex-wrap: wrap"
    ):
        with ui.card().classes("col-grow min-w-[320px]"):
            ui.label("Recent alerts").classes("text-subtitle1")
            _render_recent_alerts()
        with ui.card().classes("col-grow min-w-[320px]"):
            ui.label("Top recent discoveries").classes("text-subtitle1")
            await _render_recent_discoveries()

    # ---- Persona cards or empty state ------------------------------------
    if personas:
        ui.label("Your Personas").classes("text-h6 q-pt-md q-pb-sm")
        with ui.row().classes("w-full gap-4 flex-wrap"):
            for p in personas:
                persona_card(p)
    else:
        with ui.column().classes("w-full items-center q-pa-xl"):
            ui.icon("rocket_launch", size="64px").classes("text-grey-6")
            ui.label("No personas yet").classes("text-h6 text-grey-6 q-pb-sm")
            ui.label(
                "Create a persona to start testing algorithms"
            ).classes("text-body2 text-grey-7")
            ui.button(
                "Create your first persona",
                icon="add",
                on_click=lambda: ui.navigate.to("/personas"),
            ).props("color=primary unelevated").classes("q-mt-md")


async def _render_regime_summary_card() -> None:
    """A 4th stat card showing the current regime pulse."""
    try:
        from ..services_regime import get_current_regime

        snap = await get_current_regime()
    except Exception:
        return

    if snap.status != "ok":
        stat_card("Market Regime", "…", icon="insights")
        return

    color = {
        "bull": "positive",
        "bear": "negative",
        "sideways": "warning",
    }.get(snap.regime, "primary")
    top_pct = round(snap.probabilities.get(snap.regime, 0.0) * 100)
    stat_card(
        "Market Regime",
        f"{snap.regime.upper()} {top_pct}%",
        icon="insights",
        color=color,
    )


def _render_recent_alerts() -> None:
    """Inline preview of the 5 most recent alerts."""
    from ..alerts import alerts as _alerts

    items = _alerts().list(limit=5)
    if not items:
        ui.label(
            "Nothing flagged yet — background monitors will post here as "
            "things happen."
        ).classes("text-caption text-grey-6")
        return

    level_colors = {"info": "primary", "warning": "warning", "critical": "negative"}
    level_icons = {"info": "info", "warning": "warning", "critical": "report"}

    for a in items:
        color = level_colors.get(a.level, "primary")
        with ui.row().classes("w-full items-center gap-2 q-py-xs"):
            ui.icon(level_icons.get(a.level, "info"), color=color, size="sm")
            with ui.column().classes("gap-0 col-grow"):
                ui.label(a.title).classes("text-body2 text-weight-medium")
                if a.body:
                    ui.label(a.body[:140] + ("…" if len(a.body) > 140 else "")).classes(
                        "text-caption text-grey-5"
                    )
            ui.label(a.created_at.strftime("%H:%M")).classes(
                "text-caption text-grey-6"
            )


async def _render_recent_discoveries() -> None:
    """Top 5 significant discoveries by lift, most recent first."""
    try:
        rows = await list_discoveries(significant_only=True, limit=5)
    except Exception as exc:
        ui.label(f"Could not load discoveries: {exc}").classes("text-negative text-caption")
        return
    if not rows:
        ui.label(
            "No significant discoveries yet. Run an Exploration Agent scan "
            "from Research Lab → Discoveries to start populating this list."
        ).classes("text-caption text-grey-6")
        return

    for r in rows:
        with ui.row().classes("w-full items-center gap-2 q-py-xs"):
            ui.icon("travel_explore", color="primary", size="sm")
            with ui.column().classes("gap-0 col-grow"):
                ui.label(r.candidate_name).classes("text-body2 text-weight-medium")
                ui.label(
                    f"{r.candidate_source} · lift {r.lift:+.3f} · "
                    f"q {_fmt_q(r.q_value)} on {r.target_symbol}"
                ).classes("text-caption text-grey-5")
            ui.label(
                r.created_at.strftime("%m-%d") if r.created_at else ""
            ).classes("text-caption text-grey-6")


def _fmt_q(q) -> str:
    if q is None:
        return "—"
    return f"{q:.3f}"
