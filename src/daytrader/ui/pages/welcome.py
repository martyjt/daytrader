"""Welcome page — the three-step checklist a fresh tenant lands on.

Renders one card per onboarding step (broker keys, first persona,
notifications), each marked done ✅ or pending. Deep links to the
relevant page so the user moves through the checklist without hunting.
A "Skip for now" link drops them onto the home dashboard if they want
to look around first.
"""

from __future__ import annotations

from nicegui import ui

from ...auth.session import current_tenant_id
from ..services_onboarding import get_onboarding_status
from ..shell import page_layout


@ui.page("/welcome")
async def welcome_page() -> None:
    if not page_layout("Welcome"):
        return

    tenant_id = current_tenant_id()
    if tenant_id is None:
        ui.label("No active session").classes("text-negative")
        return

    status = await get_onboarding_status(tenant_id)

    with ui.column().classes("w-full max-w-3xl q-mx-auto q-pt-md gap-4"):
        ui.label("Welcome to Daytrader").classes("text-h4 text-weight-bold")
        ui.label(
            "Three quick steps and you're trading. Each step deep-links to "
            "the page where you finish it — come back here whenever you "
            "want to see what's left."
        ).classes("text-body1 text-grey-5")

        _step_card(
            number=1,
            title="Connect a broker",
            description=(
                "Paste API keys for Binance or Alpaca. Keys are encrypted "
                "with your APP_ENCRYPTION_KEY before they hit the database."
            ),
            done=status.has_broker_creds,
            cta_label="Open broker keys",
            cta_path="/broker-credentials",
        )
        _step_card(
            number=2,
            title="Create your first persona",
            description=(
                "A persona is one trading bot — capital pool, asset class, "
                "and risk profile. You can run it in paper mode without a "
                "broker connected."
            ),
            done=status.has_persona,
            cta_label="Create a persona",
            cta_path="/personas",
        )
        _step_card(
            number=3,
            title="Wire up notifications (optional)",
            description=(
                "Drop a Slack incoming-webhook URL onto your tenant and the "
                "system will push plugin errors and a daily digest there."
            ),
            done=status.has_webhook,
            cta_label="Open admin → tenants",
            cta_path="/admin/tenants",
        )

        with ui.row().classes("w-full justify-between items-center q-pt-md"):
            ui.button(
                "Skip for now — go to home",
                on_click=lambda: ui.navigate.to("/?welcome=skip"),
            ).props("flat color=grey")
            if status.is_complete:
                ui.button(
                    "All set — open home",
                    icon="check_circle",
                    on_click=lambda: ui.navigate.to("/?welcome=skip"),
                ).props("color=positive unelevated")


def _step_card(
    *,
    number: int,
    title: str,
    description: str,
    done: bool,
    cta_label: str,
    cta_path: str,
) -> None:
    """Render one row of the welcome checklist."""
    border_color = "positive" if done else "primary"
    icon = "check_circle" if done else "circle"
    icon_color = "positive" if done else "grey-6"
    with ui.card().classes("w-full q-pa-md").style(
        f"border-left: 4px solid var(--q-{border_color})"
    ):
        with ui.row().classes("items-center gap-3 no-wrap"):
            ui.icon(icon, color=icon_color, size="lg")
            with ui.column().classes("gap-0 col-grow"):
                with ui.row().classes("items-center gap-2"):
                    ui.label(f"Step {number}").classes(
                        "text-caption text-grey-6"
                    )
                    if done:
                        ui.badge("Done", color="positive")
                ui.label(title).classes("text-h6")
                ui.label(description).classes("text-body2 text-grey-5")
            ui.button(
                cta_label,
                on_click=lambda p=cta_path: ui.navigate.to(p),
            ).props(
                f"unelevated color={'positive' if done else 'primary'}"
            )


__all__ = ["welcome_page"]
