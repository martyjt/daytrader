"""Login, logout, and invite-redemption pages.

These pages do NOT call ``page_layout`` — they live outside the auth
guard. Successful login redirects to the home page.
"""

from __future__ import annotations

from urllib.parse import quote_plus

from nicegui import ui

from ...auth.invites import get_invite, is_expired, redeem_invite
from ...auth.service import AuthError, authenticate
from ...auth.session import login_session, logout_session


def _centered_card():
    ui.dark_mode(True)
    ui.colors(primary="#5c7cfa", secondary="#495057", accent="#22b8cf")
    return ui.card().classes("absolute-center q-pa-lg").style("min-width: 360px")


@ui.page("/login")
async def login_page(next: str = "/") -> None:
    with _centered_card():
        with ui.row().classes("items-center gap-2"):
            ui.icon("show_chart", size="md").classes("text-primary")
            ui.label("Daytrader").classes("text-h5 text-weight-bold")
        ui.label("Sign in").classes("text-subtitle1 text-grey-5")

        email = ui.input("Email").props("outlined dense").classes("w-full")
        password = (
            ui.input("Password", password=True, password_toggle_button=True)
            .props("outlined dense")
            .classes("w-full")
        )
        error = ui.label("").classes("text-negative text-caption")

        async def submit() -> None:
            try:
                user = await authenticate(email.value or "", password.value or "")
            except AuthError as e:
                error.text = str(e)
                return
            login_session(user)
            target = next if next.startswith("/") else "/"
            ui.navigate.to(target)

        password.on("keydown.enter", submit)
        ui.button("Sign in", on_click=submit).props("color=primary").classes(
            "w-full q-mt-md"
        )

        ui.separator().classes("q-my-md")
        ui.label(
            "Need access? Ask the system administrator for an invite link."
        ).classes("text-caption text-grey-6")


@ui.page("/logout")
async def logout_page() -> None:
    logout_session()
    ui.navigate.to("/login")


@ui.page("/invite/{token}")
async def invite_page(token: str) -> None:
    invite = await get_invite(token)
    with _centered_card():
        with ui.row().classes("items-center gap-2"):
            ui.icon("mail_outline", size="md").classes("text-primary")
            ui.label("Daytrader").classes("text-h5 text-weight-bold")

        if invite is None:
            ui.label("Invite not found").classes("text-h6 text-negative")
            ui.label(
                "This invite link is invalid. Please request a new one."
            ).classes("text-caption text-grey-6")
            return
        if invite.used_at is not None:
            ui.label("Invite already used").classes("text-h6 text-warning")
            ui.button(
                "Sign in",
                on_click=lambda: ui.navigate.to("/login"),
            ).props("color=primary")
            return
        if is_expired(invite):
            ui.label("Invite expired").classes("text-h6 text-warning")
            ui.label(
                "Please ask the administrator for a new invite."
            ).classes("text-caption text-grey-6")
            return

        ui.label(f"Activate account: {invite.email}").classes("text-subtitle1")
        ui.label(f"Role: {invite.role}").classes("text-caption text-grey-5")

        display_name = (
            ui.input("Display name").props("outlined dense").classes("w-full")
        )
        password = (
            ui.input("Password", password=True, password_toggle_button=True)
            .props("outlined dense")
            .classes("w-full")
        )
        confirm = (
            ui.input(
                "Confirm password", password=True, password_toggle_button=True
            )
            .props("outlined dense")
            .classes("w-full")
        )
        error = ui.label("").classes("text-negative text-caption")

        async def submit() -> None:
            if not password.value or len(password.value) < 8:
                error.text = "Password must be at least 8 characters"
                return
            if password.value != confirm.value:
                error.text = "Passwords do not match"
                return
            try:
                user_id = await redeem_invite(
                    token=token,
                    password=password.value,
                    display_name=display_name.value or None,
                )
            except AuthError as e:
                error.text = str(e)
                return
            ui.navigate.to(f"/login?next={quote_plus('/')}&activated={user_id}")

        ui.button("Activate", on_click=submit).props("color=primary").classes(
            "w-full q-mt-md"
        )
