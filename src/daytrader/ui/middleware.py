"""Auth guard for UI pages.

The single chokepoint is ``ensure_authenticated()``, called by ``page_layout``.
Pages must early-return when ``page_layout`` returns ``False``.
"""

from __future__ import annotations

from urllib.parse import quote_plus

from nicegui import ui

from ..auth.session import is_authenticated, require_role

_PUBLIC_PATHS = {"/login", "/logout"}


def ensure_authenticated(current_path: str | None = None) -> bool:
    """Return True if a user session is present; redirect to /login otherwise."""
    if is_authenticated():
        return True
    target = "/login"
    if current_path and current_path not in _PUBLIC_PATHS:
        target = f"/login?next={quote_plus(current_path)}"
    ui.navigate.to(target)
    return False


def ensure_role(roles: str | tuple[str, ...]) -> bool:
    """Return True if the session has one of the given roles; navigate home otherwise."""
    if not ensure_authenticated():
        return False
    if not require_role(roles):
        ui.notify("You don't have permission to view that page.", type="negative")
        ui.navigate.to("/")
        return False
    return True
