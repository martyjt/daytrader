"""Session helpers backed by NiceGUI's ``app.storage.user`` (signed cookie).

These functions are the only place the rest of the codebase reads or writes
session state. ``services._tenant_id()`` calls ``current_tenant_id()``;
pages call ``is_authenticated()`` / ``current_session()`` for UI gating.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from uuid import UUID

from nicegui import app

_USER_KEY = "user_id"
_TENANT_KEY = "tenant_id"
_ROLE_KEY = "role"
_EMAIL_KEY = "email"
_DISPLAY_KEY = "display_name"


@dataclass(frozen=True)
class SessionUser:
    user_id: UUID
    tenant_id: UUID
    email: str
    role: str
    display_name: str | None = None


def _storage() -> dict:
    return app.storage.user


def login_session(user: SessionUser) -> None:
    s = _storage()
    s[_USER_KEY] = str(user.user_id)
    s[_TENANT_KEY] = str(user.tenant_id)
    s[_EMAIL_KEY] = user.email
    s[_ROLE_KEY] = user.role
    s[_DISPLAY_KEY] = user.display_name or ""


def logout_session() -> None:
    s = _storage()
    for k in (_USER_KEY, _TENANT_KEY, _EMAIL_KEY, _ROLE_KEY, _DISPLAY_KEY):
        s.pop(k, None)


def is_authenticated() -> bool:
    s = _storage()
    return _USER_KEY in s and _TENANT_KEY in s


def current_session() -> SessionUser | None:
    s = _storage()
    if _USER_KEY not in s or _TENANT_KEY not in s:
        return None
    try:
        return SessionUser(
            user_id=UUID(s[_USER_KEY]),
            tenant_id=UUID(s[_TENANT_KEY]),
            email=s.get(_EMAIL_KEY, ""),
            role=s.get(_ROLE_KEY, "member"),
            display_name=s.get(_DISPLAY_KEY) or None,
        )
    except (ValueError, KeyError):
        return None


def current_user_id() -> UUID | None:
    sess = current_session()
    return sess.user_id if sess else None


def current_tenant_id() -> UUID | None:
    sess = current_session()
    return sess.tenant_id if sess else None


def require_role(roles: str | Iterable[str]) -> bool:
    """Return True if the current session has one of the given roles."""
    sess = current_session()
    if sess is None:
        return False
    if isinstance(roles, str):
        return sess.role == roles
    return sess.role in set(roles)
