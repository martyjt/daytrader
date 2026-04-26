"""Phase 1 auth: password hashing, session, invites, user service.

Public surface re-exports the helpers most call sites need. Lower-level
modules (``password``, ``session``, ``invites``, ``service``) can be
imported directly when richer access is required.
"""

from __future__ import annotations

from .password import hash_password, verify_password
from .roles import ROLE_MEMBER, ROLE_OWNER, ROLE_SUPER_ADMIN, ROLES
from .session import (
    SessionUser,
    current_session,
    current_tenant_id,
    current_user_id,
    is_authenticated,
    login_session,
    logout_session,
    require_role,
)

__all__ = [
    "ROLES",
    "ROLE_MEMBER",
    "ROLE_OWNER",
    "ROLE_SUPER_ADMIN",
    "SessionUser",
    "current_session",
    "current_tenant_id",
    "current_user_id",
    "hash_password",
    "is_authenticated",
    "login_session",
    "logout_session",
    "require_role",
    "verify_password",
]
