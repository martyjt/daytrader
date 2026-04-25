"""Role constants. Keep these as plain strings to match the DB column."""

from __future__ import annotations

ROLE_SUPER_ADMIN = "super_admin"
ROLE_OWNER = "owner"
ROLE_MEMBER = "member"

ROLES: tuple[str, ...] = (ROLE_SUPER_ADMIN, ROLE_OWNER, ROLE_MEMBER)
