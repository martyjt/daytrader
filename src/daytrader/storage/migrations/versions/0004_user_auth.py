"""Phase 1 — auth foundation: extend users + add user_invites.

Adds ``password_hash``, ``role``, ``is_active``, ``last_login_at`` to ``users``
and the ``email`` unique index. Creates the ``user_invites`` table for
invite-only registration. Existing rows are seeded a placeholder password
hash; ``seed_default_tenant`` will replace it on first boot.

Revision ID: 0004_user_auth
Revises: 0003_phase_6_7_tables
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004_user_auth"
down_revision: str | None = "0003_phase_6_7_tables"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "users",
        sa.Column("password_hash", sa.String(255), nullable=False, server_default=""),
    )
    op.add_column(
        "users",
        sa.Column("role", sa.String(20), nullable=False, server_default="member"),
    )
    op.add_column(
        "users",
        sa.Column(
            "is_active",
            sa.Boolean,
            nullable=False,
            server_default=sa.true(),
        ),
    )
    op.add_column(
        "users",
        sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    op.create_table(
        "user_invites",
        sa.Column("id", sa.Uuid, primary_key=True),
        sa.Column("token", sa.String(64), nullable=False),
        sa.Column("email", sa.String(255), nullable=False, index=True),
        sa.Column(
            "tenant_id",
            sa.Uuid,
            sa.ForeignKey("tenants.id"),
            nullable=True,
            index=True,
        ),
        sa.Column("role", sa.String(20), nullable=False, server_default="member"),
        sa.Column(
            "invited_by",
            sa.Uuid,
            sa.ForeignKey("users.id"),
            nullable=True,
        ),
        sa.Column("used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
    )
    op.create_index("ix_user_invites_token", "user_invites", ["token"], unique=True)


def downgrade() -> None:
    op.drop_index("ix_user_invites_token", table_name="user_invites")
    op.drop_table("user_invites")
    op.drop_index("ix_users_email", table_name="users")
    op.drop_column("users", "last_login_at")
    op.drop_column("users", "is_active")
    op.drop_column("users", "role")
    op.drop_column("users", "password_hash")
