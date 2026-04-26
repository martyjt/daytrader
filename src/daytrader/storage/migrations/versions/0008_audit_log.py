"""Phase 7 — tenant audit log.

Adds ``audit_log``, the durable record of meaningful tenant-affecting
actions (auth, invite, persona/broker/plugin CRUD, kill-switch). The
``(tenant_id, created_at)`` composite index supports the dominant
"last N events for tenant X" query without a sort step.

``tenant_id`` and ``user_id`` are nullable so login failures (no
authenticated user/tenant yet) and cross-tenant super-admin actions can
still be recorded.

Revision ID: 0008_audit_log
Revises: 0007_tenant_plugins
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0008_audit_log"
down_revision: str | None = "0007_tenant_plugins"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "audit_log",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "tenant_id", sa.Uuid(), sa.ForeignKey("tenants.id"), nullable=True
        ),
        sa.Column(
            "user_id", sa.Uuid(), sa.ForeignKey("users.id"), nullable=True
        ),
        sa.Column("action", sa.String(80), nullable=False),
        sa.Column("resource_type", sa.String(50), nullable=True),
        sa.Column("resource_id", sa.String(100), nullable=True),
        sa.Column("extra", sa.JSON(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_audit_log_tenant_id", "audit_log", ["tenant_id"])
    op.create_index("ix_audit_log_user_id", "audit_log", ["user_id"])
    op.create_index("ix_audit_log_action", "audit_log", ["action"])
    op.create_index(
        "ix_audit_log_resource_type", "audit_log", ["resource_type"]
    )
    op.create_index("ix_audit_log_created_at", "audit_log", ["created_at"])
    op.create_index(
        "ix_audit_log_tenant_created", "audit_log", ["tenant_id", "created_at"]
    )


def downgrade() -> None:
    op.drop_index("ix_audit_log_tenant_created", table_name="audit_log")
    op.drop_index("ix_audit_log_created_at", table_name="audit_log")
    op.drop_index("ix_audit_log_resource_type", table_name="audit_log")
    op.drop_index("ix_audit_log_action", table_name="audit_log")
    op.drop_index("ix_audit_log_user_id", table_name="audit_log")
    op.drop_index("ix_audit_log_tenant_id", table_name="audit_log")
    op.drop_table("audit_log")
