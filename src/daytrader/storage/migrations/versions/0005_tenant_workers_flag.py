"""Phase 2 — per-tenant background workers flag.

Adds ``tenants.background_workers_enabled`` so super admins can pause
expensive scheduled workloads for specific tenants without disabling them
globally.

Revision ID: 0005_tenant_workers_flag
Revises: 0004_user_auth
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005_tenant_workers_flag"
down_revision: str | None = "0004_user_auth"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "tenants",
        sa.Column(
            "background_workers_enabled",
            sa.Boolean,
            nullable=False,
            server_default=sa.true(),
        ),
    )


def downgrade() -> None:
    op.drop_column("tenants", "background_workers_enabled")
