"""Add journal_entries table.

Revision ID: 0002_journal
Revises: 0001_initial
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002_journal"
down_revision: str | None = "0001_initial"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "journal_entries",
        sa.Column("id", sa.Uuid, primary_key=True),
        sa.Column(
            "tenant_id",
            sa.Uuid,
            sa.ForeignKey("tenants.id"),
            nullable=False,
            index=True,
        ),
        sa.Column(
            "persona_id",
            sa.Uuid,
            sa.ForeignKey("personas.id"),
            nullable=True,
            index=True,
        ),
        sa.Column("event_type", sa.String(50), nullable=False, index=True),
        sa.Column("severity", sa.String(20), nullable=False, server_default="info"),
        sa.Column("summary", sa.String(500), nullable=False),
        sa.Column("detail", sa.JSON, nullable=False, server_default="{}"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            index=True,
        ),
    )


def downgrade() -> None:
    op.drop_table("journal_entries")
