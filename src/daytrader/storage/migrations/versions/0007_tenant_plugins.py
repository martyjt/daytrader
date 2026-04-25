"""Phase 6 — tenant-uploaded algorithm plugins.

Adds ``tenant_plugins``, the durable index of what each tenant has
installed via the sandboxed plugin upload flow. The on-disk file under
``plugins/uploads/<tenant_id>/<filename>`` is the artifact; this row
records who uploaded it, when, the file's sha256, and whether it's
currently enabled.

The unique constraint on ``(tenant_id, algorithm_id)`` enforces one row
per algorithm id per tenant — re-uploads replace the existing row in the
installer rather than creating duplicates.

Revision ID: 0007_tenant_plugins
Revises: 0006_strategy_discovery_link
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0007_tenant_plugins"
down_revision: Union[str, None] = "0006_strategy_discovery_link"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "tenant_plugins",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("tenant_id", sa.Uuid(), sa.ForeignKey("tenants.id"), nullable=False),
        sa.Column("algorithm_id", sa.String(100), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("filename", sa.String(255), nullable=False),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("manifest_json", sa.Text(), nullable=False),
        sa.Column("warmup_bars", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("uploaded_by", sa.Uuid(), sa.ForeignKey("users.id"), nullable=True),
        sa.Column("is_enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column(
            "uploaded_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint(
            "tenant_id", "algorithm_id", name="uq_tenant_plugins_tenant_algorithm"
        ),
    )
    op.create_index(
        "ix_tenant_plugins_tenant_id", "tenant_plugins", ["tenant_id"]
    )


def downgrade() -> None:
    op.drop_index("ix_tenant_plugins_tenant_id", table_name="tenant_plugins")
    op.drop_table("tenant_plugins")
