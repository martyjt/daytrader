"""Initial schema: tenants, users, strategies, personas, signals,
positions, orders, broker_credentials.

Revision ID: 0001_initial
Revises: -
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001_initial"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "tenants",
        sa.Column("id", sa.Uuid, primary_key=True),
        sa.Column("name", sa.String(255), nullable=False, unique=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
    )

    op.create_table(
        "users",
        sa.Column("id", sa.Uuid, primary_key=True),
        sa.Column(
            "tenant_id",
            sa.Uuid,
            sa.ForeignKey("tenants.id"),
            nullable=False,
            index=True,
        ),
        sa.Column("email", sa.String(255), nullable=False),
        sa.Column("display_name", sa.String(255), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
    )

    op.create_table(
        "strategies",
        sa.Column("id", sa.Uuid, primary_key=True),
        sa.Column(
            "tenant_id",
            sa.Uuid,
            sa.ForeignKey("tenants.id"),
            nullable=False,
            index=True,
        ),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text, server_default=""),
        sa.Column("config", sa.JSON, nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
    )

    op.create_table(
        "personas",
        sa.Column("id", sa.Uuid, primary_key=True),
        sa.Column(
            "tenant_id",
            sa.Uuid,
            sa.ForeignKey("tenants.id"),
            nullable=False,
            index=True,
        ),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("mode", sa.String(50), nullable=False, server_default="paper"),
        sa.Column(
            "asset_class", sa.String(50), nullable=False, server_default="crypto"
        ),
        sa.Column(
            "base_currency", sa.String(10), nullable=False, server_default="USDT"
        ),
        sa.Column("initial_capital", sa.Numeric(20, 8), nullable=False),
        sa.Column("current_equity", sa.Numeric(20, 8), nullable=False),
        sa.Column(
            "risk_profile",
            sa.String(50),
            nullable=False,
            server_default="balanced",
        ),
        sa.Column(
            "strategy_id",
            sa.Uuid,
            sa.ForeignKey("strategies.id"),
            nullable=True,
        ),
        sa.Column("universe", sa.JSON, server_default="[]"),
        sa.Column("meta", sa.JSON, server_default="{}"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
    )

    op.create_table(
        "signals",
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
            nullable=False,
            index=True,
        ),
        sa.Column("symbol_key", sa.String(100), nullable=False, index=True),
        sa.Column("score", sa.Float, nullable=False),
        sa.Column("confidence", sa.Float, nullable=False),
        sa.Column("source", sa.String(255), nullable=False),
        sa.Column("reason", sa.Text, server_default=""),
        sa.Column("attribution", sa.JSON, nullable=True),
        sa.Column("meta", sa.JSON, server_default="{}"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
    )

    op.create_table(
        "positions",
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
            nullable=False,
            index=True,
        ),
        sa.Column("symbol_key", sa.String(100), nullable=False, index=True),
        sa.Column("quantity", sa.Numeric(20, 8), nullable=False),
        sa.Column("avg_entry_price", sa.Numeric(20, 8), nullable=False),
        sa.Column("realized_pnl", sa.Numeric(20, 8), server_default="0"),
        sa.Column("unrealized_pnl", sa.Numeric(20, 8), server_default="0"),
        sa.Column("stop_loss", sa.Numeric(20, 8), nullable=True),
        sa.Column("take_profit", sa.Numeric(20, 8), nullable=True),
        sa.Column("opened_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("closed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("meta", sa.JSON, server_default="{}"),
    )

    op.create_table(
        "orders",
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
            nullable=False,
            index=True,
        ),
        sa.Column("symbol_key", sa.String(100), nullable=False),
        sa.Column("side", sa.String(10), nullable=False),
        sa.Column("type", sa.String(20), nullable=False),
        sa.Column("quantity", sa.Numeric(20, 8), nullable=False),
        sa.Column("price", sa.Numeric(20, 8), nullable=True),
        sa.Column("stop_price", sa.Numeric(20, 8), nullable=True),
        sa.Column("status", sa.String(30), nullable=False, server_default="pending"),
        sa.Column("filled_quantity", sa.Numeric(20, 8), server_default="0"),
        sa.Column("avg_fill_price", sa.Numeric(20, 8), nullable=True),
        sa.Column("reason", sa.Text, server_default=""),
        sa.Column("meta", sa.JSON, server_default="{}"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
    )

    op.create_table(
        "broker_credentials",
        sa.Column("id", sa.Uuid, primary_key=True),
        sa.Column(
            "tenant_id",
            sa.Uuid,
            sa.ForeignKey("tenants.id"),
            nullable=False,
            index=True,
        ),
        sa.Column("broker_name", sa.String(100), nullable=False),
        sa.Column("credential_data", sa.Text, nullable=False),
        sa.Column("is_testnet", sa.Boolean, server_default=sa.text("true")),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
    )


def downgrade() -> None:
    op.drop_table("broker_credentials")
    op.drop_table("orders")
    op.drop_table("positions")
    op.drop_table("signals")
    op.drop_table("personas")
    op.drop_table("strategies")
    op.drop_table("users")
    op.drop_table("tenants")
