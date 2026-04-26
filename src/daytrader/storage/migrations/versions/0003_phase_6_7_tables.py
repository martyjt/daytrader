"""Phase 6-7 tables: discoveries, symbol_universes, strategy_configs, shadow_runs.

These tables back the Exploration Agent (discoveries), the Universes page
(symbol_universes), the Strategy Library (strategy_configs), and the
Shadow Tournament workflow (shadow_runs). Until now they were created
implicitly by ``Base.metadata.create_all`` on SQLite — Postgres deploys
need this migration to bring the schema up.

Revision ID: 0003_phase_6_7_tables
Revises: 0002_journal
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003_phase_6_7_tables"
down_revision: str | None = "0002_journal"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "discoveries",
        sa.Column("id", sa.Uuid, primary_key=True),
        sa.Column(
            "tenant_id",
            sa.Uuid,
            sa.ForeignKey("tenants.id"),
            nullable=False,
            index=True,
        ),
        sa.Column("candidate_name", sa.String(200), nullable=False, index=True),
        sa.Column("candidate_source", sa.String(50), nullable=False, index=True),
        sa.Column("target_symbol", sa.String(100), nullable=False, index=True),
        sa.Column("target_timeframe", sa.String(10), nullable=False),
        sa.Column("baseline_metric", sa.Float, nullable=False),
        sa.Column("candidate_metric", sa.Float, nullable=False),
        sa.Column("lift", sa.Float, nullable=False, index=True),
        sa.Column("p_value", sa.Float, nullable=True),
        sa.Column("q_value", sa.Float, nullable=True),
        sa.Column(
            "significant",
            sa.Boolean,
            nullable=False,
            server_default=sa.text("false"),
            index=True,
        ),
        sa.Column("n_folds", sa.Integer, nullable=False, server_default="5"),
        sa.Column(
            "status",
            sa.String(30),
            nullable=False,
            server_default="new",
            index=True,
        ),
        sa.Column("meta", sa.JSON, nullable=False, server_default="{}"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            index=True,
        ),
    )

    op.create_table(
        "symbol_universes",
        sa.Column("id", sa.Uuid, primary_key=True),
        sa.Column(
            "tenant_id",
            sa.Uuid,
            sa.ForeignKey("tenants.id"),
            nullable=False,
            index=True,
        ),
        sa.Column("name", sa.String(120), nullable=False, index=True),
        sa.Column("symbols", sa.JSON, nullable=False, server_default="[]"),
        sa.Column("description", sa.Text, nullable=False, server_default=""),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            index=True,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
    )

    op.create_table(
        "strategy_configs",
        sa.Column("id", sa.Uuid, primary_key=True),
        sa.Column(
            "tenant_id",
            sa.Uuid,
            sa.ForeignKey("tenants.id"),
            nullable=False,
            index=True,
        ),
        sa.Column("name", sa.String(120), nullable=False, index=True),
        sa.Column("description", sa.Text, nullable=False, server_default=""),
        sa.Column("algo_id", sa.String(100), nullable=False, index=True),
        sa.Column("symbol", sa.String(100), nullable=False),
        sa.Column("timeframe", sa.String(10), nullable=False),
        sa.Column(
            "venue",
            sa.String(50),
            nullable=False,
            server_default="binance_spot",
        ),
        sa.Column("algo_params", sa.JSON, nullable=False, server_default="{}"),
        sa.Column("tags", sa.JSON, nullable=False, server_default="[]"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            index=True,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
    )

    op.create_table(
        "shadow_runs",
        sa.Column("id", sa.Uuid, primary_key=True),
        sa.Column(
            "tenant_id",
            sa.Uuid,
            sa.ForeignKey("tenants.id"),
            nullable=False,
            index=True,
        ),
        sa.Column("tournament_id", sa.Uuid, nullable=False, index=True),
        sa.Column("primary_algo_id", sa.String(100), nullable=False, index=True),
        sa.Column("candidate_algo_id", sa.String(100), nullable=False, index=True),
        sa.Column("target_symbol", sa.String(100), nullable=False, index=True),
        sa.Column("target_timeframe", sa.String(10), nullable=False),
        sa.Column("window_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("window_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("sharpe", sa.Float, nullable=False, server_default="0"),
        sa.Column("net_return_pct", sa.Float, nullable=False, server_default="0"),
        sa.Column("max_drawdown_pct", sa.Float, nullable=False, server_default="0"),
        sa.Column("num_trades", sa.Integer, nullable=False, server_default="0"),
        sa.Column("stability_score", sa.Float, nullable=False, server_default="0"),
        sa.Column(
            "is_primary",
            sa.Boolean,
            nullable=False,
            server_default=sa.text("false"),
            index=True,
        ),
        sa.Column(
            "beat_primary",
            sa.Boolean,
            nullable=False,
            server_default=sa.text("false"),
            index=True,
        ),
        sa.Column(
            "promotion_status",
            sa.String(30),
            nullable=False,
            server_default="pending",
            index=True,
        ),
        sa.Column("meta", sa.JSON, nullable=False, server_default="{}"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            index=True,
        ),
    )


def downgrade() -> None:
    op.drop_table("shadow_runs")
    op.drop_table("strategy_configs")
    op.drop_table("symbol_universes")
    op.drop_table("discoveries")
