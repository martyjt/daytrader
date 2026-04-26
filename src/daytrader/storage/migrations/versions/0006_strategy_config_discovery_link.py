"""Phase 5 — link StrategyConfig back to its source Discovery.

Adds ``strategy_configs.source_discovery_id`` so the trading loop can
hydrate a discovered feature (FRED series value, sentiment score,
cross-asset price) into ``AlgorithmContext.features`` at bar time.

Nullable: existing strategy configs predate Phase 5 and have no source
discovery; only configs created by ``promote_discovery`` set the FK.

Revision ID: 0006_strategy_discovery_link
Revises: 0005_tenant_workers_flag

(Revision string shortened to fit alembic_version's VARCHAR(32) on Postgres;
the filename retains the original descriptive name for grep-ability.)
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0006_strategy_discovery_link"
down_revision: str | None = "0005_tenant_workers_flag"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "strategy_configs",
        sa.Column(
            "source_discovery_id",
            sa.Uuid(),
            sa.ForeignKey("discoveries.id"),
            nullable=True,
        ),
    )
    op.create_index(
        "ix_strategy_configs_source_discovery_id",
        "strategy_configs",
        ["source_discovery_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_strategy_configs_source_discovery_id",
        table_name="strategy_configs",
    )
    op.drop_column("strategy_configs", "source_discovery_id")
