"""Phase 11 — per-tenant notification webhook URL.

Adds ``tenants.notification_webhook_url``: a Fernet-encrypted Slack
incoming-webhook URL. Null/empty means the tenant has no notifications
wired up and the active notifier silently drops messages for them.

Storing it on ``TenantModel`` (rather than a side-table) keeps the
single-channel-per-tenant rule explicit. If a future phase needs
multiple channels per tenant we'd promote this to a notification_targets
table; until then the column is the simplest thing that works.

Revision ID: 0009_tenant_notification_webhook
Revises: 0008_audit_log
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0009_tenant_notification_webhook"
down_revision: str | None = "0008_audit_log"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "tenants",
        sa.Column("notification_webhook_url", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("tenants", "notification_webhook_url")
