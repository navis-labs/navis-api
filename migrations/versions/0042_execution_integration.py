"""execution_requests.integration_id — bind an action to ONE connected account.

TASK 6.5.1 (T-11). `organization_id` records WHOSE action this is; it never
recorded which of that organization's accounts to perform it AS. An organization
may legitimately connect Gmail A and Gmail B, and the worker resolved the
credential from (organization, provider) alone — `resolve_gmail_send_integration`
filters by send capability and then breaks ties by RECENCY. A reply to a message
that arrived at B could therefore be sent from A: right tenant, wrong identity,
wrong thread, no error. `signal_events.integration_id` has existed since the
first migration for exactly this reason; the write path never had it.

ADDITIVE AND NON-DESTRUCTIVE. One nullable column plus its index and FK. No
existing row is read, rewritten or deleted, and no default is applied, so the
table is not rewritten and every current reader is unaffected.

NO BACKFILL, DELIBERATELY. The account a historical execution used is not
recoverable: the credential was chosen at run time by a resolver whose input
(the set of active integrations and their `created_at` order) has since changed.
Deriving one now would be FABRICATING lineage on an audit record — the same
reasoning migration 0037 gives for not guessing which id namespace a historical
provenance value belonged to. NULL means "not recorded", and the application
treats it as such rather than as "any account will do": `sync`/`worker` fall back
to the org-scoped resolver only when the organization has exactly ONE candidate
account for that provider, and refuse otherwise.

ON DELETE SET NULL mirrors `signal_events.integration_id`. Disconnecting an
integration soft-deactivates it today, but if a row is ever hard-deleted the
execution history must survive with the reference cleared, not cascade away.

Reversible: downgrade drops the index and the column. Nothing else references it,
and because there was no backfill there is no data to restore.

Revision ID: 0042_execution_integration
Revises: 0041_brief_policy_version
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0042_execution_integration"
down_revision = "0041_brief_policy_version"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "execution_requests",
        sa.Column("integration_id", sa.dialects.postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.create_foreign_key(
        "fk_execution_requests_integration_id",
        "execution_requests",
        "integrations",
        ["integration_id"],
        ["id"],
        ondelete="SET NULL",
    )
    # Supports the worker's ownership re-check ("does this integration belong to
    # this org?") and operator queries ("what did we send from account X?").
    op.create_index(
        "ix_execution_requests_integration_id",
        "execution_requests",
        ["integration_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_execution_requests_integration_id", table_name="execution_requests")
    op.drop_constraint(
        "fk_execution_requests_integration_id", "execution_requests", type_="foreignkey"
    )
    op.drop_column("execution_requests", "integration_id")
