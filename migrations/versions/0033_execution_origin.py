"""Execution — record WHICH surface proposed an action (chat vs proactive)

Revision ID: 0033_execution_origin
Revises: 0032_signal_importance
Create Date: 2026-08-09

Adds `execution_requests.origin`.

Navis has two ways an action comes into being, and they need different review
surfaces:

  chat       the founder asked for it ("schedule with Chrissy tomorrow at 5").
             They are looking at the proposal in the conversation; Chat owns it.
  proactive  Navis discovered it from signals the founder has not seen
             ("John has been waiting 4 days"). The Action Center is the queue
             for exactly these.

Execution is identical for both — same ExecutionRequest, same approval, same
worker. Only the ownership of REVIEW differs. Without this column
`GET /actions` returned every REQUESTED row, so a meeting proposed in chat also
appeared on the Action Center board: two surfaces able to approve one row, and a
"proactive" queue full of things the user had just typed.

Backfill choice — existing rows default to 'chat', which keeps them OUT of the
proactive queue. Both surfaces wrote rows before this column existed and nothing
distinguishes them retroactively, so the default is picked by consequence:
wrongly hiding an old proactive row costs one missed nudge, wrongly showing a
chat row resurrects the duplicate-surface bug this column exists to remove.

Additive and reversible; no existing column is touched.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0033_execution_origin"
down_revision: Union[str, Sequence[str], None] = "0032_signal_importance"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "execution_requests",
        sa.Column(
            "origin",
            sa.String(length=16),
            nullable=False,
            server_default="chat",
        ),
    )
    # The Action Center's feed is `WHERE status='requested' AND origin='proactive'`,
    # so the index is on the pair the query actually filters by, tenant-first.
    op.create_index(
        "ix_execution_org_origin_status",
        "execution_requests",
        ["organization_id", "origin", "status"],
    )


def downgrade() -> None:
    op.drop_index("ix_execution_org_origin_status", table_name="execution_requests")
    op.drop_column("execution_requests", "origin")
