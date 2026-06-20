"""Identity hardening — scale indexes for audit + token purge

Revision ID: 0016_identity_hardening
Revises: 0015_ml_embeddings
Create Date: 2026-06-19

Two hot paths lacked a supporting index:
  * audit_logs is read `WHERE organization_id AND occurred_at >= since ORDER BY
    occurred_at DESC` — a composite (organization_id, occurred_at DESC) makes that
    a single index range scan instead of filter-then-sort over a per-org partition.
  * revoked_tokens.purge_expired() scans `WHERE expires_at < now` — index it so the
    reaper is a range scan, not a full table scan, as the denylist grows.
"""
from typing import Sequence, Union

from alembic import op

revision: str = "0016_identity_hardening"
down_revision: Union[str, None] = "0015_ml_embeddings"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index(
        "ix_al_org_time",
        "audit_logs",
        ["organization_id", "occurred_at"],
        postgresql_ops={"occurred_at": "DESC"},
    )
    op.create_index("ix_rt_expires", "revoked_tokens", ["expires_at"])


def downgrade() -> None:
    op.drop_index("ix_rt_expires", table_name="revoked_tokens")
    op.drop_index("ix_al_org_time", table_name="audit_logs")
