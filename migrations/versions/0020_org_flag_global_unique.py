"""Organization — partial unique index for global feature flags

Revision ID: 0020_org_flag_global_unique
Revises: 0019_session_refresh_hash
Create Date: 2026-06-22

UNIQUE(organization_id, flag) does NOT dedupe global rows: Postgres treats NULLs as
DISTINCT, so two (NULL, flag) rows satisfy the constraint (proven: two inserts → 2 rows).
Concurrent set_flag(None, flag) can therefore create duplicate globals, after which flag
evaluation across ALL orgs is non-deterministic (row-order dependent) and set_flag updates
only one duplicate. A partial UNIQUE index over `flag WHERE organization_id IS NULL`
guarantees at most one global default per flag. Additive — the existing composite unique
still governs per-org rows, so non-NULL semantics are unchanged.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0020_org_flag_global_unique"
down_revision: Union[str, None] = "0019_session_refresh_hash"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index(
        "uq_org_flag_global",
        "organization_feature_flags",
        ["flag"],
        unique=True,
        postgresql_where=sa.text("organization_id IS NULL"),
    )


def downgrade() -> None:
    op.drop_index("uq_org_flag_global", table_name="organization_feature_flags")
