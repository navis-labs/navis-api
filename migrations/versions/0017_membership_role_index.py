"""Identity — composite (organization_id, role) index on memberships

Revision ID: 0017_membership_role
Revises: 0016_identity_hardening
Create Date: 2026-06-19

The last-founder guard runs `SELECT id FROM memberships WHERE organization_id=? AND
role='founder' FOR UPDATE` on every member remove/role-change. A composite
(organization_id, role) index turns that into a narrow index range scan instead of
scanning every member of a large org to filter by role.
"""
from typing import Sequence, Union

from alembic import op

revision: str = "0017_membership_role"
down_revision: Union[str, None] = "0016_identity_hardening"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index("ix_membership_org_role", "memberships", ["organization_id", "role"])


def downgrade() -> None:
    op.drop_index("ix_membership_org_role", table_name="memberships")
