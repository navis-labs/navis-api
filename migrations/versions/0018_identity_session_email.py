"""Identity — session expiry index + case-insensitive email uniqueness

Revision ID: 0018_identity_session_email
Revises: 0017_membership_role
Create Date: 2026-06-19

  * sessions: index expires_at so the cleanup reaper (`WHERE expires_at < now()`)
    is a range scan, not a full scan as the table grows (10M users x N devices).
  * users: a UNIQUE index on lower(email) enforces at the DATABASE what the app now
    normalizes — `John@X.com` and `john@x.com` can never become two identities, even
    if some future write path forgets to normalize. Defense-in-depth.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0018_identity_session_email"
down_revision: Union[str, None] = "0017_membership_role"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index("ix_sessions_expires_at", "sessions", ["expires_at"])
    op.create_index(
        "uq_users_email_lower",
        "users",
        [sa.text("lower(email)")],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("uq_users_email_lower", table_name="users")
    op.drop_index("ix_sessions_expires_at", table_name="sessions")
