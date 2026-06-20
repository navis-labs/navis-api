"""Identity — unique index on sessions.refresh_token_hash

Revision ID: 0019_session_refresh_hash
Revises: 0018_identity_session_email
Create Date: 2026-06-20

The refresh endpoint looks up `WHERE refresh_token_hash = ?` on every rotation; with
no index that's a full scan as the sessions table grows. A UNIQUE index serves the
lookup AND guarantees no two sessions can share a refresh hash (corruption guard).
issue_tokens always inserts a real (sha256) hash in a single statement, so there is
no empty-placeholder collision.
"""
from typing import Sequence, Union

from alembic import op

revision: str = "0019_session_refresh_hash"
down_revision: Union[str, None] = "0018_identity_session_email"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index(
        "uq_sessions_refresh_token_hash", "sessions", ["refresh_token_hash"], unique=True
    )


def downgrade() -> None:
    op.drop_index("uq_sessions_refresh_token_hash", table_name="sessions")
