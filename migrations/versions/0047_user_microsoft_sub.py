"""users.microsoft_sub / users.microsoft_tid — Entra sign-in identity

Revision ID: 0047_user_microsoft_sub
Revises: 0046_microsoft_providers
Create Date: 2026-09-07

Two nullable columns beside `google_sub`, so a user can be identified by their
Microsoft account the way they can already be identified by their Google one.

`microsoft_sub` is Entra's `oid` claim: stable per user, and — unlike the email
— not something the account holder can change. It is indexed because sign-in
looks a user up by it on every Microsoft login, and that lookup is the fast path
that avoids touching the email at all.

`microsoft_tid` is the tenant the user authenticated against. It is stored
because it is the difference between a work/school account and a personal
Microsoft account, and only the former has a domain whose ownership anybody
verified. It is deliberately NOT indexed: it is read alongside an
already-resolved user and never searched.

Fully reversible and safe to run against a live table — both columns are
nullable with no default, so no rewrite and no backfill. Every existing row
keeps NULL, which is the correct value: those users have never signed in with
Microsoft.
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0047_user_microsoft_sub"
down_revision: Union[str, None] = "0046_microsoft_providers"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "users", sa.Column("microsoft_sub", sa.String(length=255), nullable=True)
    )
    op.add_column(
        "users", sa.Column("microsoft_tid", sa.String(length=64), nullable=True)
    )
    op.create_index(
        "ix_users_microsoft_sub", "users", ["microsoft_sub"], unique=False
    )


def downgrade() -> None:
    op.drop_index("ix_users_microsoft_sub", table_name="users")
    op.drop_column("users", "microsoft_tid")
    op.drop_column("users", "microsoft_sub")
