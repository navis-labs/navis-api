"""memberships.role NOT NULL — make the tenancy invariant explicit.

TASK 6.4.4 (F-1). `memberships` is the tenancy join row: it says WHICH
organization a user belongs to and WITH WHAT ROLE. Migration 0000 created the
role column `nullable=True`, while the ORM has always declared it
`Mapped[Role]` (non-optional) with `default=Role.FOUNDER`. Every write path
supplies a role — `IdentityService.signup`, `IdentityService.google_signin`
and `MemberService.invite` are the only three, and all three pass one
explicitly — so the column has been non-null in practice and nullable in
schema for the whole life of the table.

That gap was not free. A NULL role resolves to the empty permission set in
`authz._principal_from_jwt` (fail-closed, correct) but reached `/auth/me` as
`Role("")`, which raises ValueError and returns 500 — an authenticated user
with a valid seat could not read their own identity. The route now resolves
the role defensively as well; this migration removes the ability to create the
state in the first place.

SAFETY — VERIFIED AGAINST PRODUCTION BEFORE WRITING THIS FILE. A read-only
probe of the live database on 2026-08-20 returned:

    SELECT count(*) FROM memberships WHERE role IS NULL   ->  0
    SELECT role, count(*) FROM memberships GROUP BY role  ->  founder: 3

There is no row to violate the constraint and none to back-fill. `upgrade`
still re-checks at run time and raises with the offending count rather than
letting Postgres fail with a bare constraint error, so any environment that
DOES hold a NULL gets told exactly what to fix.

NO SERVER DEFAULT, DELIBERATELY. A `DEFAULT 'founder'` would make an INSERT
that forgets the role silently mint an org OWNER — the highest-privilege role
in the system — instead of failing. The Python-side `default=Role.FOUNDER`
already covers the ORM path where the caller is a known code path; a raw
INSERT that omits the column should be a loud error, not a free founder seat.

Cheap: the constraint is validated by one sequential scan of a table with a
handful of rows per deployment, taking a brief ACCESS EXCLUSIVE lock. There is
no table rewrite (SET NOT NULL does not rewrite).

Reversible: downgrade drops the constraint back to nullable. No data moves in
either direction.

Revision ID: 0043_membership_role_not_null
Revises: 0042_execution_integration
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0043_membership_role_not_null"
down_revision = "0042_execution_integration"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    nulls = bind.execute(
        sa.text("SELECT count(*) FROM memberships WHERE role IS NULL")
    ).scalar_one()
    if nulls:
        # Refuse rather than guess. There is no correct role to invent for an
        # existing seat: assigning one grants privilege that was never granted,
        # and assigning the weakest one silently demotes a real member. An
        # operator must decide, per row, which it is.
        raise RuntimeError(
            f"{nulls} membership row(s) have a NULL role; assign a role to each "
            "before applying 0043 (a migration must not invent someone's privilege)"
        )
    op.alter_column("memberships", "role", existing_type=sa.Enum(name="role"), nullable=False)


def downgrade() -> None:
    op.alter_column("memberships", "role", existing_type=sa.Enum(name="role"), nullable=True)
