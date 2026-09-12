"""Remove the Phase 7 Devtool verification organization and user.

WHAT THIS IS. A DATA migration, not a schema one. It deletes exactly two rows —
one `organizations` row and one `users` row — created on 2026-09-07 by an
end-to-end verification run of the Developer Platform against this database. It
changes no table, no constraint, no index, no role and no permission.

WHY A MIGRATION RATHER THAN A psql SESSION. The deletion has to happen exactly
once, against exactly one database, with a record that it happened. A migration
is the repository's existing mechanism for all three. It is also reviewable
before it runs, which an operator's ad-hoc DELETE is not.

WHY DELETING THE ORGANIZATION IS ENOUGH. Every table that references
`organizations` does so `ON DELETE CASCADE` — forty-four of them, verified
against the live schema rather than assumed, including `memberships`,
`service_accounts`, `api_keys`, `audit_logs`, `organization_entitlements`,
`organization_onboarding`, `organization_lifecycles` and `state_snapshots`. So
the organization row is the single handle for all 27 dependent rows this
verification run produced, and no per-table DELETE list is needed. A hand-written
list would additionally be a list that goes stale the next time a table gains an
`organization_id`.

The `users` row is deleted separately because it is NOT a child of the
organization: `memberships` is what joins them, and it cascades from both sides.

AUDIT ROWS ARE CASCADED, DELIBERATELY, AND THAT IS THE APPLICATION'S OWN RULE.
`audit_logs.organization_id` is declared `ON DELETE CASCADE` in this schema. The
application therefore already treats an audit entry as dependent on the
organization it describes — a tenant's history does not outlive the tenant. All
seven entries for this organization describe actions taken on it by its own user
(`signup`, `login` x3, `org.plan.enterprise`, `service_account_create`,
`service_account_revoke`), and a query confirmed that NO audit row belonging to
any other organization references these identifiers. Nothing here overrides audit
retention to tidy the database; the cascade is the schema's declared intent and
no real tenant's trail is touched.

WHY THE GUARDS ABORT INSTEAD OF SKIPPING. Each identifier is checked against the
value it is supposed to name — the user's email and the organization's name —
before anything is deleted. A UUID that exists but names something else means the
database is not the one this migration was written for, and the only safe action
is to fail the transaction. Deleting a row because its id matched, while its
email did not, is precisely the accident this migration exists to avoid.

IDEMPOTENT. Rows already absent are a no-op, not an error, so a re-run after a
partially applied deploy is safe.

NOT REVERSIBLE. `downgrade` raises. These are two deleted rows and their
cascaded children; no downgrade can restore data, and pretending otherwise with
an empty `pass` would let `alembic downgrade` report success for a restore that
did not happen.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# 30 characters. `alembic_version.version_num` is `varchar(32)`, and a longer id
# fails on the version bump AFTER `upgrade()` has already run — the transaction
# rolls the deletes back, so nothing is lost, but the migration cannot be
# applied at all until the id fits. `test_the_revision_id_fits_the_version_column`
# pins this.
revision: str = "0047_cleanup_verification_data"
down_revision: Union[str, None] = "0046_service_account_agents"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# The four records named in the Phase 7 verification report. Exact identifiers,
# never patterns: a `LIKE '%devtool-verify%'` would match any future account
# someone names similarly, and the whole point of this migration is that it can
# only ever affect these rows.
VERIFICATION_USER_ID = "9e98e790-80bf-4bdf-a34a-c8e4a055d631"
VERIFICATION_ORG_ID = "73a59468-c7d2-4233-a232-de4c9466eb85"

# Asserted, not searched by. The id selects the row; these prove it is the right
# one before it is deleted.
EXPECTED_EMAIL = "devtool-verify+1788792715@navislabs.in"
EXPECTED_ORG_NAME = "Devtool Verification Org"


def upgrade() -> None:
    bind = op.get_bind()

    # ── Guard 1: the user id, if present, must be the verification user ──
    email = bind.execute(
        sa.text("SELECT email FROM users WHERE id = :id"),
        {"id": VERIFICATION_USER_ID},
    ).scalar()
    if email is not None and email != EXPECTED_EMAIL:
        raise RuntimeError(
            f"Refusing to delete: user {VERIFICATION_USER_ID} exists but its email is "
            f"not the expected verification address. This database is not the one this "
            f"migration was written for."
        )

    # ── Guard 2: the organization id, if present, must be the verification org ──
    org_name = bind.execute(
        sa.text("SELECT name FROM organizations WHERE id = :id"),
        {"id": VERIFICATION_ORG_ID},
    ).scalar()
    if org_name is not None and org_name != EXPECTED_ORG_NAME:
        raise RuntimeError(
            f"Refusing to delete: organization {VERIFICATION_ORG_ID} exists but is named "
            f"{org_name!r}, not {EXPECTED_ORG_NAME!r}."
        )

    # ── Guard 3: no OTHER person is a member of this organization ────────
    # Deleting it cascades every membership. If a real user ever joined, that
    # user would silently lose a seat, so the migration stops instead.
    strangers = bind.execute(
        sa.text(
            "SELECT count(*) FROM memberships "
            "WHERE organization_id = :org AND user_id <> :user"
        ),
        {"org": VERIFICATION_ORG_ID, "user": VERIFICATION_USER_ID},
    ).scalar()
    if strangers:
        raise RuntimeError(
            f"Refusing to delete: {strangers} membership(s) held by other users exist in "
            f"organization {VERIFICATION_ORG_ID}."
        )

    # ── Guard 4: the verification user holds no OTHER organization ───────
    # If it did, deleting the user would cascade a membership in a real tenant.
    elsewhere = bind.execute(
        sa.text(
            "SELECT count(*) FROM memberships "
            "WHERE user_id = :user AND organization_id <> :org"
        ),
        {"user": VERIFICATION_USER_ID, "org": VERIFICATION_ORG_ID},
    ).scalar()
    if elsewhere:
        raise RuntimeError(
            f"Refusing to delete: user {VERIFICATION_USER_ID} is a member of {elsewhere} "
            f"other organization(s)."
        )

    # ── Delete ───────────────────────────────────────────────────────────
    # Organization first: it cascades the memberships row, so the user is left
    # unattached before its own delete. Both are `WHERE id = ...` against a
    # primary key, so neither statement can reach a second row.
    org_deleted = bind.execute(
        sa.text("DELETE FROM organizations WHERE id = :id"),
        {"id": VERIFICATION_ORG_ID},
    ).rowcount
    user_deleted = bind.execute(
        sa.text("DELETE FROM users WHERE id = :id"),
        {"id": VERIFICATION_USER_ID},
    ).rowcount

    print(
        f"[0047] removed verification data: "
        f"{org_deleted} organization, {user_deleted} user "
        f"(dependent rows removed by ON DELETE CASCADE)"
    )


def downgrade() -> None:
    raise RuntimeError(
        "0047 deleted verification data and cannot be reversed. Restoring these rows "
        "would mean recreating an organization, a user, a membership, a service "
        "account, a credential and their audit trail from values this migration does "
        "not hold. Restore from a database backup instead."
    )
