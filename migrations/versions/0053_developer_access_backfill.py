"""Activate the Developer surface: backfill `memberships.developer_role` for founders.

WHAT THIS IS FOR. `0051_surface_rbac_columns` added `memberships.developer_role`
and deliberately shipped no reader, so until now `memberships.role` was the sole
authorization input and an Application administrator silently held
`manage_service_accounts` and `record_agent_runs`. P1-6 makes the column
authoritative: `identity.authz._principal_from_jwt` now resolves it through
`permissions.effective_permissions`, and the Application role maps no longer
carry any Developer permission.

THIS MIGRATION IS DATA ONLY. No column is added, dropped, renamed or retyped; no
constraint is created or altered; no index is touched. `ck_memberships_developer_role`
from 0051 already permits exactly the four values this backfill can write, and
`role_permissions.role` is already `varchar(40)`, which is wide enough for the
namespaced Developer grant keys (`developer:owner`). There was nothing left to
build, which is the whole reason the split needed no new structure.

WHY ONLY FOUNDERS, AND WHY THIS IS NOT "INVENTING A DEFAULT".
The rule is one line and reads off an explicit existing fact:

    role = 'founder'  ->  developer_role = 'owner'
    every other role  ->  developer_role stays NULL (no Developer access)

Handing a Developer role to every existing member would be inventing one, and it
is exactly the silent inheritance P1-6 exists to remove. But leaving EVERY row
NULL is not the conservative choice it looks like — it is an unrecoverable one.
Assigning a Developer role requires the actor to already hold the permissions
being granted (`MemberService._ensure_grantable`), and since the surface split no
Application role holds any. An organization with no Developer owner therefore has
nobody who could ever grant Developer access, to anyone, including themselves:
the surface would be permanently unreachable for that tenant with no in-product
recovery. That is the same state `_count_founders_locked` exists to prevent, and
the founder — the person who created the organization — is the deterministic,
already-recorded answer to "who bootstraps it".

Every `signup` and `google_signin` path now writes `developer_role='owner'`
alongside `role=FOUNDER` for the same reason, so this backfill and the live code
agree about what a founder is.

WHAT CHANGES FOR A REAL USER, MEASURED RATHER THAN ASSUMED. At the time of
writing, production holds 3 memberships, all `role='founder'`, all with
`developer_role IS NULL`. After this runs all three are `owner`, and
`effective_permissions('founder', developer_role='owner')` is the same 19
permissions `effective_permissions('founder')` returned before P1-6 — verified by
`tests/identity/test_developer_access_model.py::
test_a_founder_keeps_exactly_the_permissions_they_had_before_the_split`. So no
existing user gains or loses a single permission. The narrowing binds only on
roles that no production row currently holds (`admin`, `ceo`, `coo`,
`chief_of_staff`, `employee`), which is the intended behaviour change.

IDEMPOTENT. `WHERE developer_role IS NULL` means re-running writes nothing, and a
membership whose Developer access was deliberately set (or deliberately revoked
to NULL after this ran) is not reached by a second application... with one honest
caveat, recorded rather than hidden: a founder whose Developer access is revoked
to NULL by an operator WOULD be re-granted `owner` if this migration were applied
again. Alembic applies a revision once, so that requires manually re-running it;
the alternative — a marker column recording "the backfill has seen this row" —
would be new schema to guard against an operation nobody performs.
"""
from typing import Sequence, Union

from alembic import op

# Revision ids stay <= 32 characters: `alembic_version.version_num` is
# varchar(32), and a longer id fails on the version bump AFTER upgrade() has
# already run.
revision: str = "0053_developer_access"
down_revision: Union[str, None] = "0052_api_key_expiration"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # `role` is a Postgres enum; compared as text so this statement does not
    # depend on the enum's OID or on the ORM being importable from a migration.
    op.execute(
        """
        UPDATE memberships
           SET developer_role = 'owner'
         WHERE role::text = 'founder'
           AND developer_role IS NULL
        """
    )


def downgrade() -> None:
    # Reverses exactly what upgrade() wrote, as precisely as the data allows.
    #
    # A founder who was ALREADY `owner` before this migration cannot exist (the
    # column was NULL on every row until now), so narrowing to `= 'owner'` does
    # not clear anybody else's deliberate grant. A founder granted `owner`
    # through the API *after* this ran is indistinguishable from one the backfill
    # wrote, and is therefore also cleared — stated here because a downgrade that
    # silently revokes access somebody asked for is worse than one that says so.
    op.execute(
        """
        UPDATE memberships
           SET developer_role = NULL
         WHERE role::text = 'founder'
           AND developer_role = 'owner'
        """
    )
