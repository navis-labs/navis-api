"""Surface RBAC storage: organization kind, and per-surface membership roles.

WHAT THIS IS FOR. Application authorization and Developer authorization are
currently the same thing: one `memberships.role`, resolved through one
`DEFAULT_ROLE_PERMISSIONS` map, over one `Permission` vocabulary. So an
Application administrator silently holds `manage_service_accounts` and
`record_agent_runs` — they can register agents and mint credentials without ever
having been granted Developer access. This migration adds the storage the split
needs. It does NOT perform the split.

NOTHING READS THESE COLUMNS YET, AND THAT IS THE POINT. `memberships.role`
remains the sole authorization input after this runs; `effective_permissions`
is untouched. Every role keeps exactly the permissions it had — founder 19,
admin 16, ceo 13, coo 11, chief_of_staff 10, employee 2 — which
`tests/identity/test_permission_baseline_characterization.py` asserts by
fingerprint. A migration that adds columns and changes behaviour in the same
step cannot be rolled back by reverting one of the two.

STRICTLY ADDITIVE. Three columns. No column is dropped, renamed, retyped or
backfilled from another. `memberships.role` is not touched. `organization_id`
nullability is not touched anywhere. No permission row is written.

WHY `kind` DEFAULTS TO 'company' AND NOT 'personal'. Signup today creates an
organization and a founder membership for every user, so on the surface every
existing row looks like a one-person tenant that could be called personal.
Classifying them that way here would be inferring product meaning from an
artefact of the signup flow, and it would silently change what those tenants
are — a personal space is defined as un-invitable, SSO-ineligible and
Application-less, none of which is true of the existing rows. 'company' is the
compatibility-preserving value: it describes exactly what every existing
organization is today. Reclassification, if it ever happens, is a reviewed data
migration against explicit criteria, not a DEFAULT.

WHY NOT NULL ON `kind` BUT NULLABLE ON THE ROLES. `kind` answers a question
every organization has an answer to right now ("it is a company"), so a NULL
would be a third state nobody can act on. The two role columns are the opposite:
NULL is the meaningful value — it means "this member has no access to that
surface", which is precisely what an Application-only or Developer-only member
looks like, and what every member looks like until Phase 5 populates them.

On PostgreSQL 11+ a NOT NULL column with a server default is a catalog-only
change: the default is recorded in the catalog rather than written to every row,
so this does not rewrite `organizations` or hold a long ACCESS EXCLUSIVE lock.

VARCHAR + CHECK RATHER THAN A POSTGRES ENUM. The same choice 0048, 0049 and 0050
made for every closed vocabulary in this schema. A native enum needs its own
type, its own migration to extend, and cannot have a value removed at all; a
CHECK over a sorted literal list is visible in the DDL, matches `_one_of` in the
ORM models, and is altered by one DROP CONSTRAINT / ADD CONSTRAINT pair — which
is what Phase 5 will do if the role vocabulary grows.

THE ROLE VOCABULARIES ARE DELIBERATELY DIFFERENT. Application has no
`developer`; Developer has no `member`. They are not the same ladder with
different labels — an Application `member` consumes organizational
intelligence, a Developer `developer` builds and runs agents — and giving them
a shared vocabulary would invite exactly the coupling this whole migration
exists to remove.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# Revision ids stay <= 32 characters: `alembic_version.version_num` is
# varchar(32), and a longer id fails on the version bump AFTER upgrade() has
# already run.
revision: str = "0051_surface_rbac_columns"
down_revision: Union[str, None] = "0050_integration_health"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


#: Sorted, matching `_one_of` in the ORM models so autogenerate sees no drift.
_ORG_KINDS = "'company', 'personal'"
_ORG_KIND_CHECK = f"kind IN ({_ORG_KINDS})"

#: NULL is always permitted on both role columns — it is the "no access to this
#: surface" value, so the CHECK constrains only what a non-NULL row may say.
_APPLICATION_ROLES = "'admin', 'member', 'owner', 'viewer'"
_APPLICATION_ROLE_CHECK = f"application_role IS NULL OR application_role IN ({_APPLICATION_ROLES})"

_DEVELOPER_ROLES = "'admin', 'developer', 'owner', 'viewer'"
_DEVELOPER_ROLE_CHECK = f"developer_role IS NULL OR developer_role IN ({_DEVELOPER_ROLES})"


def upgrade() -> None:
    # ── organizations.kind ──────────────────────────────────────────────
    op.add_column(
        "organizations",
        sa.Column(
            "kind",
            sa.String(length=16),
            nullable=False,
            server_default="company",
        ),
    )
    op.create_check_constraint("ck_organizations_kind", "organizations", _ORG_KIND_CHECK)

    # ── memberships.application_role / developer_role ───────────────────
    #
    # Both NULL for every existing row. Authorization continues to read
    # `memberships.role` and nothing else until the dual-read phase.
    op.add_column(
        "memberships",
        sa.Column("application_role", sa.String(length=32), nullable=True),
    )
    op.add_column(
        "memberships",
        sa.Column("developer_role", sa.String(length=32), nullable=True),
    )
    op.create_check_constraint(
        "ck_memberships_application_role", "memberships", _APPLICATION_ROLE_CHECK
    )
    op.create_check_constraint(
        "ck_memberships_developer_role", "memberships", _DEVELOPER_ROLE_CHECK
    )


def downgrade() -> None:
    # Reverse order. Nothing here loses data that existed before upgrade():
    # every value in these three columns was written by this migration's
    # defaults or by a later phase, and `memberships.role` — the column
    # authorization actually reads — is never touched in either direction.
    op.drop_constraint("ck_memberships_developer_role", "memberships", type_="check")
    op.drop_constraint("ck_memberships_application_role", "memberships", type_="check")
    op.drop_column("memberships", "developer_role")
    op.drop_column("memberships", "application_role")
    op.drop_constraint("ck_organizations_kind", "organizations", type_="check")
    op.drop_column("organizations", "kind")
