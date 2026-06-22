"""Organization — CHECK constraints for enum-backed columns (ORG-001)

Revision ID: 0021_org_enum_checks
Revises: 0020_org_flag_global_unique
Create Date: 2026-06-22

status / plan / stage are stored as plain varchar with application-level enums; the DB
accepted any string. This adds VARCHAR + CHECK constraints (deliberately NOT native
Postgres enums — Navis evolves fast and a CHECK is far cheaper to ALTER than an enum type
when a value is added). Existing rows are validated first so the migration fails with a
clear message rather than a raw constraint violation. Fully reversible.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0021_org_enum_checks"
down_revision: Union[str, None] = "0020_org_flag_global_unique"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_STATUS = ("trial", "active", "suspended", "churned")
_PLAN = ("free", "pro", "team", "enterprise")
_STAGE = (
    "created", "profile_complete", "gmail_connected", "calendar_connected",
    "first_signal", "memory_ready", "state_ready", "first_brief", "activated",
)

# (constraint_name, table, column, allowed_values)
_CHECKS = (
    ("ck_org_lifecycle_status", "organization_lifecycles", "status", _STATUS),
    ("ck_org_lifecycle_plan", "organization_lifecycles", "plan", _PLAN),
    ("ck_org_onboarding_stage", "organization_onboarding", "stage", _STAGE),
)


def _in(column: str, values: tuple[str, ...]) -> str:
    return f"{column} IN (" + ", ".join(f"'{v}'" for v in values) + ")"


def upgrade() -> None:
    conn = op.get_bind()
    # Validate existing data FIRST — a clean, actionable error beats a raw CHECK failure.
    for name, table, column, values in _CHECKS:
        bad = conn.execute(
            sa.text(f"SELECT count(*) FROM {table} WHERE NOT ({_in(column, values)})")
        ).scalar()
        if bad:
            raise RuntimeError(
                f"{bad} {table}.{column} row(s) hold values outside the enum set; "
                f"clean them before applying {revision}"
            )
    for name, table, column, values in _CHECKS:
        op.create_check_constraint(name, table, _in(column, values))


def downgrade() -> None:
    for name, table, _column, _values in reversed(_CHECKS):
        op.drop_constraint(name, table, type_="check")
