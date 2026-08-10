"""Organization profile: onboarding form fields

Revision ID: 0027_org_profile_onboarding
Revises: 0026_signal_dedupe_integration
Create Date: 2026-06-23

Adds three nullable columns to organization_profiles to store the data collected
during the Phase 1 org onboarding form:

  team_size    VARCHAR(16)   — "1", "2-10", "11-50", "51-200", "200+"
  org_stage    VARCHAR(32)   — "idea", "pre-seed", "seed", "series-a", "series-b-plus"
  primary_goal VARCHAR(500)  — free-text: the #1 thing the user wants Navis to help with

All three are nullable so existing organizations are unaffected. The
OrganizationService.update_profile() method advances the onboarding stage to
PROFILE_COMPLETE once company_name + industry + team_size + org_stage are all set.

Reversible: downgrade simply drops the three columns (no data dependency).
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0027_org_profile_onboarding"
down_revision: Union[str, None] = "0026_signal_dedupe_integration"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "organization_profiles",
        sa.Column("team_size", sa.String(16), nullable=True),
    )
    op.add_column(
        "organization_profiles",
        sa.Column("org_stage", sa.String(32), nullable=True),
    )
    op.add_column(
        "organization_profiles",
        sa.Column("primary_goal", sa.String(500), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("organization_profiles", "primary_goal")
    op.drop_column("organization_profiles", "org_stage")
    op.drop_column("organization_profiles", "team_size")
