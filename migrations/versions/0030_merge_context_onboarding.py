"""merge alembic heads: context_bundles + org onboarding fields

Revision ID: 0030_merge_context_onboarding
Revises: 0029_context_bundles, 0027_org_profile_onboarding_fields
Create Date: 2026-06-30

Two feature branches added migrations off 0026_signal_dedupe_integration in parallel:
  * 0027_signal_channel_id -> 0028_mem_org_time_index -> 0029_context_bundles (Context layer)
  * 0027_org_profile_onboarding_fields (Google SSO / org onboarding)
This is a no-op MERGE migration that rejoins the graph into a single head. It performs no
schema changes — it only unifies the two lineages so `alembic upgrade head` is unambiguous.
"""
from typing import Sequence, Union

revision: str = "0030_merge_context_onboarding"
down_revision: Union[str, Sequence[str], None] = (
    "0029_context_bundles",
    "0027_org_profile_onboarding_fields",
)
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
