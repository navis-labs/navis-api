"""daily_briefs.policy_version — which Adaptive policy governed a brief

Revision ID: 0041_brief_policy_version
Revises: 0040_learning_surface
Create Date: 2026-08-20

Adaptive's two live fields — `brief_max_items` and `brief_min_confidence` —
change what a founder actually sees in the morning brief, and nothing recorded
that they had been applied. A brief built under an approved policy was
byte-indistinguishable from one built on code defaults, so "why did Monday show
four items and Tuesday nine?" had no answer in the data.

NULLABLE, and null is meaningful: it is the honest value for a brief that ran on
code defaults, and stays distinguishable from policy version 0. Existing rows
predate Adaptive ever being active (production holds zero policy_snapshots), so
there is nothing to backfill and no value that could be inferred for them.

Not part of `compute_brief_id`, deliberately: the brief id is
(org, reasoning_version, state_watermark), and activating a new policy must
REPLACE that generation's brief rather than mint a parallel one beside it.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0041_brief_policy_version"
down_revision: Union[str, None] = "0040_learning_surface"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "daily_briefs",
        sa.Column("policy_version", sa.Integer(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("daily_briefs", "policy_version")
