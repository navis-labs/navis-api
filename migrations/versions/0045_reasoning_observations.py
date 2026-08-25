"""reasoning_signal_observations — what a conclusion looked like at one snapshot

Revision ID: 0045_reasoning_observations
Revises: 0044_mem_org_observed_index
Create Date: 2026-08-24

`reasoning_signals` conflates two things. Its identity — `compute_signal_id` over
(org, reasoning_version, engine, engine_version, type, entity, dedup_key) — is
deliberately stable across snapshots, and must stay that way: `execution_requests`,
`learning_events`, observation payloads and `intelligence_insights` all persist
`reasoning_signal_ids` as long-lived references that cannot be recomputed once
`reasoning_version` moves. Not one `dedup_key` in the engine set carries time.

But `confidence`, `severity`, `why` and `evidence` are all functions of
`state.as_of`, and `persist_signals` wrote them ON CONFLICT DO NOTHING. So the
first detection won permanently:

    commitment f33778d8 stored at 7.16 days / confidence 0.7386, while the
    decision citing that exact row reported 12 days / 0.8951. Both were computed
    by the engines; only the first was ever written.

This table stores the second half. One row per (conclusion, state snapshot),
append-only, never updated — so a conclusion that ages produces one signal row and
many observation rows, and both the identity and the history survive.

IDENTITY IS THE PRIMARY KEY, NOT A UNIQUE INDEX. `observation_id` is
uuid5(signal_id, state_watermark) (`store.compute_observation_id`), which is how
every other identity in this codebase is expressed and is also the only form that
works here: `state_watermark` is nullable, Postgres treats NULLs as distinct in a
UNIQUE index, and an organization with no folded memory would therefore accumulate
a duplicate observation on every pass. Folding the null into a sentinel inside the
hash makes re-running a pass a no-op at the database.

NO BACKFILL, DELIBERATELY. Existing `reasoning_signals` rows keep their columns,
which are now documented as the FIRST observation — real recorded data, not a
guess. Nothing here invents an observation for a snapshot at which the engines were
never run: a conclusion Navis did not record at watermark W did not happen at W,
and representing it as unavailable is the honest answer. Observations accrue from
the next reasoning pass onward.

Additive and reversible. No existing column is altered or dropped.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0045_reasoning_observations"
down_revision: Union[str, None] = "0044_mem_org_observed_index"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "reasoning_signal_observations",
        sa.Column("observation_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "signal_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("reasoning_signals.signal_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "organization_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("state_watermark", sa.String(64), nullable=True),
        sa.Column("state_as_of", sa.DateTime(timezone=True), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("severity", sa.Float(), nullable=True),
        sa.Column("why", sa.Text(), nullable=False),
        sa.Column("evidence", postgresql.JSONB(), nullable=True),
        # Logical time (= state_as_of), never wall clock.
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        # Wall-clock write time — operational metadata only.
        sa.Column(
            "recorded_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    # "Which conclusions were observed at watermark W for this org?" — the
    # obligation count and every snapshot-scoped provenance read.
    op.create_index(
        "ix_reasoning_obs_org_watermark",
        "reasoning_signal_observations",
        ["organization_id", "state_watermark"],
    )
    # "How did this one conclusion age?" — oldest-first history.
    op.create_index(
        "ix_reasoning_obs_signal_time",
        "reasoning_signal_observations",
        ["signal_id", "state_as_of"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_reasoning_obs_signal_time", table_name="reasoning_signal_observations"
    )
    op.drop_index(
        "ix_reasoning_obs_org_watermark", table_name="reasoning_signal_observations"
    )
    op.drop_table("reasoning_signal_observations")
