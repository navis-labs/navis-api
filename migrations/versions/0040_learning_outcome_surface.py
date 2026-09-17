"""Learning: outcome_surface + failure_kind + provider_reference on learning_events

Revision ID: 0040_learning_surface
Revises: 0039_intelligence_reality_input
Create Date: 2026-08-19

Three columns, all of which close a gap between what Observation PUBLISHES and
what Learning was able to record:

  outcome_surface     Which of Observation's three sources the fact came from —
                      delivery (the machine acted / the channel broke), judgement
                      (a person's verdict on a proposal) or world (what actually
                      happened afterwards). Learning had one success/failure axis
                      and no way to say that "Gmail was down" and "the customer
                      ignored us" are different kinds of evidence.

  failure_kind        `execution.contract.FailureKind`, verbatim. Execution
                      computes it, Observation carries it, and Learning dropped
                      it — so a failure could not be told apart from any other
                      failure without re-reading `outcomes`, which Learning is
                      forbidden from doing.

  provider_reference  The external object (gmail_message_id, calendar event id).
                      Also already published, also dropped.

BACKFILL. `outcome_surface` is NOT NULL, and existing rows get it from a CASE
over `outcome_type` — a column they already store. That is a derivation, not a
reinterpretation: the surface was always implied by the type, it simply had
nowhere to live.

WHAT IS DELIBERATELY NOT BACKFILLED: `result`. `action_rejected` used to fall
through Learning's classifier to NEUTRAL and now classifies as FAILURE, but
`learning_events` is an immutable log of what was observed AND how it was
understood at the time. Rewriting a stored classification is rewriting history;
`schema_version` (bumped to 2) is what distinguishes rows recorded under the old
understanding. Re-deriving under the new one is `LearningReplayService.rebuild`'s
job, run deliberately, not a migration side effect.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0040_learning_surface"
down_revision: Union[str, None] = "0039_intelligence_reality_input"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# Mirrors navis.learning.attribution._OUTCOME_VOCABULARY. A test asserts the two
# stay in step, so this cannot silently drift from the running classifier.
_SURFACE_BY_TYPE = {
    "action_executed": "delivery",
    "action_failed": "delivery",
    "action_rejected": "judgement",
    "action_cancelled": "judgement",
    "reply_received": "world",
    "meeting_accepted": "world",
    "meeting_completed": "world",
    "no_response": "world",
}


def upgrade() -> None:
    op.add_column(
        "learning_events",
        sa.Column("outcome_surface", sa.String(length=16), nullable=True),
    )
    op.add_column(
        "learning_events",
        sa.Column("failure_kind", sa.String(length=32), nullable=True),
    )
    op.add_column(
        "learning_events",
        sa.Column(
            "provider_reference",
            postgresql.JSONB(),
            server_default=sa.text("'{}'::jsonb"),
            nullable=True,
        ),
    )

    # Backfill from the type each row already carries. Anything Learning was
    # never taught is recorded as a delivery attempt — the same inert default
    # the live classifier applies to an unknown type.
    whens = " ".join(
        f"WHEN outcome_type = '{otype}' THEN '{surface}'"
        for otype, surface in sorted(_SURFACE_BY_TYPE.items())
    )
    op.execute(
        f"UPDATE learning_events SET outcome_surface = "  # noqa: S608 - literals above
        f"CASE {whens} ELSE 'delivery' END "
        f"WHERE outcome_surface IS NULL"
    )
    op.execute("UPDATE learning_events SET provider_reference = '{}'::jsonb "
               "WHERE provider_reference IS NULL")

    op.alter_column("learning_events", "outcome_surface", nullable=False)

    # Delivery / judgement / world are the axis every consumer now filters on.
    op.create_index(
        "ix_le_org_surface", "learning_events", ["organization_id", "outcome_surface"]
    )


def downgrade() -> None:
    op.drop_index("ix_le_org_surface", table_name="learning_events")
    op.drop_column("learning_events", "provider_reference")
    op.drop_column("learning_events", "failure_kind")
    op.drop_column("learning_events", "outcome_surface")
