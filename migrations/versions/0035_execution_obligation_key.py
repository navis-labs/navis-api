"""Stable reply-obligation identity.

ONE inbound Gmail message = ONE reply obligation = at most ONE sendable
execution.

Before this, a reply's idempotency was a hash of (org, decision_id, action_type,
payload). Both `decision_id` (uuid5 over the state watermark) and the payload's
`waiting_days` are observation-scoped, so rebuilding the brief minted a new
identity for an unchanged email. Measured on real data: one message, three
briefs, three distinct idempotency keys — three independently sendable rows.

This migration:
  1. adds `obligation_key`, the readable natural key
     "reply|{candidate_kind}|{thread_id}|{rfc_message_id}";
  2. backfills it for existing reply_to_email rows straight from the payload;
  3. neutralises any pre-existing duplicate so two rows for one obligation can
     never both be sent;
  4. adds a PARTIAL unique index so the database, not application code, is the
     final guarantee.

Idempotent and safe to re-run: every step is conditional on current state.
Sends no mail and approves nothing.

Revision ID: 0035_execution_obligation
Revises: 0034_action_drafts
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0035_execution_obligation"
down_revision = "0034_action_drafts"
branch_labels = None
depends_on = None

# Mirrors navis/execution/obligation.py::reply_obligation_key. Kept as SQL so
# the backfill is one set-based statement rather than a row-by-row Python loop
# over a table that can be large.
#
#   - thread_id falls back to reply_thread_id, as the Python does
#   - the angle brackets around a Message-ID are delimiters, not identity, so
#     they are trimmed here exactly as _clean_message_id trims them
_OBLIGATION_SQL = """
    'reply|'
    || (payload->>'candidate_kind')
    || '|'
    || COALESCE(payload->>'thread_id', payload->>'reply_thread_id')
    || '|'
    || btrim(btrim(payload->>'in_reply_to_message_id'), '<>')
"""

# A row only HAS an obligation identity if every component is present and
# non-blank. Anything else keeps obligation_key NULL and is excluded from the
# partial index — it is not silently given an invented identity.
_HAS_OBLIGATION = """
    action_type = 'reply_to_email'
    AND NULLIF(btrim(COALESCE(payload->>'candidate_kind', '')), '') IS NOT NULL
    AND NULLIF(btrim(COALESCE(payload->>'thread_id',
                              payload->>'reply_thread_id', '')), '') IS NOT NULL
    AND NULLIF(btrim(btrim(COALESCE(payload->>'in_reply_to_message_id', '')),
                     '<>'), '') IS NOT NULL
"""

# Which row wins when several already exist for one obligation. An EXECUTED row
# is authoritative — it represents a message that actually left the account —
# and a human's REJECTED/CANCELLED decision must never be resurrected, so those
# rank last and simply keep obligation_key NULL.
_AUTHORITY = """
    CASE status
        WHEN 'executed'  THEN 0
        WHEN 'executing' THEN 1
        WHEN 'approved'  THEN 2
        WHEN 'requested' THEN 3
        WHEN 'failed'    THEN 4
        WHEN 'rejected'  THEN 5
        WHEN 'cancelled' THEN 6
        ELSE 7
    END
"""

# Still-sendable states. A LOSER in one of these would remain independently
# approvable, which is precisely the duplicate we are removing.
_SENDABLE = ("requested", "approved", "failed")


def upgrade() -> None:
    op.add_column(
        "execution_requests",
        sa.Column("obligation_key", sa.String(length=512), nullable=True),
    )

    # 1. Backfill the winner for every obligation. Ranked so exactly one row per
    #    (org, obligation) is claimed even where duplicates already exist.
    op.execute(
        f"""
        WITH ranked AS (
            SELECT id,
                   {_OBLIGATION_SQL} AS obl,
                   ROW_NUMBER() OVER (
                       PARTITION BY organization_id, {_OBLIGATION_SQL}
                       ORDER BY {_AUTHORITY}, created_at, id
                   ) AS rn
            FROM execution_requests
            WHERE {_HAS_OBLIGATION}
        )
        UPDATE execution_requests e
        SET obligation_key = ranked.obl
        FROM ranked
        WHERE e.id = ranked.id
          AND ranked.rn = 1
          AND e.obligation_key IS DISTINCT FROM ranked.obl
        """
    )

    # 2. Neutralise losers that could still be sent. They keep obligation_key
    #    NULL (so they never claim the identity) and are moved to CANCELLED so
    #    no one can approve a second copy of a reply that another row owns.
    #
    #    Nothing is approved and nothing is sent here — this only ever removes
    #    the ability to send. Rows already terminal are left exactly as they are.
    op.execute(
        f"""
        WITH ranked AS (
            SELECT id,
                   organization_id,
                   {_OBLIGATION_SQL} AS obl,
                   ROW_NUMBER() OVER (
                       PARTITION BY organization_id, {_OBLIGATION_SQL}
                       ORDER BY {_AUTHORITY}, created_at, id
                   ) AS rn
            FROM execution_requests
            WHERE {_HAS_OBLIGATION}
        )
        UPDATE execution_requests e
        SET status = 'cancelled',
            last_error = COALESCE(e.last_error, '')
                || 'superseded: another execution owns this reply obligation '
                || '(migration 0035)'
        FROM ranked
        WHERE e.id = ranked.id
          AND ranked.rn > 1
          AND e.status IN {_SENDABLE}
        """
    )

    # 3. The guarantee itself. Partial, because action types with no
    #    message-level identity legitimately carry NULL and must not collide
    #    with each other.
    op.create_index(
        "uq_execution_obligation",
        "execution_requests",
        ["organization_id", "obligation_key"],
        unique=True,
        postgresql_where=sa.text("obligation_key IS NOT NULL"),
    )


def downgrade() -> None:
    # The cancellations in step 2 are deliberately NOT reversed: they removed
    # duplicate sendable rows, and resurrecting them on downgrade would restore
    # the ability to double-send. The column and index go; the safety stays.
    op.drop_index("uq_execution_obligation", table_name="execution_requests")
    op.drop_column("execution_requests", "obligation_key")
