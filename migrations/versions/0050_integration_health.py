"""Integration health: auth state and sync attempt/success accounting.

WHAT WAS WRONG. `integrations.is_active` carried three different meanings at
once, and every reader collapsed them into one. It means "the user wants this
connected" — an intent, set when somebody clicks Connect and cleared when they
click Disconnect. It was ALSO being read as "authentication works" and as "data
is arriving", because there was nowhere else to look. So a Google grant whose
refresh token had been revoked stayed `is_active = true` forever, the sync loop
retried it on every pass, and `/developer/context` told an agent the source was
connected while it had ingested nothing for weeks.

The failure mode this closes is specific and it is the expensive one: an agent
that cannot distinguish HEALTHY + NO RESULT from BROKEN + NO RESULT will report
"your company has no open risks" when what actually happened is that Navis
stopped being able to see them.

FOUR COLUMNS ON ONE TABLE. Nothing is dropped, nothing is rewritten, and
`is_active` is not touched — its meaning is unchanged and every existing reader
keeps working exactly as before. Three of the four are nullable; `auth_state`
takes a NOT NULL with a server default, which on PostgreSQL 11+ is still a
catalog-only change (the default is stored in the catalog, not written to every
row), so this does not rewrite the table or hold a long ACCESS EXCLUSIVE lock.

WHY `auth_state` IS A COLUMN AND NOT A KEY IN `sync_state`. `sync_state` is
provider-specific JSONB holding cursors, Gmail watch state and Slack backfill
progress, and it is guarded per-integration by an advisory lock precisely
because advancing a cursor is a read-modify-write that must not race
(`navis/signals/locking.py`). Health is written on a DIFFERENT schedule from the
cursor — most importantly it is written when the sync FAILED and the cursor did
not move — so putting it in the same blob would couple a health write to a
cursor lock and make "which sources are broken" a query that cannot use an
index. It has to be independently queryable: the dispatch loop filters on it to
stop retrying a dead grant, and the Developer context route reads it per source.

WHY A VARCHAR + CHECK RATHER THAN A POSTGRES ENUM. The same choice 0048 and 0049
made for every closed vocabulary in this schema. A native enum needs its own
type, its own migration to extend, and cannot have a value removed at all; a
CHECK over a sorted literal list is visible in the DDL, matches `_one_of` in the
ORM models, and is altered by one `DROP CONSTRAINT` / `ADD CONSTRAINT` pair.

THE VOCABULARY IS THE ONE THE CODE ALREADY USES. `reconsent_required` is named
after `navis.signals.google.oauth.ReconsentRequired`, which Google's
`invalid_grant` already raises, and after `SlackReconsentRequired`. `revoked` is
named after Slack's `tokens_revoked` event. No new terminology is introduced
here — this migration gives the existing vocabulary somewhere to live.

BACKFILL. Every existing row becomes `ok`. That is the honest default and not
merely the convenient one: nothing has ever recorded an auth failure, so no row
is KNOWN to be broken, and marking rows `reconsent_required` on a guess would
stop syncing accounts that work. A genuinely dead grant flips to
`reconsent_required` on its next sync attempt, which is at most one beat away.
`last_success_at` stays NULL for every existing row rather than being backfilled
from `updated_at` — `updated_at` moves for reasons that have nothing to do with
a successful sync, and a fabricated success timestamp is worse than an absent
one because staleness is computed from it.

DOWNGRADE drops the four columns and the check. It destroys the recorded health,
which is the data; it is written for a forward-fix in staging, not as a routine
production operation.
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# Revision ids stay <= 32 characters: `alembic_version.version_num` is
# varchar(32), and a longer id fails on the version bump AFTER upgrade() has
# already run.
revision: str = "0050_integration_health"
down_revision: Union[str, None] = "0049_run_telemetry"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


#: Sorted, matching `_one_of` in the ORM models so autogenerate sees no drift.
_AUTH_STATES = "'ok', 'reconsent_required', 'revoked'"

_AUTH_STATE_CHECK = f"auth_state IN ({_AUTH_STATES})"


def upgrade() -> None:
    # `auth_state` — AUTHENTICATION HEALTH. Distinct from `is_active`, which
    # remains the user's intent. NOT NULL with a server default so no reader
    # ever has to handle "unknown": a row either works, needs re-consent, or has
    # been revoked. The server default also makes this safe for an older process
    # still inserting rows without the column.
    op.add_column(
        "integrations",
        sa.Column(
            "auth_state",
            sa.String(24),
            nullable=False,
            server_default="ok",
        ),
    )
    op.create_check_constraint(
        "ck_integrations_auth_state", "integrations", _AUTH_STATE_CHECK
    )

    # ── SYNC HEALTH ──────────────────────────────────────────────────────
    # Three facts, and they answer three different questions. `last_attempt_at`
    # says whether the sync loop is still reaching this row at all — without it,
    # a source skipped by a scheduling bug is indistinguishable from one that is
    # healthy and quiet. `last_success_at` is the one staleness is computed from
    # and the only one that says data actually ARRIVED. `last_failure_reason`
    # says what went wrong in words an operator can act on.
    #
    # All three nullable, because "has never been attempted" and "has never
    # succeeded" are real states for a freshly-connected integration and must
    # not be forged into a timestamp.
    op.add_column(
        "integrations",
        sa.Column("last_attempt_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "integrations",
        sa.Column("last_success_at", sa.DateTime(timezone=True), nullable=True),
    )
    # Bounded like every other operator-facing free-text column in this schema.
    # A provider error message is a classic place for a token or a URL with
    # credentials in the query string to appear, so the writer truncates and
    # scrubs; the width is the backstop, not the control.
    op.add_column(
        "integrations",
        sa.Column("last_failure_reason", sa.String(500), nullable=True),
    )

    # THE INDEX THE DISPATCH GATE USES. `_dispatch_all` pages every active
    # integration and now excludes the ones known to need re-consent, so the
    # predicate is (is_active, auth_state). Without this the gate turns the
    # fan-out into a full scan of a table that only grows.
    op.create_index(
        "ix_integrations_active_auth_state",
        "integrations",
        ["is_active", "auth_state"],
    )


def downgrade() -> None:
    op.drop_index("ix_integrations_active_auth_state", table_name="integrations")
    op.drop_column("integrations", "last_failure_reason")
    op.drop_column("integrations", "last_success_at")
    op.drop_column("integrations", "last_attempt_at")
    op.drop_constraint("ck_integrations_auth_state", "integrations", type_="check")
    op.drop_column("integrations", "auth_state")
