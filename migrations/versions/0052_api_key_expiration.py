"""Credential expiration: api_keys.expires_at.

WHAT THIS IS FOR. A Navis API key is currently valid from creation until
somebody revokes it. Nothing bounds its life, so a key that leaks into a CI log,
a laptop backup or a screenshot stays usable until a human notices and acts.
Every comparable platform treats a bounded lifetime as the default posture —
OpenAI "strongly recommend[s] setting an expiration date when you create a
project API key", and Anthropic's Admin API takes the expiry at creation — and
the audit of the Developer credential lifecycle found this to be its one real
gap.

STRICTLY ADDITIVE, AND DELIBERATELY DOES NOTHING TO EXISTING KEYS. One nullable
column. No column is dropped, renamed, retyped or backfilled. No row is
rewritten. No default expiry is applied.

WHY NULLABLE, AND WHY NULL IS NOT A DEFECT. NULL means "does not expire", which
is exactly what every one of today's credentials is, so this migration cannot
break a live integration. Forcing an expiry on existing keys would be a silent
outage scheduled for whenever the backfilled date arrived — the opposite of a
security improvement, because the first thing an operator would do is mint a
replacement with no expiry at all. A tenant that wants bounded lifetimes opts in
per credential, and a future phase may add an organization-level maximum; that
is a policy decision with an owner, not a column default.

EXPIRY IS NOT REVOCATION, and the schema keeps them apart on purpose.
`revoked_at` records a DECISION somebody made; `expires_at` records a LIMIT
chosen when the credential was minted. They answer different questions in an
audit ("who cut this off, and when" versus "when was this always going to
stop"), a key can be both, and collapsing them into one timestamp would lose
which of the two happened. Authentication refuses either.

NO INDEX. `verify()` looks a credential up by `prefix` — an indexed column —
and evaluates expiry on the handful of rows that match, so an index on
`expires_at` would serve no read this system performs. The only query that
would want one is a future "what is expiring soon" sweep across all tenants,
which does not exist; adding the index now would be paying write cost on every
key creation for a reader that may never be written.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# Revision ids stay <= 32 characters: `alembic_version.version_num` is
# varchar(32), and a longer id fails on the version bump AFTER upgrade() has
# already run.
revision: str = "0052_api_key_expiration"
down_revision: Union[str, None] = "0051_surface_rbac_columns"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Nullable with NO server default: adding a nullable column without a
    # default is a catalog-only change on PostgreSQL, so this neither rewrites
    # `api_keys` nor holds a long ACCESS EXCLUSIVE lock on the table every
    # authenticated request reads.
    op.add_column(
        "api_keys",
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    # Loses only values this migration made it possible to write. Every
    # credential reverts to "does not expire", which is what they all were
    # before it ran — so a downgrade restores behaviour exactly rather than
    # leaving keys in a state the old code cannot interpret.
    op.drop_column("api_keys", "expires_at")
