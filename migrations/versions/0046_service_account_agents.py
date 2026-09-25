"""service_accounts: the three descriptive columns an AGENT needs, and a tenant index.

WHY NO `agents` TABLE. `/api/v1/developer/agents` is a product-facing view over
`service_accounts`, not a new entity. That row already carries every STRUCTURAL
fact an agent has — organization, name, scope list, revocation clock, timestamps
— and `authz._principal_from_api_key` already closes credential -> agent ->
organization against it, including the refusal when a key and its account name
different tenants. A second table would be a second machine identity: a second
row to revoke, a second place a tenant is recorded, and a second answer to "who
is calling" the day the two disagree. So this migration ADDS COLUMNS.

WHAT THE COLUMNS ARE, AND WHAT THEY ARE NOT. `description`, `environment` and
`runtime` are DESCRIPTIVE. No gate reads them: an agent tagged `production` holds
exactly the permissions its scopes name, and `custom` is a first-class runtime
rather than a fallback. They exist so an operator can tell one agent from another
in a list, which is the whole difference between a fleet and a pile of keys.

There is deliberately no `status` column. `revoked_at` is already the authority
authentication reads to refuse a revoked agent; a second column would be a second
authority the auth path does not consult, and the wire `status` is derived from
`revoked_at` instead (`navis.identity.agents.status_of`).

NOT NULL WITH SERVER DEFAULTS, so existing rows are back-filled by the ALTER
itself and no application deploy order is required — an old process writing a row
without these columns still produces a valid one. Postgres 11+ adds a NOT NULL
column with a constant default without rewriting the table, so this is a
catalog-only change plus a brief ACCESS EXCLUSIVE lock.

THE CHECK CONSTRAINTS ARE THE POINT OF DOING THIS IN SQL AT ALL. The HTTP edge
validates both vocabularies, but the edge is one of several writers over a
table's life (a fixture, a migration, an operator's psql session). A free-text
tier or runtime cannot be grouped, filtered or counted later, which is the only
reason to record either. They are declared identically on the ORM model, so
`Base.metadata.create_all` (what the test suites build from) and a migrated
database produce the same schema.

THE INDEX. `ix_service_accounts_org_created` matches
`GET /api/v1/developer/agents` exactly — `WHERE organization_id = ? ORDER BY
created_at DESC` — and leads with the tenant key, so one organization's fleet
size never affects another's scan. Created CONCURRENTLY is not used: the table is
small in every deployment and a concurrent build cannot run inside Alembic's
transaction.

SAFETY. Additive in both directions. `downgrade` drops what `upgrade` added and
touches no other column, so a rollback loses the descriptive metadata and nothing
an agent's identity or authorization depends on.

Revision ID: 0046_service_account_agents
Revises: 0045_reasoning_observations
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0046_service_account_agents"
down_revision: Union[str, None] = "0045_reasoning_observations"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Kept in lockstep with `navis.identity.agents`. Spelled out here rather than
# imported: a migration must describe the schema at ITS point in history, and an
# import would silently re-point this file at a future vocabulary.
_ENVIRONMENTS = ("development", "production", "staging")
_RUNTIMES = ("claude-code", "codex", "cursor", "custom", "gemini", "openai-agents")


def _in_list(column: str, values: tuple[str, ...]) -> str:
    return f"{column} IN ({', '.join(repr(v) for v in sorted(values))})"


def upgrade() -> None:
    op.add_column(
        "service_accounts",
        sa.Column("description", sa.String(length=500), nullable=False, server_default=""),
    )
    op.add_column(
        "service_accounts",
        sa.Column(
            "environment", sa.String(length=16), nullable=False, server_default="development"
        ),
    )
    op.add_column(
        "service_accounts",
        sa.Column("runtime", sa.String(length=32), nullable=False, server_default="custom"),
    )
    op.create_check_constraint(
        "ck_sa_environment", "service_accounts", _in_list("environment", _ENVIRONMENTS)
    )
    op.create_check_constraint(
        "ck_sa_runtime", "service_accounts", _in_list("runtime", _RUNTIMES)
    )
    op.create_index(
        "ix_service_accounts_org_created",
        "service_accounts",
        ["organization_id", sa.text("created_at DESC")],
    )


def downgrade() -> None:
    op.drop_index("ix_service_accounts_org_created", table_name="service_accounts")
    op.drop_constraint("ck_sa_runtime", "service_accounts", type_="check")
    op.drop_constraint("ck_sa_environment", "service_accounts", type_="check")
    op.drop_column("service_accounts", "runtime")
    op.drop_column("service_accounts", "environment")
    op.drop_column("service_accounts", "description")
