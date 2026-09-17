"""integrationprovider += outlook / outlook_calendar / onedrive / teams

Revision ID: 0046_microsoft_providers
Revises: 0045_reasoning_observations
Create Date: 2026-09-06

`integrations.provider` is a NATIVE PostgreSQL enum, not a string column
(`SAEnum(IntegrationProvider, values_callable=...)` in navis/identity/models.py).
Adding a member is therefore DDL against the type, not a Python-only change, and
it comes with two Postgres constraints that shape this whole file:

  1. `ALTER TYPE ... ADD VALUE` cannot run inside a transaction block on the
     versions of Postgres this deployment targets. Alembic wraps every migration
     in one by default, so this uses `autocommit_block()`.

     NOTE: do NOT reach for the raw DBAPI connection here. The obvious
     `op.get_bind().connection.set_isolation_level(0)` is the psycopg2 idiom,
     and this deployment runs ASYNCPG (see requirements.txt and
     migrations/env.py, which builds an async engine). SQLAlchemy's asyncpg
     adapter exposes a `set_isolation_level` that accepts the call but then
     compares the stored value against the lowercase string `"autocommit"` —
     so the integer 0 never enables autocommit, a transaction is opened
     anyway, and `isolation=0` is handed to asyncpg. `autocommit_block()` is
     the driver-agnostic API and does the right thing on both.

  2. A newly added enum value cannot be USED in the same transaction that added
     it. That is not a problem here because this migration writes no rows — but
     it is exactly why this migration must ship a release AHEAD of the code that
     writes `outlook` rows. Deploying both at once works only because the
     migration commits before the app starts; deploying the app first does not.

There is no downgrade. Postgres has no `ALTER TYPE ... DROP VALUE`; removing a
member means recreating the type, rewriting every dependent column, and
rebuilding every index and constraint on them — a rewrite of the `integrations`
table to undo four rows of vocabulary. The honest downgrade is a no-op: the
extra members are inert if nothing writes them, and `_CAPABILITY` in
orchestration/sync.py refuses to sync any provider it has no mapping for, so an
orphaned value cannot start ingesting on its own.

`ADD VALUE IF NOT EXISTS` makes this re-runnable, which matters because a failed
deploy that already committed one of the four ALTERs would otherwise wedge the
migration on retry.
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op

revision: str = "0046_microsoft_providers"
down_revision: Union[str, None] = "0045_reasoning_observations"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Order is cosmetic — enum members sort by their internal sort order, and these
# append after the existing Google/Slack values.
_NEW_PROVIDERS = ("outlook", "outlook_calendar", "onedrive", "teams")

_ENUM_TYPE = "integrationprovider"


def upgrade() -> None:
    # Escape Alembic's surrounding transaction so each ALTER TYPE commits on
    # its own. `autocommit_block()` handles the driver differences (see the
    # module docstring) and restores the previous mode on exit.
    with op.get_context().autocommit_block():
        for value in _NEW_PROVIDERS:
            op.execute(
                f"ALTER TYPE {_ENUM_TYPE} ADD VALUE IF NOT EXISTS '{value}'"
            )


def downgrade() -> None:
    """Deliberately a no-op — see the module docstring.

    Removing an enum member in Postgres requires recreating the type and
    rewriting `integrations`, which is a far larger and riskier operation than
    the one being undone. Unused members cost nothing: no row references them,
    and the sync orchestrator fails closed on any provider absent from its
    capability map.
    """
