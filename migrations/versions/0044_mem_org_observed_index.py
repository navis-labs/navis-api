"""memory_records (organization_id, created_at, memory_id) index — the observation cursor

Revision ID: 0044_mem_org_observed_index
Revises: 0043_membership_role_not_null
Create Date: 2026-08-24

State V2 folds in EVENT order and resumes in OBSERVATION order. `ix_mem_org_time`
(0028) serves the first — streaming an org's whole history by
(event_timestamp, memory_id). Nothing served the second, because until
projection_version 2 the resume cursor was the fold key.

It cannot be, and that was the P0: the Calendar connector deliberately stamps a
signal with the MEETING'S START (`signals/calendar/connector.py::_to_event`), so
event time runs ahead of wall clock. One meeting booked for tomorrow pushed the
cursor past every record observed afterwards, and `stream(after=watermark)`
silently returned none of them. `memory_records.created_at` is the only timestamp
on the row that cannot be in the future — Postgres assigns it at INSERT via
`server_default=func.now()` and the application never writes it — so the cursor
now walks (created_at, memory_id).

WHY THIS INDEX IS NECESSARY, measured rather than assumed. Against production,
`MemoryReader.observed_since`'s keyset predicate plans as:

    Limit -> Sort (Sort Key: created_at, memory_id)
               -> Seq Scan on memory_records
                    Filter: organization_id = $1
                            AND ROW(created_at, memory_id) > ROW($2, $3)

A sequential scan plus an explicit sort, on the largest table in the pipeline, on
the hot path of every brief build and every /context read. `memory_records` is
append-only and grows without bound, so that plan degrades linearly and forever.
With this index the same query is an index range scan that returns rows already in
cursor order — no sort node, and the scan starts at the cursor instead of at the
first row.

`count()` needed nothing: it already plans as an Index Only Scan on ix_mem_entity,
whose leading column is organization_id.

Pure-additive and reversible. Created CONCURRENTLY so it never locks the table at
scale (requires autocommit; no-op safe on an empty/new table).
"""
from typing import Sequence, Union

from alembic import op

revision: str = "0044_mem_org_observed_index"
down_revision: Union[str, None] = "0043_membership_role_not_null"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.get_context().autocommit_block():
        op.create_index(
            "ix_mem_org_observed", "memory_records",
            ["organization_id", "created_at", "memory_id"],
            postgresql_concurrently=True, if_not_exists=True,
        )


def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.drop_index(
            "ix_mem_org_observed", table_name="memory_records",
            postgresql_concurrently=True, if_exists=True,
        )
