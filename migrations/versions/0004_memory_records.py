"""memory_records — append-only event-sourced history layer

Revision ID: 0004_memory_records
Revises: 0003_encrypt_tokens
Create Date: 2026-06-17

Immutable fact log. memory_id is a deterministic UUIDv5 PK (idempotent replay).
Full provenance to signal/integration/extractor. JSONB payload + schema_version
for evolution.

SCALE NOTE (Phase 8): for 100M+ rows this table should be RANGE-partitioned by
event_timestamp (monthly). To convert, recreate as:

    CREATE TABLE memory_records (... , PRIMARY KEY (memory_id, event_timestamp))
        PARTITION BY RANGE (event_timestamp);
    CREATE TABLE memory_records_2026_06 PARTITION OF memory_records
        FOR VALUES FROM ('2026-06-01') TO ('2026-07-01');
    -- + a default partition, monthly create-ahead, and detach+archive old months.

The base table below is non-partitioned so dev/create_all and CI integration tests
work unchanged; partitioning is an operational migration applied in production.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0004_memory_records"
down_revision: Union[str, None] = "0003_encrypt_tokens"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "memory_records",
        sa.Column("memory_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("organization_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("entity_type", sa.String(length=40), nullable=False),
        sa.Column("entity_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("source_signal_event_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("source_integration_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("extractor_name", sa.String(length=80), nullable=False),
        sa.Column("extractor_version", sa.String(length=20), nullable=False),
        sa.Column("schema_version", sa.Integer(), nullable=False),
        sa.Column("event_timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False, server_default="1.0"),
        sa.Column("dedup_key", sa.String(length=800), nullable=False),
        sa.Column("payload", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False
        ),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["source_signal_event_id"], ["signal_events.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["source_integration_id"], ["integrations.id"], ondelete="SET NULL"),
    )
    op.create_index("ix_mem_entity", "memory_records", ["organization_id", "entity_type", "entity_id"])
    op.create_index(
        "ix_mem_org_type_time", "memory_records", ["organization_id", "entity_type", "event_timestamp"]
    )
    op.create_index("ix_mem_signal", "memory_records", ["source_signal_event_id"])


def downgrade() -> None:
    op.drop_index("ix_mem_signal", table_name="memory_records")
    op.drop_index("ix_mem_org_type_time", table_name="memory_records")
    op.drop_index("ix_mem_entity", table_name="memory_records")
    op.drop_table("memory_records")
