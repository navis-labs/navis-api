"""identity_aliases + identity_link_events + decision_dismissals

Revision ID: 0029_identity_res_dismissals
Revises: 0028_mem_org_time_index
Create Date: 2026-07-07

Three additive tables — Memory schema is UNCHANGED.

    * identity_aliases       — one row per (provider, kind, value) alias per org
    * identity_link_events   — append-only edges: two aliases refer to the same person
    * decision_dismissals    — user-dismissed decisions (Learning consumes as negative signal)

All PKs are deterministic UUIDv5, so INSERT ... ON CONFLICT DO NOTHING makes replay and
at-least-once delivery converge to identical rows. The identity tables enable KG-style
projections (aliases = nodes, link events = edges) with no further schema migration.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0029_identity_res_dismissals"
# Merges the two prior heads (0028_mem_org_time_index and
# 0027_org_profile_onboarding) so alembic advances to a single head
# after this revision. Additive only — no cross-branch schema conflicts.
down_revision: Union[str, Sequence[str], None] = (
    "0028_mem_org_time_index",
    "0027_org_profile_onboarding",
)
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # --- identity_aliases -------------------------------------------------------
    op.create_table(
        "identity_aliases",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("organization_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("provider", sa.String(length=40), nullable=False),
        sa.Column("kind", sa.String(length=40), nullable=False),
        sa.Column("value", sa.String(length=320), nullable=False),
        sa.Column("display_name", sa.String(length=320), nullable=True),
        sa.Column(
            "first_seen_at", sa.DateTime(timezone=True),
            server_default=sa.text("now()"), nullable=False,
        ),
        sa.Column(
            "last_seen_at", sa.DateTime(timezone=True),
            server_default=sa.text("now()"), nullable=False,
        ),
        sa.Column("first_seen_signal_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["organization_id"], ["organizations.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["first_seen_signal_id"], ["signal_events.id"], ondelete="SET NULL"
        ),
        sa.UniqueConstraint(
            "organization_id", "provider", "kind", "value",
            name="uq_identity_alias_natural",
        ),
    )
    op.create_index(
        "ix_identity_alias_org_value",
        "identity_aliases", ["organization_id", "kind", "value"],
    )
    op.create_index(
        "ix_identity_alias_org_provider",
        "identity_aliases", ["organization_id", "provider"],
    )

    # --- identity_link_events ---------------------------------------------------
    op.create_table(
        "identity_link_events",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("organization_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("alias_a_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("alias_b_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("source_signal_event_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=False, server_default="1.0"),
        sa.Column(
            "evidence", postgresql.JSONB(),
            nullable=False, server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("event_timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True),
            server_default=sa.text("now()"), nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["organization_id"], ["organizations.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["alias_a_id"], ["identity_aliases.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["alias_b_id"], ["identity_aliases.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["source_signal_event_id"], ["signal_events.id"], ondelete="SET NULL"
        ),
    )
    op.create_index(
        "ix_identity_link_alias_a",
        "identity_link_events", ["organization_id", "alias_a_id"],
    )
    op.create_index(
        "ix_identity_link_alias_b",
        "identity_link_events", ["organization_id", "alias_b_id"],
    )
    op.create_index(
        "ix_identity_link_signal",
        "identity_link_events", ["source_signal_event_id"],
    )

    # --- decision_dismissals ----------------------------------------------------
    op.create_table(
        "decision_dismissals",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("organization_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("decision_ref", sa.String(length=64), nullable=False),
        sa.Column("reason", sa.String(length=32), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True),
            server_default=sa.text("now()"), nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["organization_id"], ["organizations.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], ondelete="SET NULL"
        ),
        sa.UniqueConstraint(
            "organization_id", "decision_ref", "user_id",
            name="uq_decision_dismissal_natural",
        ),
    )
    op.create_index(
        "ix_decision_dismissal_org_time",
        "decision_dismissals", ["organization_id", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_decision_dismissal_org_time", table_name="decision_dismissals")
    op.drop_table("decision_dismissals")

    op.drop_index("ix_identity_link_signal", table_name="identity_link_events")
    op.drop_index("ix_identity_link_alias_b", table_name="identity_link_events")
    op.drop_index("ix_identity_link_alias_a", table_name="identity_link_events")
    op.drop_table("identity_link_events")

    op.drop_index("ix_identity_alias_org_provider", table_name="identity_aliases")
    op.drop_index("ix_identity_alias_org_value", table_name="identity_aliases")
    op.drop_table("identity_aliases")
