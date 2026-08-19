"""Reality-safe versioning + engine provenance on intelligence_insights.

Intelligence has only ever consumed Learning, so an insight's version dimension
was `learning_version` — the count of learning events. That is correct for a
fact derived from the outcome log and wrong for one derived from CURRENT
organizational reality, which moves when State moves, not when an outcome is
recorded. Keyed on `learning_version`, every reality insight for an org would
collapse onto one id and — the table being append-only with ON CONFLICT DO
NOTHING — freeze at whatever was true the first time it was projected.

`state_watermark` becomes that second version dimension (see
`compute_insight_id`), which is why it is promoted from a key inside the
`evidence` JSONB to a real column: it now has to be queryable and indexable.
`engine_name` / `engine_version` answer "which code produced this row, at what
version" — `reasoning_signals` has carried both since it existed and
`intelligence_insights` never did.

STRICTLY ADDITIVE. Three nullable columns and one index. No backfill: a row
written before this migration does not know which State snapshot it described
or which generator version produced it, and writing a guess would be fabricated
provenance. No existing row is read, rewritten or locked for rewrite, and no
insight_id changes — the id key string is unchanged.

Revision ID: 0039_intelligence_reality_input
Revises: 0038_pgvector_embeddings
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0039_intelligence_reality_input"
down_revision = "0038_pgvector_embeddings"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "intelligence_insights",
        sa.Column("state_watermark", sa.String(64), nullable=True),
    )
    op.add_column(
        "intelligence_insights",
        sa.Column("engine_name", sa.String(64), nullable=True),
    )
    op.add_column(
        "intelligence_insights",
        sa.Column("engine_version", sa.String(16), nullable=True),
    )
    # "What did Intelligence conclude about this org at this watermark?" — the
    # replay question, and the one a reality-derived insight is looked up by.
    # Tenant-first so an org filter is always index-served and a watermark can
    # never be resolved across organizations.
    op.create_index(
        "ix_intelligence_insights_org_watermark",
        "intelligence_insights",
        ["organization_id", "state_watermark"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_intelligence_insights_org_watermark",
        table_name="intelligence_insights",
    )
    op.drop_column("intelligence_insights", "engine_version")
    op.drop_column("intelligence_insights", "engine_name")
    op.drop_column("intelligence_insights", "state_watermark")
