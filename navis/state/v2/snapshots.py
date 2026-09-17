"""State snapshot cache table.

A snapshot is a CACHE of the projection, not a source of truth — it can be dropped
and rebuilt from Memory at any time. Keyed by (org, projection_version); upserted.
No UUID4 PK (determinism law): identity is the natural key.

Version / evolution contract (the table owns this invariant):
  * `projection_version` is part of the PK. Bumping it (a projection LOGIC change OR
    any change to the SHAPE of `full_projection`) yields a FRESH row, so the reader's
    `latest(projection_version=CURRENT)` misses the stale row and a clean rebuild from
    Memory happens automatically. Old rows are inert and harmless until vacuumed.
  * `schema_version` only TAGS the serialized shape for observability/debugging — it
    does NOT gate reads. Therefore any change to `full_projection`'s shape MUST bump
    `projection_version`; bumping `schema_version` alone would let a stale-shape blob be
    deserialized under the same key. (When in doubt, bump projection_version.)

Recovery: corruption or loss of a snapshot is always recoverable — delete the row (or
let it be overwritten) and the next read rebuilds deterministically from `memory_records`.
Nothing here is ever the system of record.

Integrity: CHECK constraints make a partial/corrupt write impossible to persist (the
watermark must be a complete cursor or empty; versions must be >= 1). They constrain
metadata only — never `full_projection` — so replay determinism is untouched.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Integer, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from navis.platform.database.base import Base


class StateSnapshot(Base):
    __tablename__ = "state_snapshots"
    __table_args__ = (
        # Integrity invariants (defense-in-depth; the writer already upholds them, this
        # makes a partial/corrupt write impossible to PERSIST). None affects determinism:
        # they constrain metadata only, never `full_projection`'s content.
        #
        # The watermark is ONE cursor (timestamp, memory_id): both set, or both NULL for an
        # empty projection. A half-set watermark is a partial-write corruption — incremental
        # folding would resume from a desynced cursor and diverge from a full rebuild.
        CheckConstraint(
            "(watermark_timestamp IS NULL) = (watermark_memory_id IS NULL)",
            name="ck_state_snapshots_watermark_paired",
        ),
        # Versions are 1-indexed and monotonic; 0/negative is always a bug.
        CheckConstraint(
            "projection_version >= 1", name="ck_state_snapshots_projection_version_positive"
        ),
        CheckConstraint(
            "schema_version >= 1", name="ck_state_snapshots_schema_version_positive"
        ),
    )

    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        primary_key=True,
    )
    projection_version: Mapped[int] = mapped_column(Integer, primary_key=True)
    schema_version: Mapped[int] = mapped_column(Integer, nullable=False)

    # OBSERVATION cursor into Memory: every record Navis had observed up to (and
    # including) this point is folded in. `watermark_timestamp` is
    # `memory_records.created_at` — when the fact was LEARNED — not
    # `event_timestamp`, which is when it happened and can be in the future for a
    # calendar meeting. Bounded by wall clock by construction; see
    # `state/v2/models.py::Watermark`. (projection_version 2 onward; version 1
    # rows carry an event-time cursor and are ignored by `latest()`.)
    watermark_timestamp: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    watermark_memory_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    as_of: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    full_projection: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)

    # Cache-write metadata (NOT part of the projection; deterministic content lives
    # entirely in full_projection).
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
