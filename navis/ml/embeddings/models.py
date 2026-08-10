"""MemoryEmbedding — append-only, versioned vectors over immutable Memory facts.

Mirrors Memory's discipline: deterministic UUIDv5 PK, never updated. Re-embedding
with a new model/version writes a NEW row (append-only history), so retrieval can
pin a version and the whole index is rebuildable from Memory + the embedder.

Vectors are stored as JSONB float arrays — portable across any Postgres and testable
today. The scale upgrade to pgvector (ANN index) is a column-type migration that
changes nothing in the service contract.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from navis.platform.database.base import Base

_EMB_NAMESPACE = uuid.UUID("b9f1c3d5-2a47-5e86-9c01-3f5a7b9d1e24")


def compute_embedding_id(memory_id: uuid.UUID, model: str, version: int) -> uuid.UUID:
    """Deterministic — same fact + model + version => same row (idempotent)."""
    return uuid.uuid5(_EMB_NAMESPACE, f"{memory_id}|{model}|{version}")


class MemoryEmbedding(Base):
    __tablename__ = "memory_embeddings"
    __table_args__ = (
        UniqueConstraint("memory_id", "model", "version", name="uq_mem_model_version"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    memory_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("memory_records.memory_id", ondelete="CASCADE"), nullable=False, index=True
    )
    model: Mapped[str] = mapped_column(String(64), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    dim: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding: Mapped[list] = mapped_column(JSONB, nullable=False)
    # Denormalized for retrieval scoping without a Memory join.
    entity_type: Mapped[str] = mapped_column(String(40), nullable=False, default="")
    source_text: Mapped[str] = mapped_column(String(2000), nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class EmbeddingBackfillProgress(Base):
    """Resumable cursor + counters for one (org, model, version) backfill.

    A re-embed is a long, interruptible job over an append-only table. Without a
    durable cursor it is neither observable ("how far along is the migration?")
    nor efficiently resumable — the previous implementation re-scanned the same
    rows on every call and could never advance past its own page limit.

    `cursor_memory_id` is the keyset position: every memory_id <= cursor has been
    considered for this index. Advancing it is idempotent because the embedding
    insert is `ON CONFLICT DO NOTHING` on a deterministic id, so replaying a page
    costs one no-op insert per row and never duplicates.
    """

    __tablename__ = "embedding_backfill_progress"
    __table_args__ = (
        UniqueConstraint("organization_id", "model", "version", name="uq_backfill_org_model_version"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    model: Mapped[str] = mapped_column(String(64), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    # Keyset position. NULL = not started; every memory_id <= this is done.
    cursor_memory_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    embedded_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    skipped_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    failed_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Set once the keyset walk reaches the end. A row with completed_at NULL and a
    # stale updated_at is the signal that a migration stalled.
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


def compute_progress_id(org_id: uuid.UUID, model: str, version: int) -> uuid.UUID:
    """Deterministic PK so concurrent workers converge on ONE progress row."""
    return uuid.uuid5(_EMB_NAMESPACE, f"progress|{org_id}|{model}|{version}")
