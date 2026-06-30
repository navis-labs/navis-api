"""Durable, content-addressed ContextBundle cache (PostgreSQL).

A cache, never a source of truth — every row is rebuildable from State (which is
rebuildable from Memory). Keyed by `context_id` (a digest of all inputs), so the same
inputs always map to the same row and an upsert is idempotent (do-nothing on conflict:
identical id ⇒ identical content). Tenant-scoped via an org FK with ON DELETE CASCADE.

Importing this module registers the ORM table; it is wired into the app only in Phase 2
(registry + migration). The service uses NullContextCache until a repository is injected.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, func, select
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from navis.context import metrics
from navis.context.contracts import CONTEXT_SCHEMA_VERSION, CONTEXT_VERSION, ContextBundle
from navis.platform.database.base import Base
from navis.platform.monitoring import get_logger

logger = get_logger("context.cache")


class ContextSnapshot(Base):
    """Content-addressed cache row for an assembled bundle."""

    __tablename__ = "context_bundles"

    # The bundle digest IS the identity (content-addressed) — no uuid4 (determinism law).
    context_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    context_version: Mapped[int] = mapped_column(Integer, nullable=False)
    schema_version: Mapped[int] = mapped_column(Integer, nullable=False)
    bundle: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class ContextCacheRepository:
    """ContextCache backed by `context_bundles`. Implements the ContextCache protocol."""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def get(self, context_id: str) -> ContextBundle | None:
        row = await self.db.scalar(
            select(ContextSnapshot).where(ContextSnapshot.context_id == context_id)
        )
        if row is None:
            return None
        # Graceful invalidation: a row whose logic/shape version no longer matches the
        # running code is treated as a MISS (the caller rebuilds deterministically). The
        # content-addressed context_id already folds context_version, so this primarily
        # guards a schema_version bump under the same id.
        if row.context_version != CONTEXT_VERSION or row.schema_version != CONTEXT_SCHEMA_VERSION:
            metrics.incr("cache.version_mismatch")
            return None
        try:
            return ContextBundle.from_dict(row.bundle)
        except Exception:
            # Corrupt cache row never propagates — degrade to a rebuild from State.
            metrics.incr("cache.corrupt")
            logger.exception("corrupt context bundle context_id=%s -> rebuild", context_id)
            return None

    async def put(self, bundle: ContextBundle) -> None:
        # Content-addressed: an existing row with this id already holds identical content,
        # so DO NOTHING on conflict is correct and idempotent.
        stmt = (
            pg_insert(ContextSnapshot)
            .values(
                context_id=bundle.context_id,
                organization_id=bundle.org_id,
                context_version=bundle.context_version,
                schema_version=bundle.schema_version,
                bundle=bundle.to_dict(),
            )
            .on_conflict_do_nothing(index_elements=["context_id"])
        )
        await self.db.execute(stmt)

    async def delete(self, org_id: uuid.UUID) -> None:
        """Dispose every cached bundle for an org (GDPR/erasure: drop so any rebuild
        reflects erased State). Org deletion is also handled by the FK CASCADE."""
        from sqlalchemy import delete as sa_delete

        await self.db.execute(
            sa_delete(ContextSnapshot).where(ContextSnapshot.organization_id == org_id)
        )
