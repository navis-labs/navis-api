"""SignalRepository — the only writer/reader of the signal_events log.

Idempotency is enforced at the database with ``INSERT ... ON CONFLICT DO
NOTHING`` on the dedup constraint, so concurrent workers ingesting the same fact
never race or error — at most one row wins, the rest are no-ops.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Sequence

from sqlalchemy import and_, delete, or_, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from navis.signals.base import SignalEvent, SignalSource
from navis.signals.models import (
    SignalDeadLetter,
    SignalEventRecord,
    SignalImportance,
    SignalOutbox,
)


class SignalRepository:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def insert_idempotent(self, event: SignalEvent) -> bool:
        """Insert one event. Returns True if it was newly written, False if the
        (org, source, external_id, external_version) tuple already existed."""
        # `_row` maps the contract to columns (ORM attr is `extra`, DB column "metadata" —
        # using `metadata=` would collide with the declarative Base.metadata object).
        stmt = pg_insert(SignalEventRecord).values(self._row(event)).on_conflict_do_nothing(
            constraint="uq_signal_dedupe"
        )
        result = await self.db.execute(stmt)
        return (result.rowcount or 0) == 1

    @staticmethod
    def _row(event: SignalEvent) -> dict[str, Any]:
        return {
            "id": event.event_id,
            "organization_id": event.org_id,
            "user_id": event.user_id,
            "integration_id": event.integration_id,
            "source": event.source.value,
            "event_type": event.event_type.value,
            "external_id": event.external_id,
            "external_version": event.external_version,
            # Denormalize the channel for channel-scoped retrieval (NULL when absent). Derived
            # from metadata, never part of identity — the deterministic event_id is unchanged.
            "channel_id": event.metadata.get("channel_id"),
            "occurred_at": event.timestamp,
            "captured_at": event.captured_at or event.timestamp,
            "schema_version": event.schema_version,
            "payload": event.payload,
            "extra": event.metadata,
        }

    async def insert_many_returning_ids(self, events: Sequence[SignalEvent]) -> list[uuid.UUID]:
        """Batch-insert events in ONE multi-row `INSERT ... ON CONFLICT DO NOTHING` and
        RETURN the ids actually written (SIGNAL-REPO-002). The returned set is exactly the
        newly-persisted events — duplicates (existing rows) are excluded by RETURNING — so
        the caller can publish pointers for only the new ones.

        Multi-row upsert, NOT COPY: COPY cannot express ON CONFLICT, so a re-fetched
        overlapping window would create duplicate rows — idempotency is non-negotiable.
        Intra-batch duplicates are de-duped by event_id first (the PK is the deterministic
        UUIDv5 of the dedupe key) so the statement can't conflict with itself."""
        if not events:
            return []
        seen: set[uuid.UUID] = set()
        rows: list[dict[str, Any]] = []
        for e in events:
            if e.event_id not in seen:
                seen.add(e.event_id)
                rows.append(self._row(e))
        stmt = (
            pg_insert(SignalEventRecord)
            .values(rows)
            .on_conflict_do_nothing(constraint="uq_signal_dedupe")
            .returning(SignalEventRecord.id)
        )
        result = await self.db.execute(stmt)
        return [row[0] for row in result.all()]

    async def insert_many_idempotent(self, events: Sequence[SignalEvent]) -> int:
        """Convenience wrapper: batch-insert and return the count newly written."""
        return len(await self.insert_many_returning_ids(events))

    async def get(self, event_id: uuid.UUID) -> SignalEventRecord | None:
        return await self.db.get(SignalEventRecord, event_id)

    # ---- transactional outbox (Issue B fix) ----
    async def append_outbox(self, rows: Sequence[dict[str, Any]]) -> None:
        """Append pointer rows IN THE CALLER'S TRANSACTION (no commit) — atomic with the
        signal insert. Flush so they share the same durable commit."""
        if not rows:
            return
        self.db.add_all([SignalOutbox(**r) for r in rows])
        await self.db.flush()

    # ---- derived: importance ----
    async def append_importance(self, rows: Sequence[dict[str, Any]]) -> None:
        """Append importance scores IN THE CALLER'S TRANSACTION (no commit).

        `ON CONFLICT DO NOTHING` on the deterministic id makes re-scoring free:
        an ingest retry, a replay, or a backfill that overlaps live traffic all
        converge on the same single row per (signal, policy version).
        """
        if not rows:
            return
        await self.db.execute(
            pg_insert(SignalImportance).values(list(rows)).on_conflict_do_nothing(
                index_elements=["id"]
            )
        )

    async def pending_outbox(self, *, limit: int = 500) -> Sequence[SignalOutbox]:
        """Claim unpublished pointers FOR UPDATE SKIP LOCKED so multiple relay workers
        never publish the same row (oldest-first)."""
        stmt = (
            select(SignalOutbox)
            .where(SignalOutbox.published_at.is_(None))
            .order_by(SignalOutbox.created_at.asc())
            .limit(limit)
            .with_for_update(skip_locked=True)
        )
        return (await self.db.scalars(stmt)).all()

    async def mark_outbox_published(self, ids: Sequence[uuid.UUID]) -> None:
        if not ids:
            return
        await self.db.execute(
            update(SignalOutbox)
            .where(SignalOutbox.id.in_(list(ids)))
            .values(published_at=datetime.now(timezone.utc))
        )

    async def scrub_payload(self, event_id: uuid.UUID, payload: dict[str, Any]) -> None:
        """Governance-only in-place rewrite of a single signal's payload (retention/erasure
        tombstone). The append-only log's IMMUTABILITY is intentionally overridden ONLY here —
        retention + the legal right-to-erasure are the sanctioned exceptions; identity,
        occurred_at, and lineage are never touched, so replay + audit stay coherent. Also
        denormalize a tombstoned channel_id stays as-is (channel is not personal data)."""
        await self.db.execute(
            update(SignalEventRecord)
            .where(SignalEventRecord.id == event_id)
            .values(payload=payload)
        )

    async def scan_expired_content(
        self, *, before: datetime, limit: int = 500, after_id: uuid.UUID | None = None
    ) -> Sequence[SignalEventRecord]:
        """Page (keyset by id) signals older than `before` whose content is not yet scrubbed.
        Retention filters per-row by policy in the caller (retention_class lives in JSONB)."""
        stmt = (
            select(SignalEventRecord)
            .where(
                SignalEventRecord.occurred_at < before,
                SignalEventRecord.payload["_scrubbed"].astext.is_distinct_from("true"),
            )
            .order_by(SignalEventRecord.id)
            .limit(limit)
        )
        if after_id is not None:
            stmt = stmt.where(SignalEventRecord.id > after_id)
        return (await self.db.scalars(stmt)).all()

    async def scan_org_signals(
        self, org_id: uuid.UUID, *, limit: int = 500, after_id: uuid.UUID | None = None
    ) -> Sequence[SignalEventRecord]:
        """Page (keyset by id) all of an org's signals — erasure filters per-row by subject
        (identifiers are plaintext, so no decryption needed to find a person)."""
        stmt = (
            select(SignalEventRecord)
            .where(SignalEventRecord.organization_id == org_id)
            .order_by(SignalEventRecord.id)
            .limit(limit)
        )
        if after_id is not None:
            stmt = stmt.where(SignalEventRecord.id > after_id)
        return (await self.db.scalars(stmt)).all()

    async def purge_published_outbox(self, *, before: datetime) -> int:
        """Delete PUBLISHED outbox rows older than `before` (retention — bounds outbox
        growth at scale). Unpublished rows are never purged. Returns the count removed."""
        result = await self.db.execute(
            delete(SignalOutbox).where(
                SignalOutbox.published_at.is_not(None),
                SignalOutbox.published_at < before,
            )
        )
        return result.rowcount or 0

    async def for_replay(
        self,
        org_id: uuid.UUID,
        *,
        source: SignalSource | None = None,
        since: datetime | None = None,
        after: tuple[datetime, uuid.UUID] | None = None,
        limit: int = 1000,
    ) -> Sequence[SignalEventRecord]:
        """Read stored signals for replay, oldest-first, with a STABLE total order.

        Ordered by ``(occurred_at, id)`` — the secondary ``id`` key (deterministic UUIDv5)
        breaks same-timestamp ties so replay is reproducible (SIGNAL-REPO-001); ordering by
        ``occurred_at`` alone is non-deterministic when two facts share a timestamp.

        ``after=(occurred_at, id)`` is a KEYSET cursor (SIGNAL-REPO-003/005): pass the last
        row of the previous page to resume — no OFFSET, so it stays O(limit) at 100M+ rows
        and supports checkpointed / distributed replay."""
        stmt = select(SignalEventRecord).where(SignalEventRecord.organization_id == org_id)
        if source is not None:
            stmt = stmt.where(SignalEventRecord.source == source.value)
        if since is not None:
            stmt = stmt.where(SignalEventRecord.occurred_at >= since)
        if after is not None:
            # Keyset: strictly past (occurred_at, id) in the same total order. Expanded
            # form (vs a row-value comparison) is index-friendly and unambiguous.
            stmt = stmt.where(
                or_(
                    SignalEventRecord.occurred_at > after[0],
                    and_(
                        SignalEventRecord.occurred_at == after[0],
                        SignalEventRecord.id > after[1],
                    ),
                )
            )
        stmt = stmt.order_by(
            SignalEventRecord.occurred_at.asc(), SignalEventRecord.id.asc()
        ).limit(limit)
        return (await self.db.scalars(stmt)).all()

    async def stream_replay(
        self,
        org_id: uuid.UUID,
        *,
        source: SignalSource | None = None,
        since: datetime | None = None,
        batch_size: int = 1000,
    ) -> AsyncIterator[SignalEventRecord]:
        """Memory-safe replay (SIGNAL-REPO P2): yield events in keyset-paginated batches
        instead of materializing the whole result set — safe for multi-million-event
        replays. Internally advances the ``(occurred_at, id)`` cursor, so it is resumable
        and never holds more than one batch in memory."""
        after: tuple[datetime, uuid.UUID] | None = None
        while True:
            batch = await self.for_replay(
                org_id, source=source, since=since, after=after, limit=batch_size
            )
            if not batch:
                return
            for row in batch:
                yield row
            if len(batch) < batch_size:
                return
            last = batch[-1]
            after = (last.occurred_at, last.id)


class DeadLetterRepository:
    """PostgreSQL-backed dead-letter store — the source of truth for poison signals."""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def record(
        self,
        *,
        stage: str,
        error: str,
        org_id: uuid.UUID | None = None,
        integration_id: uuid.UUID | None = None,
        user_id: uuid.UUID | None = None,
        source: str | None = None,
        event_id: uuid.UUID | None = None,
        dedupe_key: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> SignalDeadLetter:
        row = SignalDeadLetter(
            organization_id=org_id,
            integration_id=integration_id,
            user_id=user_id,
            source=source,
            stage=stage,
            event_id=event_id,
            dedupe_key=dedupe_key,
            error=error[:8000],
            payload=payload or {},
        )
        self.db.add(row)
        await self.db.flush()
        return row

    async def list_unresolved(
        self, org_id: uuid.UUID, *, limit: int = 100
    ) -> Sequence[SignalDeadLetter]:
        stmt = (
            select(SignalDeadLetter)
            .where(
                SignalDeadLetter.organization_id == org_id,
                SignalDeadLetter.resolved_at.is_(None),
            )
            .order_by(SignalDeadLetter.created_at.asc())
            .limit(limit)
        )
        return (await self.db.scalars(stmt)).all()

    async def mark_replayed(self, ids: Sequence[uuid.UUID], *, resolved: bool) -> None:
        if not ids:
            return
        now = datetime.now(timezone.utc)
        values: dict[str, Any] = {"replayed_at": now}
        if resolved:
            values["resolved_at"] = now
        await self.db.execute(
            update(SignalDeadLetter).where(SignalDeadLetter.id.in_(list(ids))).values(**values)
        )

    async def record_replay_failure(
        self, dlq_id: uuid.UUID, *, error: str, max_attempts: int
    ) -> bool:
        """Account a FAILED replay attempt: bump attempts, record last_error, stamp
        replayed_at. Once attempts reach max_attempts the row is RESOLVED (permanently
        dead-lettered) so it leaves the retry set — the poison-loop guard. Returns True if
        the row was exhausted (permanently failed) by this attempt."""
        row = await self.db.get(SignalDeadLetter, dlq_id)
        if row is None:
            return False
        now = datetime.now(timezone.utc)
        row.attempts = (row.attempts or 0) + 1
        row.last_error = error[:8000]
        row.replayed_at = now
        exhausted = row.attempts >= max_attempts
        if exhausted:
            row.resolved_at = now  # give up: leaves list_unresolved, no more retries
        await self.db.flush()
        return exhausted

    async def purge_resolved(self, *, before: datetime) -> int:
        """Delete RESOLVED dead letters older than `before`; returns the count removed
        (SIGNAL-REPO-004 — bounds unbounded DLQ growth AND caps how long the second copy of
        PII in `payload` is retained). Unresolved poison is never purged — only rows already
        replayed/resolved. A scheduled job (orchestration) calls this with a retention cutoff."""
        result = await self.db.execute(
            delete(SignalDeadLetter).where(
                SignalDeadLetter.resolved_at.is_not(None),
                SignalDeadLetter.resolved_at < before,
            )
        )
        return result.rowcount or 0
