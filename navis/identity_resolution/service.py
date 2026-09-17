"""IdentityResolutionService — subscribe to signal pointer events, extract aliases,
append to the identity graph.

Wired as an event-bus subscriber on the same pointer events Memory consumes
(GMAIL_RECEIVED, CALENDAR_EVENT_CREATED, SLACK_MESSAGE_RECEIVED, SIGNAL_INGESTED).
Both handlers run concurrently — this one reads the signal payload directly and
appends to `identity_aliases` / `identity_link_events`; it does not depend on the
Memory subscriber having finished (or run at all).

Idempotent by deterministic PK: at-least-once delivery and full replay converge
to identical rows.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Iterable

from sqlalchemy import text as sql_text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from navis.identity_resolution.ids import link_event_id
from navis.identity_resolution.models import IdentityAlias, IdentityLinkEvent
from navis.identity_resolution.resolver import Alias, extract_aliases
from navis.memory.service import SignalViewFactory
from navis.platform.database import session_scope
from navis.platform.events import DomainEvent, EventType, event_bus, subscribe
from navis.platform.monitoring import get_logger
from navis.platform.monitoring.metrics import metrics
from navis.signals.base import SignalSource
from navis.signals.repository import SignalRepository

logger = get_logger("identity_resolution")


class IdentityResolutionService:
    """Turn one SignalEvent into alias + link rows. Idempotent, provenanced, append-only."""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def ingest_signal(self, event_id: uuid.UUID) -> tuple[int, int]:
        """Read the signal, extract aliases and structural co-observations, append.

        Returns (aliases_written, links_written). Both are 0 if the source has no
        registered extractor (safe no-op) or the signal is not found (log warn).

        All appends commit ATOMICALLY before publishing IDENTITY_ALIAS_* events —
        so subscribers that read back never race the writer."""
        rec = await SignalRepository(self.db).get(event_id)
        if rec is None:
            logger.warning("identity_resolution: signal %s not found", event_id)
            return 0, 0
        view = SignalViewFactory.from_record(rec)
        aliases, pairs = extract_aliases(view.source, view.payload)
        if not aliases:
            return 0, 0

        alias_ids_written = await self._append_aliases(
            view.org_id, aliases, first_seen_signal_id=view.event_id
        )
        links_written = await self._append_links(
            view.org_id,
            pairs,
            source_signal_event_id=view.event_id,
            event_timestamp=view.event_timestamp,
        )
        await self.db.commit()  # aliases + links visible before we announce them
        if alias_ids_written or links_written:
            metrics.incr("identity.aliases_written", value=alias_ids_written)
            metrics.incr("identity.links_written", value=links_written)
            logger.info(
                "identity: signal %s -> aliases=%d links=%d org=%s",
                event_id, alias_ids_written, links_written, view.org_id,
            )
            await self._publish(view.org_id, view.event_id, aliases, pairs)
        return alias_ids_written, links_written

    async def _append_aliases(
        self,
        org_id: uuid.UUID,
        aliases: Iterable[Alias],
        *,
        first_seen_signal_id: uuid.UUID | None,
    ) -> int:
        """Idempotent per-alias insert. On duplicate, bump `last_seen_at` monotonically
        (a replay with an out-of-order event_timestamp cannot rewind last_seen_at)."""
        written = 0
        now = datetime.now(timezone.utc)
        for a in aliases:
            aid = a.compute_id(org_id)
            stmt = (
                pg_insert(IdentityAlias)
                .values(
                    id=aid,
                    organization_id=org_id,
                    provider=a.provider.value,
                    kind=a.kind.value,
                    value=a.value,
                    display_name=a.display_name,
                    first_seen_at=now,
                    last_seen_at=now,
                    first_seen_signal_id=first_seen_signal_id,
                )
                .on_conflict_do_update(
                    index_elements=["id"],
                    # Only last_seen_at is mutable — first_seen_at and display_name
                    # are set once (first observation wins) so replay converges.
                    set_={"last_seen_at": now},
                )
                # xmax == 0 on a pure INSERT; non-zero when ON CONFLICT UPDATE fired.
                # Lets us distinguish "new row" from "touched existing row" so the
                # `written` counter is a true new-row count, not touch count.
                .returning(sql_text("(xmax = 0) AS inserted"))
            )
            result = await self.db.execute(stmt)
            row = result.first()
            if row is not None and row[0]:
                written += 1
        return written

    async def _append_links(
        self,
        org_id: uuid.UUID,
        pairs: Iterable[tuple[Alias, Alias]],
        *,
        source_signal_event_id: uuid.UUID,
        event_timestamp: datetime,
    ) -> int:
        """Idempotent per-link insert. PK collapses (A,B) and (B,A) into one row."""
        written = 0
        for a, b in pairs:
            aid_a = a.compute_id(org_id)
            aid_b = b.compute_id(org_id)
            if aid_a == aid_b:  # same alias — nothing to link
                continue
            stmt = (
                pg_insert(IdentityLinkEvent)
                .values(
                    id=link_event_id(org_id, source_signal_event_id, aid_a, aid_b),
                    organization_id=org_id,
                    alias_a_id=aid_a,
                    alias_b_id=aid_b,
                    source_signal_event_id=source_signal_event_id,
                    confidence=1.0,  # structural — same-record co-observation
                    evidence={
                        "reason": "co_observation",
                        "a": {"provider": a.provider.value, "kind": a.kind.value},
                        "b": {"provider": b.provider.value, "kind": b.kind.value},
                    },
                    event_timestamp=event_timestamp,
                )
                .on_conflict_do_nothing(index_elements=["id"])
            )
            result = await self.db.execute(stmt)
            if (result.rowcount or 0) == 1:
                written += 1
        return written

    async def _publish(
        self,
        org_id: uuid.UUID,
        source_signal_event_id: uuid.UUID,
        aliases: list[Alias],
        pairs: list[tuple[Alias, Alias]],
    ) -> None:
        """Announce the new facts on the bus — additive, non-blocking for consumers.
        Payload carries ids + lineage only; consumers hit the DB for detail."""
        for a in aliases:
            aid = a.compute_id(org_id)
            await event_bus.publish(DomainEvent(
                type=EventType.IDENTITY_ALIAS_OBSERVED,
                org_id=org_id,
                source="identity_resolution",
                payload={
                    "alias_id": str(aid),
                    "provider": a.provider.value,
                    "kind": a.kind.value,
                    "source_signal_event_id": str(source_signal_event_id),
                },
            ))
        for a, b in pairs:
            aid_a = a.compute_id(org_id)
            aid_b = b.compute_id(org_id)
            if aid_a == aid_b:
                continue
            await event_bus.publish(DomainEvent(
                type=EventType.IDENTITY_ALIAS_LINKED,
                org_id=org_id,
                source="identity_resolution",
                payload={
                    "alias_a_id": str(aid_a),
                    "alias_b_id": str(aid_b),
                    "source_signal_event_id": str(source_signal_event_id),
                },
            ))


# --- Subscriber wiring -------------------------------------------------------
# Subscribes to the SAME pointer events Memory consumes. Both handlers run
# concurrently on the same event; identity resolution reads the signal directly
# (via SignalRepository) and never depends on Memory ordering.

async def _on_signal_pointer(event: DomainEvent) -> None:
    payload = event.payload or {}
    event_id = payload.get("event_id")
    source_value = payload.get("source")
    if not event_id or not source_value:
        return
    try:
        source = SignalSource(source_value)
    except ValueError:
        return
    # Fast filter: skip sources with no extractor without opening a DB session.
    if source not in {SignalSource.GMAIL, SignalSource.CALENDAR, SignalSource.SLACK}:
        return
    async with session_scope() as db:
        await IdentityResolutionService(db).ingest_signal(uuid.UUID(event_id))


# EVERY calendar pointer, not just CREATED — and that is not a widening, it is
# the list this module always claimed to have.
#
# `SignalIngestionService` routes by the ACTUAL event_type, so a calendar signal
# arrives as one of THREE bus events, and `CalendarConnector._event_type` only
# returns CREATED when `raw["created"] == raw["updated"]` — an event never touched
# since it was made. Every edited or re-synced event is UPDATED, and a deletion is
# CANCELLED. Subscribing to CREATED alone therefore dropped the normal case.
#
# Measured against production on 2026-08-21, this was not a corner case: all NINE
# calendar signals were `calendar.event.updated`, every one carried a populated
# `attendees` array with `email` keys, all nine were captured AFTER this
# subscriber went live — and `identity_aliases` held 48 rows, every one of them
# `provider='gmail'`. Not a single calendar attendee had ever entered the identity
# graph, so a person you only ever MEET (never email) was invisible to
# `/identity/persons/*` and to the chat scheduling agent that reads them.
#
# `memory.service` subscribes to all four and always did; the docstring above
# promising "the SAME pointer events Memory consumes" was describing that list,
# not this one. These two lines make the sentence true.
#
# Purely additive: `_calendar_aliases` emits aliases and NO co-observation pairs,
# so this can only ever append alias rows — it cannot create a link edge and
# therefore cannot merge two people. Inserts are idempotent on the deterministic
# PK, so re-delivery and replay converge.
for _evt in (
    EventType.GMAIL_RECEIVED,
    EventType.CALENDAR_EVENT_CREATED,
    EventType.CALENDAR_EVENT_UPDATED,
    EventType.CALENDAR_EVENT_CANCELLED,
    EventType.MEETING_COMPLETED,
    EventType.SLACK_MESSAGE_RECEIVED,
    EventType.SIGNAL_INGESTED,
):
    subscribe(_evt)(_on_signal_pointer)
