"""Read-side lineage: which signal produced a fact, and how important was it?

One email becomes six memory facts — a thread, two people, two relationships, a
company. All six are true, all six are separately retrievable, and when the email
matters they all score alike, so an answer can end up saying the same thing five
times:

    imp=0.9 'Email thread "Accepted: Meeting with mulayam…"'
    imp=0.9 'Person Mulayam <mulayamabmrc@gmail.com>, sender…'
    imp=0.9 'Received email from mulayamabmrc@gmail.com…'

Collapsing them needs to know which event each fact came from, which is exactly
what `memory_records.source_signal_event_id` records. Retrieval asks Memory for
it rather than reading the Signals tables, because the ML layer may not import
Signals (see `tests/ml/test_ml_layer.py`) — and because Memory owns lineage.

Importance rides along in the SAME query on purpose: the lookup is round-trip
bound (~635ms against a remote database, whether 24 ids or 160), so fetching the
two facets separately would double the cost of every search for no benefit.

Read-only. Never fabricates: a fact whose signal has not been scored comes back
with `importance=None`, and callers treat that as neutral rather than as zero.
"""
from __future__ import annotations

import uuid
from collections.abc import Iterable
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from navis.memory.models import MemoryRecord
from navis.signals.importance import IMPORTANCE_VERSION
from navis.signals.models import SignalImportance


@dataclass(frozen=True, slots=True)
class FactLineage:
    """Where a memory fact came from, what it is about, and what it was worth."""

    source_signal_event_id: uuid.UUID
    importance: float | None = None
    # Which real-world ENTITY the fact describes. Retrieval needs it to answer
    # "have I already shown the founder this person?" — the same person is
    # re-observed on every message they touch (measured: 5.7 memory records per
    # person entity), and those are all DIFFERENT signal events, so collapsing
    # by event alone cannot remove the repetition. Rides in this query because
    # it is the same row the lineage lookup already reads.
    entity_id: uuid.UUID | None = None


async def lineage_by_memory_id(
    db: AsyncSession,
    memory_ids: Iterable[uuid.UUID],
    *,
    version: int = IMPORTANCE_VERSION,
) -> dict[uuid.UUID, FactLineage]:
    """Map memory_id -> (source signal, importance) in one round trip.

    Uses an OUTER join on importance so a fact is still returned — and therefore
    still de-duplicable — when its signal has not been scored yet.
    """
    ids = list(memory_ids)
    if not ids:
        return {}
    rows = (
        await db.execute(
            select(
                MemoryRecord.memory_id,
                MemoryRecord.source_signal_event_id,
                SignalImportance.score,
                MemoryRecord.entity_id,
            )
            .outerjoin(
                SignalImportance,
                (SignalImportance.signal_event_id == MemoryRecord.source_signal_event_id)
                & (SignalImportance.version == version),
            )
            .where(MemoryRecord.memory_id.in_(ids))
        )
    ).all()
    return {
        memory_id: FactLineage(
            source_signal_event_id=signal_id,
            importance=None if score is None else float(score),
            entity_id=entity_id,
        )
        for memory_id, signal_id, score, entity_id in rows
    }


async def classifications_by_entity(
    db: AsyncSession,
    org_id: uuid.UUID,
    *,
    entity_type: str,
    version: int = IMPORTANCE_VERSION,
) -> dict[str, frozenset[str]]:
    """entity_id -> the SET of signal classifications observed in its lineage.

    The same join as `lineage_by_memory_id`, aggregated the other way: instead of
    "what was this one fact worth?", it answers "what kinds of communication has
    this entity ever been part of?".

    MONOTONE BY CONSTRUCTION. A set union over an append-only log: new evidence
    can only ADD a classification, never retract one. That is the whole point —
    `is_automated` is a per-MESSAGE property folded latest-wins onto a long-lived
    address, so one calendar RSVP can flip a real counterparty to "automated"
    and a later human reply flips it back. An address that has ever been observed
    writing a real message keeps that evidence permanently.

    Read-only, and never fabricates: an entity whose signals carry no importance
    row simply does not appear in the result, which callers must treat as UNKNOWN
    rather than as "automated" or as "human".

    Org isolation is enforced on `memory_records.organization_id`. That is
    sufficient rather than incidental: a signal_event belongs to exactly one
    organization, and the memory record carrying its id is filtered here, so no
    row from another tenant can reach the join.
    """
    rows = (
        await db.execute(
            select(MemoryRecord.entity_id, SignalImportance.classification)
            .join(
                SignalImportance,
                (SignalImportance.signal_event_id == MemoryRecord.source_signal_event_id)
                & (SignalImportance.version == version),
            )
            .where(
                MemoryRecord.organization_id == org_id,
                MemoryRecord.entity_type == entity_type,
            )
            .distinct()
        )
    ).all()
    out: dict[str, set[str]] = {}
    for entity_id, classification in rows:
        if classification:
            out.setdefault(str(entity_id), set()).add(classification)
    return {k: frozenset(v) for k, v in out.items()}


async def integrations_for_threads(
    db: AsyncSession, org_id: uuid.UUID, thread_ids: Iterable[str]
) -> dict[str, uuid.UUID]:
    """thread_id -> the ONE connected account that captured it. Ambiguity omitted.

    WHY THIS IS A LINEAGE QUESTION (T-11). `execution_requests.integration_id`
    records which of an organization's connected accounts an action must be
    performed AS, and the honest source for a reply is the account the thread
    actually arrived on. `memory_records` already carries exactly that on every
    fact — `source_integration_id`, mandatory provenance since the first
    migration — so this re-DERIVES the account from persisted state rather than
    accepting one from a caller. Same shape as the two lookups above: read-only,
    org-scoped, and it never fabricates.

    ONE QUERY for the whole brief, served by `ix_mem_entity`
    (organization_id, entity_type, entity_id). A brief proposes at most
    `brief_max_items` actions, and issuing a lookup per proposal would put that
    many round trips on the scheduled path.

    A THREAD WITH SEVERAL CAPTURING ACCOUNTS IS OMITTED, NOT RANKED. Two
    connected mailboxes can both hold the same conversation (a shared thread, a
    forwarded chain), and nothing about the thread says which identity a reply
    should come from. Omitting leaves `integration_id` NULL, which the credential
    boundary already treats as "not recorded" — and `resolve_integration` then
    refuses a multi-account organization outright. Guessing here would be the
    newest-row heuristic this whole boundary exists to remove.

    Org isolation is enforced on `memory_records.organization_id`, and
    `entity_id` is itself a UUIDv5 over the organization, so a thread id shared
    between two tenants produces two different entity ids and cannot collide.
    """
    from navis.memory.envelope import EntityType
    from navis.memory.ids import entity_id as compute_entity_id

    wanted = [t for t in dict.fromkeys(thread_ids) if t]
    if not wanted:
        return {}
    by_entity = {
        compute_entity_id(org_id, EntityType.THREAD.value, str(t)): str(t) for t in wanted
    }
    rows = (
        await db.execute(
            select(MemoryRecord.entity_id, MemoryRecord.source_integration_id)
            .where(
                MemoryRecord.organization_id == org_id,
                MemoryRecord.entity_type == EntityType.THREAD.value,
                MemoryRecord.entity_id.in_(list(by_entity)),
                MemoryRecord.source_integration_id.isnot(None),
            )
            .distinct()
        )
    ).all()
    candidates: dict[str, set[uuid.UUID]] = {}
    for entity, integration_id in rows:
        thread_id = by_entity.get(entity)
        if thread_id is not None and integration_id is not None:
            candidates.setdefault(thread_id, set()).add(integration_id)
    return {t: next(iter(ids)) for t, ids in candidates.items() if len(ids) == 1}
