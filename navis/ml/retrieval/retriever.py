"""MemoryRetriever — the read API for semantic recall over Memory.

Answers questions like "What happened with YC?" by embedding the query, searching
the org's vectors, and reranking — returning Memory facts WITH lineage (memory_id),
never generated text. No LLM hallucination: every hit is a real, immutable fact.

Optional `since` / `until` narrow the candidate set to a `MemoryRecord.event_timestamp`
window (half-open) before ranking — required for time-scoped questions like "this
week", where pure semantic similarity would otherwise mix in older memories.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from navis.memory.lineage import lineage_by_memory_id
from navis.memory.meetings import cancelled_meeting_memory_ids
from navis.ml.embeddings.pins import EmbeddingPin, read_pin
from navis.ml.retrieval.dedupe import collapse_by_event
from navis.ml.retrieval.reranker import rerank
from navis.ml.retrieval.vector_search import Hit, search

# How many candidates the reranker gets to see, as a multiple of k.
#
# This was k*3, and that truncation — not the reranker — was what kept important
# facts out of the answer. Measured on the founder benchmark: of the
# high-importance facts available to "What am I waiting on?", ONE was inside the
# top-24 and fourteen were inside the top-200; the decision facts sat at cosine
# ranks 24, 35 and 36, one position outside the window. The reranker ranked its
# candidates correctly and never received them.
#
# 10 is derived from where answer quality stops improving, not from how deep the
# important facts go. Mean importance of the returned set by pool width:
#
#     k*3 -> 0.519   k*5 -> 0.595   k*10 -> 0.621   k*20 -> 0.621   k*50 -> 0.610
#
# It plateaus at k*10 and DEGRADES by k*50: with cosine spanning only ~0.08 across
# hundreds of ranks, a deep enough pool lets importance promote facts that are
# barely relevant at all. k*10 is the smallest width that captures the whole gain
# while keeping relevance meaningful.
#
# Widening is free. `search()` already scores every vector in the org and sorts
# them, then truncates — so a larger k returns more of an already-computed list,
# with no extra query, no extra cosine work, and (measured) no change in the
# importance lookup, which is round-trip bound at ~635ms for 24 or 160 ids alike.
CANDIDATE_MULTIPLIER = 10


class MemoryRetriever:
    """Reads the READ-pinned index — never whichever index is being written.

    During a re-embed the write pin moves to the new (model, version) while this
    stays on the current, complete one, so queries never see a half-built index.
    The query is embedded with the read pin's OWN provider: comparing a vector
    from one model against vectors from another is meaningless, so the pin
    carries the provider rather than just the model name.
    """

    def __init__(self, db: AsyncSession, *, pin: EmbeddingPin | None = None) -> None:
        self.db = db
        self.pin = pin or read_pin()
        self.provider = self.pin.provider
        self.model = self.pin.model

    async def search(
        self,
        org_id: uuid.UUID,
        query: str,
        *,
        k: int = 8,
        version: int | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> list[Hit]:
        qvec = (await self.provider.embed([query])).vectors[0]
        hits = await search(
            self.db,
            org_id,
            qvec,
            model=self.model,
            version=self.pin.version if version is None else version,
            k=k * CANDIDATE_MULTIPLIER,
            since=since,
            until=until,
        )
        # One lookup for the candidates, giving BOTH the importance weight and the
        # event each fact came from. Asked of Memory rather than Signals because
        # the ML layer may not import Signals (tests/ml/test_ml_layer.py), and
        # Memory owns this lineage. Combined into a single query because the
        # lookup is round-trip bound (~635ms) regardless of how many ids it takes.
        lineage = await lineage_by_memory_id(self.db, [h.memory_id for h in hits])
        # Absent => unscored => neutral. Never coerce a missing score to zero.
        importance = {
            memory_id: fact.importance
            for memory_id, fact in lineage.items()
            if fact.importance is not None
        }
        ranked = rerank(query, hits, importance=importance)

        # Drop facts about meetings that are currently cancelled.
        #
        # Memory is append-only, so cancelling a meeting does not remove the
        # `status: confirmed` record written when it was booked — and that
        # record has far more text (title, attendees, organiser) than the
        # cancellation stub, so similarity search reliably ranks it higher.
        # Without this filter the assistant would keep answering that a deleted
        # meeting exists, citing its own memory, and be telling the truth about
        # what it remembered while being wrong about the world.
        #
        # Suppression is a READ concern only: nothing is deleted, and anything
        # that legitimately wants to know what used to be true still can.
        cancelled = await cancelled_meeting_memory_ids(
            self.db, org_id, [h.memory_id for h in ranked]
        )
        if cancelled:
            ranked = [h for h in ranked if h.memory_id not in cancelled]

        # Collapse to one fact per event BEFORE taking the top k, so dedup frees
        # slots for other events instead of shortening the answer. This is why
        # the wider candidate pool from Phase 2B matters: there is enough material
        # behind the duplicates to fill k with distinct events.
        def event_of(hit: Hit) -> uuid.UUID | None:
            fact = lineage.get(hit.memory_id)
            return fact.source_signal_event_id if fact else None

        return collapse_by_event(ranked, event_of)[:k]
