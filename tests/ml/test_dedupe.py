"""Retrieval-time deduplication: one event, one answer slot.

Memory extracts a thread, the people on it and the relationships it evidences
from a single email. All are true and independently retrievable; showing all of
them answers the founder's question five times with the same fact.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from navis.ml.retrieval.dedupe import collapse_by_event
from navis.ml.retrieval.vector_search import Hit

NOW = datetime(2026, 8, 6, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _the_faked_index_has_rows(monkeypatch):
    """The retriever tests below fake every I/O call and return hits, so they
    model a POPULATED index — which `MemoryRetriever` now confirms before it
    embeds (`index_has_rows`). Answered here, with no database, as the fakes'
    own premise."""
    import navis.ml.retrieval.retriever as mod

    async def _has_rows(db, org_id, *, model, version):
        return True

    monkeypatch.setattr(mod, "index_has_rows", _has_rows)


def hit(text: str, cosine: float = 0.3, entity_type: str = "thread") -> Hit:
    return Hit(uuid.uuid4(), cosine, entity_type, text, NOW)


def test_one_event_occupies_one_slot():
    """The reported symptom: thread + person + relationship + relationship."""
    meeting = uuid.uuid4()
    facts = [
        hit('Email thread "Accepted: Meeting with mulayam@…"', entity_type="thread"),
        hit("Person Mulayam <mulayam@gmail.com>, sender", entity_type="person"),
        hit("Person Mulayam <mulayam@gmail.com>, sender", entity_type="person"),
        hit("Received email from mulayam@gmail.com", entity_type="relationship"),
        hit("Received email from mulayam@gmail.com", entity_type="relationship"),
    ]
    kept = collapse_by_event(facts, lambda h: meeting)
    assert len(kept) == 1


def test_the_same_entity_across_different_events_occupies_one_slot():
    """The repetition event-collapse could NOT remove, seen in production: an
    address query returned the identical string twice in one 8-slot answer,
    because a person is re-observed on every message they touch and each of
    those is a DIFFERENT signal event (measured: 5.7 records per person entity).
    The retriever therefore collapses on entity as well — same operation, other
    notion of "the same thing"."""
    person = uuid.uuid4()
    facts = [
        hit("Person Mulayam <mulayam@gmail.com>, sender", entity_type="person"),
        hit("Person mulayam@gmail.com, recipient, at gmail.com", entity_type="person"),
        hit("Person mulayam@gmail.com, recipient, at gmail.com", entity_type="person"),
    ]
    # Every fact has its own event, so event-collapse keeps all three ...
    assert len(collapse_by_event(facts, lambda h: uuid.uuid4())) == 3
    # ... while entity-collapse leaves one.
    assert collapse_by_event(facts, lambda h: person) == [facts[0]]


def test_high_multiplicity_types_cannot_crowd_out_the_rest():
    """The bias behind the repetition: person/relationship/company accumulate
    ~3-6 records per entity against ~1 for thread/meeting/document, so without
    entity-collapse they take every answer slot."""
    alice, rel, thread, meeting = (uuid.uuid4() for _ in range(4))
    facts = [
        hit("Person Alice", entity_type="person"),
        hit("Person Alice again", entity_type="person"),
        hit("Person Alice once more", entity_type="person"),
        hit("Sent email to Alice", entity_type="relationship"),
        hit("Sent email to Alice again", entity_type="relationship"),
        hit('Email thread "Q3 pricing"', entity_type="thread"),
        hit('Meeting "Board sync"', entity_type="meeting"),
    ]
    entity = dict(zip([f.memory_id for f in facts],
                      [alice, alice, alice, rel, rel, thread, meeting]))
    kept = collapse_by_event(facts, lambda h: entity[h.memory_id])
    assert [h.entity_type for h in kept] == ["person", "relationship", "thread", "meeting"]


def test_the_representative_is_the_highest_ranked_fact():
    """Input is already best-first; dedup makes no ordering judgement of its own."""
    event = uuid.uuid4()
    best, worse = hit("the informative thread"), hit("a bare person record")
    assert collapse_by_event([best, worse], lambda h: event) == [best]
    # Reverse the ranking and the representative follows it.
    assert collapse_by_event([worse, best], lambda h: event) == [worse]


def test_distinct_events_are_all_kept():
    a, b, c = hit("meeting"), hit("invoice"), hit("decision")
    events = {a.memory_id: uuid.uuid4(), b.memory_id: uuid.uuid4(), c.memory_id: uuid.uuid4()}
    assert collapse_by_event([a, b, c], lambda h: events[h.memory_id]) == [a, b, c]


def test_relative_order_of_survivors_is_preserved():
    e1, e2 = uuid.uuid4(), uuid.uuid4()
    first, dup, second = hit("first"), hit("dup of first"), hit("second")
    events = {first.memory_id: e1, dup.memory_id: e1, second.memory_id: e2}
    assert collapse_by_event([first, dup, second], lambda h: events[h.memory_id]) == [first, second]


def test_unknown_lineage_is_never_merged():
    """Absence of an event id is not evidence that two facts are the same."""
    a, b = hit("orphan a"), hit("orphan b")
    assert collapse_by_event([a, b], lambda h: None) == [a, b]


def test_unknown_lineage_does_not_block_known_duplicates():
    event = uuid.uuid4()
    orphan, x, y = hit("orphan"), hit("x"), hit("y")
    mapping = {x.memory_id: event, y.memory_id: event}
    kept = collapse_by_event([orphan, x, y], lambda h: mapping.get(h.memory_id))
    assert kept == [orphan, x]


def test_identical_text_from_DIFFERENT_events_is_kept():
    """Deliberate: three separate Drive files can legitimately share a title.
    Same-text/different-event collapsing is a different problem, not this one."""
    a, b = hit("Document \"Navis in 5 years\""), hit("Document \"Navis in 5 years\"")
    events = {a.memory_id: uuid.uuid4(), b.memory_id: uuid.uuid4()}
    assert collapse_by_event([a, b], lambda h: events[h.memory_id]) == [a, b]


def test_empty_input_is_empty_output():
    assert collapse_by_event([], lambda h: uuid.uuid4()) == []


def test_dedup_is_deterministic():
    event = uuid.uuid4()
    facts = [hit(f"fact {i}") for i in range(5)]
    assert collapse_by_event(facts, lambda h: event) == collapse_by_event(facts, lambda h: event)


def test_nothing_is_rewritten_or_invented():
    event = uuid.uuid4()
    facts = [hit("original text"), hit("duplicate")]
    kept = collapse_by_event(facts, lambda h: event)
    assert kept[0] is facts[0]
    assert kept[0].source_text == "original text"


async def test_retriever_collapses_before_taking_top_k():
    """Dedup must FREE slots for other events, not shorten the answer."""
    import navis.ml.retrieval.retriever as mod
    from navis.memory.lineage import FactLineage
    from navis.ml.embeddings.pins import EmbeddingPin
    from navis.ml.providers.deterministic import DeterministicProvider
    from navis.ml.retrieval.retriever import MemoryRetriever

    shared = uuid.uuid4()
    # 3 facts from ONE event, then 8 from distinct events.
    dupes = [hit("shared event fact", cosine=0.9) for _ in range(3)]
    singles = [hit(f"distinct event {i}", cosine=0.8) for i in range(8)]
    lineage = {h.memory_id: FactLineage(shared, 0.5) for h in dupes}
    lineage.update({h.memory_id: FactLineage(uuid.uuid4(), 0.5) for h in singles})

    async def fake_search(db, org_id, qvec, *, model, version, k, since, until):
        return dupes + singles

    async def fake_lineage(db, _org_id, ids):
        return lineage

    # Retrieval also asks Memory which of these facts are about a CANCELLED
    # meeting (so a deleted meeting can't be cited as live). Stubbed to "none"
    # here — the suppression itself is covered in test_cancelled_meetings.py.
    # Keyed on ENTITY ids: retrieval resolves candidates to entities from the
    # lineage it already holds rather than paying a round trip to re-derive them.
    async def fake_cancelled(db, org_id, entity_ids):
        return set()

    orig_s, orig_l, orig_c = (
        mod.search, mod.lineage_by_memory_id, mod.cancelled_meeting_entity_ids
    )
    mod.search, mod.lineage_by_memory_id, mod.cancelled_meeting_entity_ids = (
        fake_search, fake_lineage, fake_cancelled
    )
    try:
        pin = EmbeddingPin(DeterministicProvider(), "navis-hash-256", 1)
        out = await MemoryRetriever(None, pin=pin).search(uuid.uuid4(), "shared event fact", k=8)
    finally:
        mod.search, mod.lineage_by_memory_id, mod.cancelled_meeting_entity_ids = (
            orig_s, orig_l, orig_c
        )

    assert len(out) == 8, "dedup should free slots, not shorten the answer"
    events = {lineage[h.memory_id].source_signal_event_id for h in out}
    assert len(events) == 8, "every slot should be a distinct event"


async def test_retriever_treats_unscored_importance_as_neutral_not_zero():
    """FactLineage.importance=None must not be coerced into the importance map."""
    import navis.ml.retrieval.retriever as mod
    from navis.memory.lineage import FactLineage
    from navis.ml.embeddings.pins import EmbeddingPin
    from navis.ml.providers.deterministic import DeterministicProvider
    from navis.ml.retrieval.retriever import MemoryRetriever

    h = hit("unscored fact", cosine=0.9)
    captured = {}

    async def fake_search(db, org_id, qvec, *, model, version, k, since, until):
        return [h]

    async def fake_lineage(db, _org_id, ids):
        return {h.memory_id: FactLineage(uuid.uuid4(), None)}

    real_rerank = mod.rerank

    def spy_rerank(query, hits, **kw):
        captured["importance"] = kw.get("importance")
        return real_rerank(query, hits, **kw)

    async def fake_cancelled(db, org_id, ids):
        return set()

    orig = (
        mod.search, mod.lineage_by_memory_id, mod.rerank,
        mod.cancelled_meeting_entity_ids,
    )
    (
        mod.search, mod.lineage_by_memory_id, mod.rerank,
        mod.cancelled_meeting_entity_ids,
    ) = (fake_search, fake_lineage, spy_rerank, fake_cancelled)
    try:
        pin = EmbeddingPin(DeterministicProvider(), "navis-hash-256", 1)
        await MemoryRetriever(None, pin=pin).search(uuid.uuid4(), "unscored fact", k=8)
    finally:
        (
            mod.search, mod.lineage_by_memory_id, mod.rerank,
            mod.cancelled_meeting_entity_ids,
        ) = orig

    assert captured["importance"] == {}, "None must be absent, never 0.0"


# ---- Phase 3E.3: cancellation suppression must not cost a round trip --------
#
# Measured against the production database: server-side execution of the whole
# retrieval query is 4.3ms while ONE round trip is ~930ms, so retrieval is
# round-trip bound. The cancellation check opened with a query that mapped
# candidate memory_ids to meeting entity_ids — data the retriever already holds
# on `Hit.entity_type` and in `lineage`. These pin BOTH halves: the suppression
# still happens, and it no longer pays for what is already in memory.


async def _run_retriever(monkeypatch, hits, lineage, cancelled_entities, calls):
    import navis.ml.retrieval.retriever as mod
    from navis.ml.embeddings.pins import EmbeddingPin
    from navis.ml.providers.deterministic import DeterministicProvider
    from navis.ml.retrieval.retriever import MemoryRetriever

    async def fake_search(db, org_id, qvec, *, model, version, k, since, until):
        return list(hits)

    async def fake_lineage(db, _org_id, ids):
        return lineage

    async def fake_cancelled(db, org_id, entity_ids):
        calls.append(set(entity_ids))
        return cancelled_entities

    monkeypatch.setattr(mod, "search", fake_search)
    monkeypatch.setattr(mod, "lineage_by_memory_id", fake_lineage)
    monkeypatch.setattr(mod, "cancelled_meeting_entity_ids", fake_cancelled)
    pin = EmbeddingPin(DeterministicProvider(), "navis-hash-256", 1)
    return await MemoryRetriever(None, pin=pin).search(uuid.uuid4(), "the meeting", k=8)


async def test_a_cancelled_meetings_facts_are_still_suppressed(monkeypatch):
    """The correctness property the round-trip removal must not break: EVERY
    fact about the cancelled meeting goes, including the rich `confirmed` one
    that outranks the cancellation stub, while a live meeting stays."""
    from navis.memory.lineage import FactLineage

    dead, live = uuid.uuid4(), uuid.uuid4()
    confirmed = hit("Meeting 'Investor sync' with a@x.com", cosine=0.9, entity_type="meeting")
    stub = hit("Meeting cancelled", cosine=0.5, entity_type="meeting")
    other = hit("Meeting 'Standup'", cosine=0.4, entity_type="meeting")
    thread = hit("Email thread about the investor sync", cosine=0.8, entity_type="thread")
    lineage = {
        confirmed.memory_id: FactLineage(uuid.uuid4(), 0.5, dead),
        stub.memory_id: FactLineage(uuid.uuid4(), 0.5, dead),
        other.memory_id: FactLineage(uuid.uuid4(), 0.5, live),
        thread.memory_id: FactLineage(uuid.uuid4(), 0.5, uuid.uuid4()),
    }
    calls: list[set] = []

    out = await _run_retriever(
        monkeypatch, [confirmed, stub, other, thread], lineage, {dead}, calls
    )

    kept = {h.memory_id for h in out}
    assert confirmed.memory_id not in kept and stub.memory_id not in kept
    assert other.memory_id in kept, "a live meeting must survive"
    assert thread.memory_id in kept, "a non-meeting fact is never suppressed"
    # Asked about the MEETING entities only — resolved from lineage, not re-queried.
    assert calls == [{dead, live}]


async def test_no_meeting_candidates_costs_no_query(monkeypatch):
    """The common case. Measured on the real org, "What am I waiting on?"
    returned 13 candidates with zero meetings among them and still paid ~900ms
    to be told there was nothing to suppress."""
    from navis.memory.lineage import FactLineage

    hits = [hit("An email thread", entity_type="thread"),
            hit("A person", entity_type="person")]
    lineage = {h.memory_id: FactLineage(uuid.uuid4(), 0.5, uuid.uuid4()) for h in hits}
    calls: list[set] = []

    out = await _run_retriever(monkeypatch, hits, lineage, set(), calls)

    assert calls == [], "no meeting candidate must mean no round trip"
    assert len(out) == 2
