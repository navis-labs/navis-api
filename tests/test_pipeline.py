"""End-to-end pipeline (no DB): Memory -> State V2 -> Reasoning V2 -> Decision.

Proves the layers compose deterministically through their real contracts.
"""
from __future__ import annotations

from navis.decisions.base import DecisionCategory
from navis.decisions.what_matters_today import WhatMattersTodayEngine
from navis.memory.envelope import EntityType
from navis.memory.service import build_memory_records
from navis.reasoning.v2 import REASONING_VERSION, run_reasoning_v2
from navis.state.v2.builders import build_projection
from tests.memory.fakes import gmail_view
from tests.state.fakes import ORG_A, corpus, rec


def test_memory_extracts_people_from_email():
    records = build_memory_records(
        gmail_view(payload={"from": "Jane Doe <jane@acme.com>", "to": "me@startup.com",
                            "body_text": "", "thread_id": ""})
    )
    emails = {r.payload["email"] for r in records if r.entity_type == EntityType.PERSON}
    assert "jane@acme.com" in emails and "me@startup.com" in emails


def test_memory_extracts_commitment_promise():
    records = build_memory_records(
        gmail_view(payload={"from": "founder@startup.com", "subject": "Re: pricing",
                            "body_text": "Thanks for the call. I'll send you the revised proposal by Friday.",
                            "to": "", "thread_id": ""})
    )
    commitments = [r for r in records if r.entity_type == EntityType.COMMITMENT]
    assert commitments and "send you" in commitments[0].payload["text"].lower()


def test_state_reasoning_decision_compose_deterministically():
    # Stale corpus so risk signals fire (heartbeat pushes as_of ~40 days ahead).
    recs = corpus(ORG_A) + [
        rec(EntityType.PERSON, "hb@acme.com", {"email": "hb@acme.com", "role": "from"},
            offset=40 * 86400, tag="hb", org_id=ORG_A)
    ]
    state = build_projection(ORG_A, recs)
    signals = run_reasoning_v2(state)
    wm = str(state.watermark.last_memory_id)

    engine = WhatMattersTodayEngine()
    a = engine.decide(signals, org_id=ORG_A, reasoning_version=REASONING_VERSION,
                      state_watermark=wm, as_of=state.as_of)
    b = engine.decide(signals, org_id=ORG_A, reasoning_version=REASONING_VERSION,
                      state_watermark=wm, as_of=state.as_of)

    assert a, "expected decisions from the brief"
    assert [d.to_dict() for d in a] == [d.to_dict() for d in b]  # deterministic
    # Ranks are dense 1..N and every decision is traceable + categorized.
    assert [d.priority_rank for d in a] == list(range(1, len(a) + 1))
    for d in a:
        assert d.org_id == ORG_A
        assert d.reasoning_signal_ids and d.reasoning_version == REASONING_VERSION
        assert d.state_watermark == wm
        assert d.created_at == state.as_of  # logical clock, not now()
        assert isinstance(d.category, DecisionCategory)


# ── THE LINEAGE CHAIN ──────────────────────────────────────────────────────
#
# "Why is Navis showing me this?" is answerable only if every hop names the row
# the hop before it wrote. This walks one real Gmail message all the way to the
# Today card it becomes and asserts the IDs actually chain — not that each layer
# runs, but that each layer's output points back at the previous layer's row.
#
# It is deliberately the WAITING_ON_USER_REPLY path: it is the archetypal Today
# card, it is the only decision category the proposer can currently turn into a
# real executable action, and it is one of the two decision-worthy signal types
# no Intelligence generator consumes (pinned in tests/intelligence). So this also
# pins the honest shape of that gap — a decision that reaches Today with an empty
# `supporting_insights`, rather than one dressed in borrowed corroboration.


def _waiting_thread_records(days_ago: int = 6):
    """One inbound Gmail message, through the real memory extractors."""
    from datetime import timedelta

    from navis.memory.service import build_memory_records
    from tests.memory.fakes import EVENT_TS

    view = gmail_view(
        payload={
            "from": "Nick Alvarez <nick@acme.com>",
            "to": "founder@navis.io",
            "sender_email": "nick@acme.com",
            "sender_name": "Nick Alvarez",
            "subject": "Re: pricing for the Q4 rollout",
            "thread_id": "19fdb57e89581d43",
            "rfc_message_id": "<CAF9x1n@mail.gmail.com>",
            "body_text": "Could you confirm the discount tier before we sign?",
            "label_ids": ["INBOX"],
        },
        event_timestamp=EVENT_TS - timedelta(days=days_ago),
    )
    return view, build_memory_records(view)


def test_one_gmail_message_becomes_one_today_card_with_an_unbroken_id_chain():
    import uuid

    from navis.decisions.what_matters_today import WhatMattersTodayEngine
    from navis.execution.proposer import decision_action
    from navis.state.v2.builders import MemoryRecordView, build_projection
    from tests.memory.fakes import EVENT_TS, ORG

    view, records = _waiting_thread_records()

    # 1. SIGNAL -> MEMORY. Every record names the signal event it came from.
    thread_records = [r for r in records if r.entity_type is EntityType.THREAD]
    assert thread_records, "the message produced no thread fact"
    [thread_record] = thread_records
    assert thread_record.source_signal_event_id == view.event_id

    # 2. MEMORY -> STATE. The fold carries the record's own id as provenance, and
    #    the watermark IS a memory_id — not a timestamp, not a sequence number.
    #
    #    Every record shares one `observed_at`, which is what real ingestion does:
    #    `memory_records.created_at` defaults to `now()`, and Postgres `now()` is
    #    the TRANSACTION timestamp — so one signal's facts are one observation.
    state = build_projection(
        ORG,
        [
            MemoryRecordView(
                memory_id=r.memory_id, org_id=r.org_id, entity_type=r.entity_type.value,
                entity_id=r.entity_id, event_timestamp=r.event_timestamp,
                payload=r.payload, source_signal_event_id=r.source_signal_event_id,
                schema_version=r.schema_version, confidence=r.confidence,
                observed_at=EVENT_TS,
            )
            for r in records
        ],
        as_of=EVENT_TS,
    )
    [thread] = list(state.threads())
    watermark = str(state.watermark.last_memory_id)
    assert thread.entity_id == str(thread_record.entity_id)
    assert str(thread_record.memory_id) in thread.provenance["memory_ids"]
    assert str(view.event_id) in thread.provenance["signal_ids"]
    assert watermark in {str(r.memory_id) for r in records}

    # 3. STATE -> REASONING. The conclusion names the State entity and the
    #    snapshot it was true of, and it is the thread's DIRECTION that produced
    #    it — the fact, not a guess about the sender.
    signals = run_reasoning_v2(state)
    waiting = [s for s in signals if s.signal_type.value == "waiting_on_user_reply"]
    assert waiting, "an unanswered inbound thread produced no waiting signal"
    [s] = waiting
    assert thread.direction == "inbound"
    assert s.entity_type == "thread" and s.entity_id == thread.entity_id
    assert s.state_watermark == watermark
    assert s.state_as_of == state.as_of  # logical time, never wall clock
    assert s.evidence["thread_id"] == "19fdb57e89581d43"

    # 4. REASONING -> DECISION. The decision names the exact signal, and its own
    #    identity is derived from it — so the chain is verifiable, not asserted.
    from navis.decisions.base import compute_decision_id

    # 3b. STATE -> CONTEXT. A pure selection over State, carrying each entity's
    #     citations. This is the hop that lets Intelligence name the memory record
    #     and the business-system event behind an entity.
    from navis.orchestration.brief import _context_rows, _intelligence_view, _reality_view
    from navis.state.v2 import MorningBrief, scope_for

    context = scope_for(state, MorningBrief(), None, as_of=state.as_of)
    assert context.state_watermark == watermark
    [ctx_thread] = [c for c in _context_rows(context) if c.entity_type == "thread"]
    assert ctx_thread.entity_id == thread.entity_id
    assert str(thread_record.memory_id) in ctx_thread.memory_ids
    assert str(view.event_id) in ctx_thread.signal_event_ids

    # 3c. CONTEXT + REASONING -> INTELLIGENCE. The obligation family consumes
    #     this signal type; before it existed the projection was empty here.
    from navis.intelligence.insights import LearningView
    from navis.intelligence.insights import project as project_insights
    from navis.intelligence.service import draft_to_values

    learning = LearningView(org_id=ORG, learning_version=0, as_of=None,
                            signal_aggs=[], decision_aggs=[], outcome_aggs=[], patterns=[])
    reality = _reality_view(ORG, state, signals, context)
    insight_rows = [
        draft_to_values(learning, draft, reality)
        for draft in project_insights(learning, reality)
    ]
    [obligation] = [
        r for r in insight_rows
        if r["insight_type"] == "obligation_intelligence"
        and r["key"] == "waiting_on_user_reply"
    ]
    assert obligation["evidence"]["supporting_reasoning_signal_ids"] == [str(s.signal_id)]
    assert obligation["state_watermark"] == watermark
    # ... and the insight cites the memory record and the captured Gmail event.
    [named] = obligation["evidence"]["entities"]
    assert named["entity_id"] == thread.entity_id
    assert named["memory_ids"] == [str(thread_record.memory_id)]
    assert named["signal_event_ids"] == [str(view.event_id)]

    class _Row:
        def __init__(self, values):
            self.__dict__.update(values)

    [d] = [
        x
        for x in WhatMattersTodayEngine().decide(
            signals, org_id=ORG, reasoning_version=REASONING_VERSION,
            state_watermark=watermark, as_of=state.as_of,
            intelligence=_intelligence_view(ORG, [_Row(r) for r in insight_rows]),
        )
        if x.signal_type == "waiting_on_user_reply"
    ]
    assert d.reasoning_signal_ids == [str(s.signal_id)]
    assert d.decision_id == compute_decision_id(
        ORG, REASONING_VERSION, watermark, uuid.UUID(str(s.signal_id))
    )
    assert d.category is DecisionCategory.WAITING_REPLY

    # 5. ONE SNAPSHOT, ALL THE WAY DOWN. Reasoning, Decision and State must not
    #    silently describe different organizations.
    assert d.state_watermark == watermark == s.state_watermark
    assert d.created_at == state.as_of == s.created_at
    assert d.reasoning_version == s.reasoning_version == REASONING_VERSION

    # 6. THE DECISION CITES THE INTELLIGENCE THAT WAS FOLDED FROM ITS OWN SIGNAL,
    #    and says which snapshot that insight describes. This assertion used to
    #    read `== []`: no generator consumed `waiting_on_user_reply`, so the most
    #    actionable card on Today reached the founder with no Intelligence behind
    #    it at all.
    payload = d.to_dict()
    assert payload["supporting_insights"] == [
        {
            "insight_id": str(obligation["insight_id"]),
            "relation": "derived_from_same_signal",
            "insight_type": "obligation_intelligence",
            # Intelligence's own headline, verbatim — the citation a founder can
            # read instead of a UUID.
            "title": "1 outstanding reply obligation",
            "sample_size": 1,
            "state_watermark": watermark,
            "describes_current_snapshot": True,
        }
    ]

    # 7. DECISION -> EXECUTION. The action is the registered one, it names the
    #    obligation by the message it answers, and it cannot run without a human.
    action = decision_action(payload)
    assert action is not None
    assert action.action_type == "reply_to_email" and action.provider == "gmail"
    assert action.requires_approval is True
    assert "19fdb57e89581d43" in action.obligation_key
    assert "CAF9x1n@mail.gmail.com" in action.obligation_key


def test_the_card_disappears_when_the_founder_replies_no_teardown_needed():
    """The loop closes through the business system, not through a Navis write.

    A sent reply re-enters as an OUTBOUND Gmail signal, the fold flips
    `ThreadState.direction`, and the engine stops firing. Nothing marks the
    decision resolved and nothing needs to — which is why `reply_to_email`
    declares no reconciler while the calendar verbs do.
    """
    import uuid
    from datetime import timedelta

    from navis.memory.service import build_memory_records
    from navis.signals.base import SignalEventType, SignalSource
    from navis.state.v2.builders import MemoryRecordView, build_projection
    from tests.memory.fakes import EVENT_TS, ORG, make_view

    _view, inbound = _waiting_thread_records()
    # The org's OWN message, captured by the same connector — Google stamps
    # outbound mail SENT and `ThreadExtractor` records that as `direction`.
    reply = make_view(
        source=SignalSource.GMAIL,
        event_type=SignalEventType.EMAIL_RECEIVED,
        payload={
            "from": "founder@navis.io",
            "to": "nick@acme.com",
            "sender_email": "founder@navis.io",
            "subject": "Re: pricing for the Q4 rollout",
            "thread_id": "19fdb57e89581d43",
            "body_text": "Confirmed — tier 2 applies.",
            "label_ids": ["SENT"],
        },
        event_id=uuid.uuid5(uuid.NAMESPACE_URL, "reply"),
        external_id="msg-reply",
        event_timestamp=EVENT_TS - timedelta(minutes=5),
    )

    def project(records):
        return build_projection(
            ORG,
            [
                MemoryRecordView(
                    memory_id=r.memory_id, org_id=r.org_id, entity_type=r.entity_type.value,
                    entity_id=r.entity_id, event_timestamp=r.event_timestamp,
                    payload=r.payload, source_signal_event_id=r.source_signal_event_id,
                    schema_version=r.schema_version, confidence=r.confidence,
                )
                for r in records
            ],
            as_of=EVENT_TS,
        )

    def waiting(records):
        return [
            s for s in run_reasoning_v2(project(records))
            if s.signal_type.value == "waiting_on_user_reply"
        ]

    assert waiting(inbound), "precondition: the card exists before the reply"
    assert not waiting(inbound + build_memory_records(reply)), (
        "the reply landed and the thread still reads as waiting"
    )


# ══════════════════════════════════════════════════════════════════════════════
# 7.5.7 ACCEPTANCE — the real AJITESH case, end to end
#
# Reconstructed from the read-only production trace of org 071ae55b: a Gmail
# message in which the counterparty promised to come back, and a relationship
# with the same person that had gone quiet. Production served these as TWO
# UNRELATED CARDS whose only shared "organizational context" was a footprint
# insight naming 143 entities — while State had recorded
# `relationship --commitments--> commitment` the whole time.
#
# This walks the real contracts (no DB, every layer pure) and asserts the
# connection survives all the way to what the founder reads.
# ══════════════════════════════════════════════════════════════════════════════

AJITESH = "ajitesh@example.com"


def _ajitesh_state():
    """The production entity shape: person + relationship + commitment + meetings,
    joined by the values the extractors really record (`speaker_email` matching
    `person_email`, and the meeting's attendee list)."""
    records = [
        rec(EntityType.PERSON, AJITESH, {"email": AJITESH, "name": "Ajitesh"},
            offset=0, tag="aj-p"),
        rec(EntityType.RELATIONSHIP, AJITESH,
            {"person_email": AJITESH, "direction": "outbound"}, offset=1, tag="aj-r"),
        rec(EntityType.COMMITMENT, "aj-c",
            {"text": "Thanks for following up on this — I'll come back to you on it.",
             "speaker_email": AJITESH, "promised_at": "2026-06-01T12:00:01+00:00"},
            offset=1, tag="aj-c"),
        rec(EntityType.MEETING, "aj-m1",
            {"title": "Nick and Abhi", "attendees": [AJITESH]}, offset=2, tag="aj-m1"),
        rec(EntityType.MEETING, "aj-m2",
            {"title": "Intro", "attendees": [AJITESH]}, offset=2, tag="aj-m2"),
        # Moves the clock far enough past the promise that BOTH the commitment
        # (>5d) and the relationship (>7d) cross their reasoning thresholds.
        rec(EntityType.PERSON, "someone-else@example.com",
            {"email": "someone-else@example.com", "name": "Other"},
            offset=60 * 60 * 24 * 12, tag="aj-clock"),
    ]
    return build_projection(ORG_A, records)


def _ajitesh_pipeline():
    """State -> Context -> Reasoning -> Intelligence -> Decision -> explanation."""
    from types import SimpleNamespace

    from navis.decisions.explanation import explain
    from navis.intelligence.insights import LearningView
    from navis.intelligence.service import project_values
    from navis.orchestration.brief import _intelligence_view, _reality_view
    from navis.state.v2.scope import MorningBrief, scope_for

    state = _ajitesh_state()
    context = scope_for(state, MorningBrief(), None, as_of=state.as_of)
    # The organizational-relevance evidence the composition root reads from
    # `memory.lineage.classifications_by_entity`; production recorded exactly
    # ["commitment", "meeting"] for this relationship.
    rel_id = next(
        e.entity_id for e in context.entities if e.entity_type == "relationship"
    )
    signals = run_reasoning_v2(state, evidence={rel_id: {"commitment", "meeting"}})
    reality = _reality_view(ORG_A, state, signals, context)

    empty_learning = LearningView(
        org_id=ORG_A, learning_version=0, as_of=None, state_watermark=None,
        signal_aggs=[], decision_aggs=[], outcome_aggs=[], patterns=[],
    )
    rows = [SimpleNamespace(**v) for v in project_values(empty_learning, reality)]
    decisions = WhatMattersTodayEngine().decide(
        signals, org_id=ORG_A, reasoning_version=REASONING_VERSION,
        state_watermark=str(state.watermark.last_memory_id), as_of=state.as_of,
        intelligence=_intelligence_view(ORG_A, rows),
    )
    payloads = [d.to_dict() for d in decisions]
    return {
        "state": state, "context": context, "signals": signals, "reality": reality,
        "insights": [v for v in project_values(empty_learning, reality)],
        "decisions": payloads,
        "explanations": {p["decision_id"]: explain(p) for p in payloads},
    }


def test_the_real_case_reaches_both_conclusions():
    got = _ajitesh_pipeline()
    types = {s.signal_type.value for s in got["signals"]}
    assert "commitment_slipping" in types
    assert "relationship_cooling" in types


def test_context_hands_intelligence_the_edges_state_recorded():
    """The hop that did not exist: `build_brief` never called `scope_for`, and
    `EntityRef` carried no links even when it did."""
    reality = _ajitesh_pipeline()["reality"]
    edges = {
        (row.entity_type, link.role, link.entity_type)
        for row in reality.context
        for link in row.links
    }
    assert ("relationship", "commitments", "commitment") in edges
    assert ("relationship", "meetings", "meeting") in edges
    assert ("commitment", "owner", "person") in edges


def test_intelligence_produces_one_connected_situation_not_only_counts():
    got = _ajitesh_pipeline()
    situations = [
        i for i in got["insights"] if i["insight_type"] == "situation_intelligence"
    ]
    assert len(situations) == 1
    title = situations[0]["title"]
    assert AJITESH in title
    assert "unresolved commitment" in title and "cooling relationship" in title
    # The old-world answer must still exist — this is additive, not a swap.
    assert any(i["insight_type"] == "risk_intelligence" for i in got["insights"])


def test_the_situation_cites_both_signals_and_reaches_both_decisions():
    """ONE organizational sentence, corroborating the commitment card AND the
    relationship card — because it names both of their signals."""
    got = _ajitesh_pipeline()
    situation = next(
        i for i in got["insights"] if i["insight_type"] == "situation_intelligence"
    )
    cited = set(situation["evidence"]["supporting_reasoning_signal_ids"])
    assert len(cited) == 2

    carrying = [
        d for d in got["decisions"]
        if any(
            s["insight_id"] == str(situation["insight_id"])
            for s in d["supporting_insights"]
        )
    ]
    assert {d["entity_type"] for d in carrying} == {"commitment", "relationship"}


def test_the_founder_facing_context_is_the_situation_not_a_bucket_count():
    """THE ACCEPTANCE CRITERION. Production's only organizational explanation was
    "2 open commitment slipping risks"; it must now be the connected sentence."""
    got = _ajitesh_pipeline()
    statements = {
        e["organizational_context"]["statement"]
        for e in got["explanations"].values()
        if e.get("organizational_context")
    }
    assert statements, "every decision must carry organizational context"
    for statement in statements:
        assert AJITESH in statement
        assert "open commitment slipping risks" not in statement


def test_the_situation_is_traceable_to_memory_and_business_system_events():
    """Every sentence must be evidence-backed all the way down to the captured
    Gmail/Calendar event — the hop insight provenance used to stop short of."""
    got = _ajitesh_pipeline()
    situation = next(
        i for i in got["insights"] if i["insight_type"] == "situation_intelligence"
    )
    evidence = situation["evidence"]
    state = got["state"]

    assert evidence["memory_ids"] and evidence["signal_event_ids"]
    known_memory = {
        m
        for bucket in state.entities.values()
        for raw in bucket.values()
        for m in raw["provenance"]["memory_ids"]
    }
    # Nothing invented: every cited record is one State actually folded.
    assert set(evidence["memory_ids"]) <= known_memory


def test_nothing_outside_the_persisted_graph_is_claimed():
    """No company, customer, project or role appears — State recorded none."""
    got = _ajitesh_pipeline()
    situation = next(
        i for i in got["insights"] if i["insight_type"] == "situation_intelligence"
    )
    named = {r["entity_type"] for r in situation["evidence"]["related_entities"]}
    assert named <= {"commitment", "meeting", "person"}


def test_existing_decision_behaviour_is_unchanged_by_the_situation():
    """Title, why, evidence, confidence and severity still come from Reasoning
    verbatim. Intelligence corroborates; it never rewrites the decision."""
    got = _ajitesh_pipeline()
    by_signal = {str(s.signal_id): s for s in got["signals"]}
    for d in got["decisions"]:
        source = by_signal[d["reasoning_signal_ids"][0]]
        assert d["title"] == source.title
        assert d["why"] == source.why
        assert d["evidence"] == source.evidence
        assert d["confidence"] == round(source.confidence, 4)
        assert d["severity"] == round(source.severity, 4)
