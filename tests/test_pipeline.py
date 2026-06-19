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
