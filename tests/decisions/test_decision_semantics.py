"""A SIGNAL IS NOT A DECISION.

Regression suite for the semantics of what may enter a founder's brief.

Production briefs were promoting raw inbound activity as "decisions":
a GoDaddy promotion, a Microsoft invoice, a Calendly notification and a
newsletter each appeared under the identical title "Active inbound
engagement", alongside "Organizational context" — a recitation of entity
counts. None of them asked anything of anyone.

Two independent guarantees keep that from coming back, and both are
tested here:

  1. REASONING already refuses to treat automated mail as a human
     obligation (engines/waiting.py filters is_automated, non-
     conversational categories and no-reply senders). These tests pin
     that behaviour so it cannot regress.

  2. DECISION only admits signal types that represent a decision point
     (decisions/base.py). Interpretation signals stay ReasoningSignals —
     available as context and evidence, never promoted to the brief.

The bar throughout: a quiet brief is correct. Fabricating a decision to
fill the page is not.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from navis.decisions.base import (
    DecisionCategory,
    category_for,
    is_decision_worthy,
)
from navis.decisions.what_matters_today import WhatMattersTodayEngine
from navis.reasoning.v2.contract import ReasoningSignal, SignalType

ORG = uuid.UUID("11111111-1111-1111-1111-111111111111")
AS_OF = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
WATERMARK = "wm-1"


def signal(
    signal_type: SignalType,
    *,
    title: str = "t",
    entity_type: str | None = None,
    entity_id: str | None = None,
    confidence: float = 0.9,
    severity: float = 0.8,
    tag: str = "s",
) -> ReasoningSignal:
    """A ReasoningSignal with real provenance, built without a DB."""
    return ReasoningSignal(
        signal_id=uuid.uuid5(uuid.NAMESPACE_URL, f"sig:{tag}"),
        org_id=ORG,
        signal_type=signal_type,
        entity_type=entity_type,
        entity_id=entity_id,
        title=title,
        why="because",
        evidence={"tag": tag},
        confidence=confidence,
        severity=severity,
        engine_name="test",
        engine_version="1.0.0",
        state_as_of=AS_OF,
        state_watermark=WATERMARK,
        reasoning_version=2,
        created_at=AS_OF,
    )


def decide(signals: list[ReasoningSignal]):
    return WhatMattersTodayEngine().decide(
        signals,
        org_id=ORG,
        reasoning_version=2,
        state_watermark=WATERMARK,
        as_of=AS_OF,
    )


# ── The classification boundary ──────────────────────────────────────

# Every interpretation type, with the production title it emitted and the
# real inbound item that produced it.
INTERPRETATION = [
    pytest.param(
        SignalType.OPPORTUNITY, "Active inbound engagement",
        id="promotional-email",
    ),
    pytest.param(
        SignalType.OPPORTUNITY, "Active inbound engagement",
        id="automated-invoice",
    ),
    pytest.param(
        SignalType.OPPORTUNITY, "Active inbound engagement",
        id="newsletter",
    ),
    pytest.param(
        SignalType.OPPORTUNITY, "Active inbound engagement",
        id="automated-calendly-notification",
    ),
    pytest.param(
        SignalType.CONTEXT_SUMMARY, "Organizational context",
        id="organizational-context",
    ),
    pytest.param(SignalType.PRIORITY_LOAD, "Open-risk load concentrated", id="priority-load"),
    pytest.param(SignalType.ORG_DRIFT, "Rising commitment volume", id="org-drift"),
    pytest.param(SignalType.LEVERAGE_POINT, "Hub company: acme.com", id="leverage-point"),
]


@pytest.mark.parametrize("signal_type,title", INTERPRETATION)
def test_interpretation_never_becomes_a_decision(signal_type, title):
    """Inbound activity and state recitations are not decisions.

    Each of these arrived in a real brief. They are true observations that
    require no judgment, approval, response, prioritization or delegation.
    """
    assert not is_decision_worthy(signal_type)
    assert category_for(signal_type) is None
    # Even at maximum confidence and severity — CONTEXT_SUMMARY ships with
    # confidence=1.0 and used to draw the largest category bias in the
    # ranker, which is exactly why it outranked real work.
    assert decide([signal(signal_type, title=title, confidence=1.0, severity=1.0)]) == []


# Every type that IS a decision point, and the judgment it demands.
DECISION_WORTHY = [
    (SignalType.WAITING_ON_USER_REPLY, DecisionCategory.WAITING_REPLY),
    (SignalType.COMMITMENT_FOLLOW_UP, DecisionCategory.FOLLOW_UP),
    (SignalType.DECISION_OWED, DecisionCategory.DECISION_OWED),
    (SignalType.COMMITMENT_SLIPPING, DecisionCategory.RISK),
    (SignalType.RELATIONSHIP_COOLING, DecisionCategory.RISK),
    (SignalType.PROJECT_STALLED, DecisionCategory.RISK),
    (SignalType.CUSTOMER_ENGAGEMENT_DROP, DecisionCategory.RISK),
    (SignalType.DECISION_UNRESOLVED, DecisionCategory.RISK),
    (SignalType.MEETING_OVERLOAD, DecisionCategory.RISK),
]


@pytest.mark.parametrize("signal_type,expected", DECISION_WORTHY)
def test_real_decision_signals_remain_eligible(signal_type, expected):
    """The fix must not silence genuine obligations."""
    assert is_decision_worthy(signal_type)
    assert category_for(signal_type) is expected
    out = decide([signal(signal_type, entity_type="thread", entity_id="e1")])
    assert len(out) == 1
    assert out[0].category is expected


def test_a_human_waiting_for_a_reply_survives_alongside_noise():
    """The end-to-end shape of the bug: one real obligation buried in
    interpretation signals. Only the obligation may reach the brief."""
    out = decide(
        [
            signal(SignalType.CONTEXT_SUMMARY, title="Organizational context",
                   confidence=1.0, tag="ctx"),
            signal(SignalType.OPPORTUNITY, title="Active inbound engagement",
                   entity_type="relationship", entity_id="godaddy", tag="promo"),
            signal(SignalType.OPPORTUNITY, title="Active inbound engagement",
                   entity_type="relationship", entity_id="microsoft", tag="invoice"),
            signal(SignalType.WAITING_ON_USER_REPLY, title="Nick is waiting for your reply",
                   entity_type="thread", entity_id="thread-nick", tag="nick"),
        ]
    )
    assert [d.title for d in out] == ["Nick is waiting for your reply"]


def test_every_signal_type_is_deliberately_classified():
    """A new SignalType must be classified, not silently promoted.

    `category_for` used to fall back to PRIORITY, so any type added to the
    enum became a decision the moment it existed.
    """
    for st in SignalType:
        classified = is_decision_worthy(st) or category_for(st) is None
        assert classified, f"{st} is not classified"


# ── Dedup: one decision per underlying thing ─────────────────────────

def test_two_signals_about_the_same_entity_produce_one_decision():
    """A customer going quiet can raise both RELATIONSHIP_COOLING and
    CUSTOMER_ENGAGEMENT_DROP off one relationship — one situation, so one
    card. The stronger signal wins."""
    out = decide(
        [
            signal(SignalType.RELATIONSHIP_COOLING, title="weaker",
                   entity_type="relationship", entity_id="acme",
                   severity=0.4, tag="cool"),
            signal(SignalType.CUSTOMER_ENGAGEMENT_DROP, title="stronger",
                   entity_type="relationship", entity_id="acme",
                   severity=0.9, tag="drop"),
        ]
    )
    assert len(out) == 1
    assert out[0].title == "stronger"


def test_distinct_entities_are_not_collapsed():
    out = decide(
        [
            signal(SignalType.WAITING_ON_USER_REPLY, entity_type="thread",
                   entity_id="t1", tag="a"),
            signal(SignalType.WAITING_ON_USER_REPLY, entity_type="thread",
                   entity_id="t2", tag="b"),
        ]
    )
    assert len(out) == 2


# ── Silence is a valid answer ────────────────────────────────────────

def test_no_signals_produce_no_decisions():
    assert decide([]) == []


def test_only_interpretation_produces_an_empty_brief():
    """A quiet brief is correct. Nothing may be invented to fill it."""
    out = decide(
        [
            signal(SignalType.CONTEXT_SUMMARY, confidence=1.0, tag="a"),
            signal(SignalType.ORG_DRIFT, tag="b"),
            signal(SignalType.LEVERAGE_POINT, tag="c"),
            signal(SignalType.OPPORTUNITY, tag="d"),
        ]
    )
    assert out == []


def test_low_confidence_signals_are_still_excluded():
    """The pre-existing confidence floor must survive the new filter."""
    assert decide([signal(SignalType.WAITING_ON_USER_REPLY, confidence=0.1)]) == []


# ── Provenance survives the filter ───────────────────────────────────

def test_surviving_decisions_remain_traceable():
    """Filtering must not cost traceability: every decision still points
    back at the exact signal, reasoning version and state watermark."""
    s = signal(SignalType.WAITING_ON_USER_REPLY, entity_type="thread", entity_id="t1")
    (d,) = decide([s])
    assert d.reasoning_signal_ids == [str(s.signal_id)]
    assert d.reasoning_version == 2
    assert d.state_watermark == WATERMARK
    assert d.created_at == AS_OF  # logical time, never now()
    assert d.evidence == s.evidence
    assert d.org_id == ORG


def test_dedup_keeps_the_provenance_of_the_signal_it_kept():
    """The surviving card must cite the signal it actually came from."""
    weak = signal(SignalType.RELATIONSHIP_COOLING, entity_type="relationship",
                  entity_id="acme", severity=0.4, tag="weak")
    strong = signal(SignalType.CUSTOMER_ENGAGEMENT_DROP, entity_type="relationship",
                    entity_id="acme", severity=0.9, tag="strong")
    (d,) = decide([weak, strong])
    assert d.reasoning_signal_ids == [str(strong.signal_id)]
    assert d.evidence == strong.evidence


def test_ranking_is_deterministic():
    signals = [
        signal(SignalType.WAITING_ON_USER_REPLY, entity_type="thread",
               entity_id=f"t{i}", severity=0.5, tag=f"s{i}")
        for i in range(5)
    ]
    assert [d.to_dict() for d in decide(signals)] == [
        d.to_dict() for d in decide(list(reversed(signals)))
    ]
