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


# ---- reasoning provenance is durable (Phase 3F.1) ---------------------------
#
# Execution, Observation, Learning and Intelligence all persist
# `reasoning_signal_ids`. The signals themselves lived only in process memory —
# Decision copied the few that became decisions into the brief and the rest were
# discarded — so those stored ids pointed at nothing.
#
# Recomputation cannot stand in for storage: `compute_signal_id` hashes
# `reasoning_version`, so bumping it (2 -> 3) permanently changes every id. The
# one reasoning reference in production today was written at v2 and can never be
# re-derived.


def test_brief_persists_reasoning_provenance_before_decision_consumes_it():
    """Remove the write and downstream references silently dangle again."""
    import ast
    import pathlib

    src = (pathlib.Path(__file__).resolve().parents[2]
           / "navis" / "orchestration" / "brief.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "build_brief"
    )
    calls = [
        n.func.id if isinstance(n.func, ast.Name) else getattr(n.func, "attr", "")
        for n in ast.walk(fn) if isinstance(n, ast.Call)
    ]
    assert "persist_signals" in calls, "build_brief no longer stores reasoning provenance"
    # Order matters: provenance must exist before Decision can reference it.
    assert calls.index("persist_signals") < calls.index("decide"), (
        "provenance must be written before Decision consumes the signals"
    )
    # ...and before Intelligence is handed the same signals as current reality,
    # so an insight can never cite a signal that has not been written yet.
    assert "_project_intelligence" in calls, (
        "build_brief no longer feeds Intelligence the organization's current "
        "reality; Intelligence falls back to describing historical outcomes only"
    )
    assert "_reality_view" in calls, "the reality projection is no longer built"
    assert calls.index("persist_signals") < calls.index("_project_intelligence"), (
        "reasoning signals must be persisted before Intelligence receives them"
    )


def test_reasoning_engines_stay_pure_of_the_store():
    """The store deliberately lives OUTSIDE navis/reasoning/v2 so the engines
    remain a pure function of State — writes are driven from orchestration."""
    import pathlib

    v2 = pathlib.Path(__file__).resolve().parents[2] / "navis" / "reasoning" / "v2"
    for py in v2.rglob("*.py"):
        assert "reasoning.store" not in py.read_text(encoding="utf-8"), (
            f"{py.name} imports the persistence layer; v2 must stay pure"
        )


# ── Phase 4.2 — Intelligence → Decision, the contract and the wiring ──────────
#
# Decision has only ever ranked ReasoningSignal[]. Intelligence holds what the
# organization has LEARNED about itself, and Decision may not import it — the
# dependency would invert the pipeline. So the facts arrive as a flat,
# primitive-only view built by orchestration, exactly as `OrganizationalReality`
# View reaches Intelligence.
#
# This phase wires the join and deliberately does NOT rank on it. These tests pin
# both halves: the view really reaches Decision, and nothing about the existing
# output moved because it did.

import uuid as _uuid
from types import SimpleNamespace

from navis.decisions.base import IntelligenceFact, IntelligenceView

# This module's single org constant, plus a second one for the isolation guard.
ORG_A = ORG
ORG_B = uuid.UUID("22222222-2222-2222-2222-222222222222")


def _insight_row(**over):
    """One stored insight, shaped exactly as IntelligenceRepository returns it."""
    fields = {
        "insight_id": _uuid.uuid5(_uuid.NAMESPACE_OID, "insight-1"),
        "organization_id": ORG_A,
        "insight_type": "risk_intelligence",
        "key": "relationship_cooling",
        "title": "Relationship cooling",
        "summary": "Reasoning currently identifies 2 entities with relationship cooling.",
        "confidence": 0.9,
        "sample_size": 2,
        "evidence": {
            "affected": 2,
            "supporting_reasoning_signal_ids": ["sig-a", "sig-b"],
            "state_watermark": "wm-1",
        },
        "supporting_learning_event_ids": [],
        "learning_version": 7,
        "state_watermark": "wm-1",
        "created_at": datetime(2026, 8, 17, 12, 0, tzinfo=timezone.utc),
    }
    fields.update(over)
    return SimpleNamespace(**fields)


# 2. Decision does not import Intelligence.

def test_decision_domain_never_imports_intelligence():
    """The whole reason the view exists. An import here inverts the pipeline and
    makes the ranker a reader of an append-only projection."""
    import ast
    import pathlib

    pkg = pathlib.Path(__file__).resolve().parents[2] / "navis" / "decisions"
    violations = []
    for py in pkg.rglob("*.py"):
        tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        for node in ast.walk(tree):
            mods = set()
            if isinstance(node, ast.Import):
                mods = {a.name for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods = {node.module}
            for mod in mods:
                if mod == "navis.intelligence" or mod.startswith("navis.intelligence."):
                    violations.append(f"{py.name}:{node.lineno} imports {mod}")
    assert not violations, (
        "Decision must receive Intelligence as primitives from orchestration:\n"
        + "\n".join(violations)
    )


# 3. No ORM object, session or repository crosses the boundary.

def test_the_view_carries_only_primitives():
    from navis.orchestration.brief import _intelligence_view

    view = _intelligence_view(ORG_A, [_insight_row()])
    assert isinstance(view, IntelligenceView) and view.org_id == ORG_A
    assert isinstance(view.facts, tuple) and len(view.facts) == 1

    fact = view.facts[0]
    assert isinstance(fact, IntelligenceFact)
    assert isinstance(fact.insight_id, str)  # str, not uuid.UUID and not a row
    assert isinstance(fact.supporting_reasoning_signal_ids, tuple)
    for value in (
        fact.insight_id, fact.insight_type, fact.key, fact.summary,
        fact.confidence, fact.sample_size, fact.evidence,
        fact.supporting_reasoning_signal_ids, fact.state_watermark,
        fact.learning_version,
    ):
        assert isinstance(value, (str, int, float, dict, tuple, type(None))), value
    # Frozen: an input to a deterministic ranker cannot be edited in place.
    with pytest.raises(Exception):
        fact.confidence = 0.1  # type: ignore[misc]


def test_the_view_copies_evidence_and_cannot_dirty_a_stored_insight():
    """The ORM hands out the live JSONB dict; a shared reference would let a
    consumer mutate a persisted insight by touching what looks like a plain dict."""
    from navis.orchestration.brief import _intelligence_view

    row = _insight_row()
    fact = _intelligence_view(ORG_A, [row]).facts[0]
    fact.evidence["injected"] = True
    assert "injected" not in row.evidence


# 4. The facts are actually available, with their lineage intact.

def test_the_view_hoists_the_signal_lineage_decision_already_ranks():
    from navis.orchestration.brief import _intelligence_view

    fact = _intelligence_view(ORG_A, [_insight_row()]).facts[0]
    assert fact.supporting_reasoning_signal_ids == ("sig-a", "sig-b")
    assert fact.key == "relationship_cooling" and fact.sample_size == 2
    assert fact.state_watermark == "wm-1" and fact.learning_version == 7


def test_the_view_invents_no_lineage_when_an_insight_folded_no_signals():
    from navis.orchestration.brief import _intelligence_view

    fact = _intelligence_view(ORG_A, [_insight_row(evidence={"affected": 1})]).facts[0]
    assert fact.supporting_reasoning_signal_ids == ()


def test_the_view_is_deterministic_and_uses_logical_time():
    from navis.orchestration.brief import _intelligence_view

    rows = [_insight_row(), _insight_row(insight_id=_uuid.uuid4(), key="commitment_slipping")]
    a, b = _intelligence_view(ORG_A, rows), _intelligence_view(ORG_A, rows)
    assert a == b
    assert a.as_of == max(r.created_at for r in rows)  # never wall clock
    assert _intelligence_view(ORG_A, []).as_of is None


# 5. Organization isolation.

async def test_decide_refuses_another_organizations_intelligence():
    """Fail loud rather than rank one org's signals against another's
    understanding of itself. Nothing would collide — every id hashes org_id — so
    the corruption would be silent cross-tenant attribution."""
    from navis.decisions.service import DecisionService

    # db=None proves the guard runs BEFORE any database access.
    svc = DecisionService(None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="intelligence view is for org"):
        await svc.decide(
            ORG_A, [],
            reasoning_version=4, state_watermark="wm-1", as_of=None,
            intelligence=IntelligenceView(org_id=ORG_B),
        )


# 6 + 7. Existing behaviour and existing ids are provably untouched.

def test_intelligence_never_changes_a_decision_id():
    """THE id-stability proof. Phase 4.2 pinned this structurally (the view could
    not reach the ranker); Phase 4.3 lets it in, so the guarantee is now proved
    empirically: identity is (org, reasoning_version, watermark, signal), and
    corroboration is not part of it."""
    import inspect

    from navis.decisions.service import DecisionService

    sigs = [
        signal(SignalType.COMMITMENT_SLIPPING, entity_type="commitment",
               entity_id="c-1", tag="a"),
        signal(SignalType.WAITING_ON_USER_REPLY, entity_type="thread",
               entity_id="t-1", tag="b"),
    ]
    without = decide(sigs)
    with_view = WhatMattersTodayEngine().decide(
        sigs, org_id=ORG, reasoning_version=2, state_watermark=WATERMARK,
        as_of=AS_OF, intelligence=_view(_fact(signal_ids=(str(sigs[0].signal_id),))),
    )
    assert {str(d.decision_id) for d in without} == {str(d.decision_id) for d in with_view}

    src = inspect.getsource(DecisionService.decide)
    assert "compute_brief_id(org_id, reasoning_version, state_watermark)" in src, (
        "brief identity changed; every stored brief id would fork"
    )


def test_decide_still_accepts_no_intelligence():
    """Backward compatibility: the subscriber path and every existing caller."""
    import inspect

    from navis.decisions.service import DecisionService

    param = inspect.signature(DecisionService.decide).parameters["intelligence"]
    assert param.default is None


# 1. Orchestration is the composition root, in the right order.

def test_build_brief_reads_intelligence_through_its_repository_and_hands_it_over():
    import ast
    import pathlib

    src = (pathlib.Path(__file__).resolve().parents[2]
           / "navis" / "orchestration" / "brief.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "build_brief"
    )
    calls = [
        n.func.id if isinstance(n.func, ast.Name) else getattr(n.func, "attr", "")
        for n in ast.walk(fn) if isinstance(n, ast.Call)
    ]
    assert "_current_intelligence" in calls, "Decision is no longer given Intelligence"
    # Projected first, THEN read: Decision must rank against the insights derived
    # from the snapshot it is about to rank, not the previous one.
    assert calls.index("_project_intelligence") < calls.index("_current_intelligence"), (
        "Decision would receive the PREVIOUS snapshot's insights"
    )
    assert calls.index("_current_intelligence") < calls.index("decide")

    reader = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "_current_intelligence"
    )
    reader_calls = [
        n.func.id if isinstance(n.func, ast.Name) else getattr(n.func, "attr", "")
        for n in ast.walk(reader) if isinstance(n, ast.Call)
    ]
    assert "get_insights" in reader_calls and "IntelligenceRepository" in reader_calls, (
        "orchestration must read Intelligence through its own boundary, not with "
        "a query of its own — that boundary owns which projection is current"
    )


# ── Phase 4.3 — Intelligence corroborates ranking ────────────────────────────
#
# Intelligence now affects PRIORITY, bounded so it can corroborate what
# Reasoning found and never outweigh it. These tests pin both directions: it
# must move the right decision, and it must be unable to move anything else.

def _fact(*, insight_id="i-1", insight_type="risk_intelligence", key="commitment_slipping",
          confidence=0.9, sample_size=1, signal_ids=(), entities=(), evidence=None):
    ev = dict(evidence or {})
    if entities:
        ev["entities"] = [dict(e) for e in entities]
    return IntelligenceFact(
        insight_id=insight_id, insight_type=insight_type, key=key,
        summary="s", confidence=confidence, sample_size=sample_size,
        evidence=ev, supporting_reasoning_signal_ids=tuple(signal_ids),
        state_watermark=WATERMARK, learning_version=3,
    )


def _view(*facts, org=ORG):
    return IntelligenceView(org_id=org, as_of=AS_OF, facts=tuple(facts))


def _decide_with(signals, view):
    return WhatMattersTodayEngine().decide(
        signals, org_id=ORG, reasoning_version=2, state_watermark=WATERMARK,
        as_of=AS_OF, intelligence=view,
    )


def _two_close_signals():
    """Two decisions whose weights differ by less than one bias unit."""
    return [
        signal(SignalType.COMMITMENT_SLIPPING, title="lead", entity_type="commitment",
               entity_id="c-1", confidence=0.9, severity=0.80, tag="lead"),
        signal(SignalType.COMMITMENT_SLIPPING, title="trail", entity_type="commitment",
               entity_id="c-2", confidence=0.9, severity=0.75, tag="trail"),
    ]


# A. empty view == current behaviour

def test_empty_intelligence_preserves_decisions_exactly():
    sigs = _two_close_signals()
    base = decide(sigs)
    for view in (None, _view()):
        got = _decide_with(sigs, view)
        assert [ (str(d.decision_id), d.priority_rank, d.category, d.title) for d in got ] == \
               [ (str(d.decision_id), d.priority_rank, d.category, d.title) for d in base ]
        assert all(d.supporting_insight_ids == () for d in got)
        # ...and the serialized contract still carries the key, always.
        assert all(d.to_dict()["supporting_insight_ids"] == [] for d in got)


# B. a relevant, high-confidence fact materially affects the right decision

def test_a_relevant_fact_promotes_the_decision_it_is_about():
    sigs = _two_close_signals()
    lead, trail = sigs[0], sigs[1]
    assert [d.title for d in decide(sigs)] == ["lead", "trail"]

    got = _decide_with(sigs, _view(_fact(signal_ids=(str(trail.signal_id),))))
    assert [d.title for d in got] == ["trail", "lead"], "Intelligence did not move the rank"
    promoted = got[0]
    assert promoted.priority_rank == 1
    assert promoted.supporting_insight_ids == ("i-1",)
    assert got[1].supporting_insight_ids == ()   # the other decision is untouched


def test_relevance_also_resolves_through_the_entity_the_insight_names():
    """The second deterministic relationship: the insight names this signal's
    entity in its own evidence, exactly as the reality engines write it."""
    sigs = _two_close_signals()
    fact = _fact(insight_id="i-ent", signal_ids=(), entities=(
        {"entity_type": "commitment", "entity_id": "c-2", "label": "x", "severity": 0.6},
    ))
    got = _decide_with(sigs, _view(fact))
    assert [d.title for d in got] == ["trail", "lead"]
    assert got[0].supporting_insight_ids == ("i-ent",)


# C. an irrelevant fact changes nothing

def test_an_irrelevant_fact_changes_no_decision():
    sigs = _two_close_signals()
    base = decide(sigs)
    unrelated = _view(
        _fact(insight_id="i-x", insight_type="organizational_status", key="footprint",
              confidence=1.0, sample_size=110),
        _fact(insight_id="i-y", key="other", signal_ids=("00000000-0000-0000-0000-000000000000",)),
        _fact(insight_id="i-z", entities=({"entity_type": "person", "entity_id": "p-9"},)),
    )
    got = _decide_with(sigs, unrelated)
    assert [d.title for d in got] == [d.title for d in base]
    assert [d.priority_rank for d in got] == [d.priority_rank for d in base]
    assert all(d.supporting_insight_ids == () for d in got)


# D. weak intelligence cannot overpower stronger direct evidence

def test_low_confidence_intelligence_cannot_overpower_stronger_evidence():
    sigs = _two_close_signals()
    trail = sigs[1]
    weak = _view(_fact(confidence=0.02, sample_size=1, signal_ids=(str(trail.signal_id),)))
    assert [d.title for d in _decide_with(sigs, weak)] == ["lead", "trail"]
    # ...and it is still recorded as having contributed, honestly.
    assert _decide_with(sigs, weak)[1].supporting_insight_ids == ("i-1",)


def test_intelligence_cannot_displace_a_decision_that_leads_by_more_than_the_ceiling():
    """The hard bound: one bias unit, whatever the insight claims."""
    from navis.decisions.what_matters_today import _INTELLIGENCE_BIAS_CEILING

    far_ahead = signal(SignalType.COMMITMENT_SLIPPING, title="lead", entity_type="commitment",
                       entity_id="c-1", confidence=1.0, severity=0.95, tag="lead")
    behind = signal(SignalType.COMMITMENT_SLIPPING, title="trail", entity_type="commitment",
                    entity_id="c-2", confidence=1.0, severity=0.95 - _INTELLIGENCE_BIAS_CEILING - 0.01,
                    tag="trail")
    maximal = _view(_fact(confidence=1.0, sample_size=10_000,
                          signal_ids=(str(behind.signal_id),)))
    assert [d.title for d in _decide_with([far_ahead, behind], maximal)] == ["lead", "trail"]


def test_sample_size_is_not_treated_as_evidential_weight():
    """A 110-entity footprint recitation must not outrank a real obligation."""
    sigs = _two_close_signals()
    huge = _view(_fact(insight_type="organizational_status", key="footprint",
                       confidence=0.10, sample_size=110,
                       signal_ids=(str(sigs[1].signal_id),)))
    assert [d.title for d in _decide_with(sigs, huge)] == ["lead", "trail"]


# Intelligence never manufactures a decision

def test_intelligence_cannot_create_a_decision_without_actionable_evidence():
    interpretation = [signal(SignalType.CONTEXT_SUMMARY, tag="ctx"),
                      signal(SignalType.OPPORTUNITY, tag="opp")]
    view = _view(_fact(signal_ids=tuple(str(s.signal_id) for s in interpretation)))
    assert _decide_with(interpretation, view) == []
    assert _decide_with([], view) == []


def test_intelligence_cannot_revive_a_signal_below_the_confidence_floor():
    weak = signal(SignalType.COMMITMENT_SLIPPING, entity_type="commitment",
                  entity_id="c-1", confidence=0.10, tag="weak")
    view = _view(_fact(confidence=1.0, signal_ids=(str(weak.signal_id),)))
    assert _decide_with([weak], view) == []


# F. provenance records exactly what was used

def test_supporting_insight_ids_are_exactly_the_facts_that_scored():
    sigs = _two_close_signals()
    trail = sigs[1]
    view = _view(
        _fact(insight_id="i-a", signal_ids=(str(trail.signal_id),)),
        _fact(insight_id="i-b", entities=({"entity_type": "commitment", "entity_id": "c-2"},)),
        _fact(insight_id="i-unrelated", entities=({"entity_type": "person", "entity_id": "p-1"},)),
    )
    got = {d.title: d for d in _decide_with(sigs, view)}
    assert got["trail"].supporting_insight_ids == ("i-a", "i-b")  # sorted, deduped by id
    assert got["lead"].supporting_insight_ids == ()
    assert got["trail"].to_dict()["supporting_insight_ids"] == ["i-a", "i-b"]


def test_provenance_never_touches_the_signals_evidence():
    """`Decision.evidence` IS the ReasoningSignal's own dict, already persisted."""
    sigs = _two_close_signals()
    before = dict(sigs[1].evidence)
    got = _decide_with(sigs, _view(_fact(signal_ids=(str(sigs[1].signal_id),))))
    assert sigs[1].evidence == before
    assert all("supporting_insight_ids" not in d.evidence for d in got)


# J. determinism

def test_same_inputs_produce_identical_decisions():
    sigs = _two_close_signals()
    view = _view(_fact(signal_ids=(str(sigs[1].signal_id),)),
                 _fact(insight_id="i-2", entities=({"entity_type": "commitment", "entity_id": "c-1"},)))
    a = [d.to_dict() for d in _decide_with(sigs, view)]
    b = [d.to_dict() for d in _decide_with(sigs, view)]
    assert a == b


# H. the execution proposer stays compatible

def test_the_execution_proposer_still_reads_these_decisions():
    from navis.execution.proposer import propose_from_decisions

    waiting = signal(SignalType.WAITING_ON_USER_REPLY, entity_type="thread",
                     entity_id="t-1", tag="wait")
    got = _decide_with([waiting], _view(_fact(signal_ids=(str(waiting.signal_id),))))
    payload = [d.to_dict() for d in got]
    assert payload[0]["supporting_insight_ids"] == ["i-1"]
    # The proposer reads named keys only; the extra key must not disturb it.
    propose_from_decisions(payload, ORG)


# The engine stays pure: no I/O, no per-decision query, one materialized view.

def test_the_ranker_performs_no_io_and_imports_no_infrastructure():
    import ast
    import pathlib

    pkg = pathlib.Path(__file__).resolve().parents[2] / "navis" / "decisions"
    forbidden_roots = {"navis.intelligence", "navis.memory", "navis.signals",
                       "navis.state", "navis.ml", "navis.context", "navis.execution"}
    violations = []
    for py in pkg.rglob("*.py"):
        src = py.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(py))
        for node in ast.walk(tree):
            mods = set()
            if isinstance(node, ast.Import):
                mods = {a.name for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods = {node.module}
            for mod in mods:
                root = ".".join(mod.split(".")[:2])
                if root in forbidden_roots:
                    violations.append(f"{py.name}:{node.lineno} imports {mod}")
    assert not violations, "Decision must stay free of infrastructure:\n" + "\n".join(violations)

    engine = (pkg / "what_matters_today.py").read_text(encoding="utf-8")
    for banned in ("sqlalchemy", "session", "await ", "async def", "select(", "requests"):
        assert banned not in engine, f"the ranker gained `{banned}` — it must stay pure"


# ── Hardening: an org-wide aggregate can never out-bias the specific insight ──
#
# `organizational_status/footprint` cites EVERY risk signal in the org (see
# `insights.status_intelligence`), so it corroborates every RISK decision — a
# category-level constant arriving through the fact channel, on top of the RISK
# category bias. That would be double-counting if it could ever set the bias.
#
# It cannot, and the reason is structural rather than lucky:
#
#   * `_composite_confidence` is the MINIMUM over contributing signals;
#   * footprint's contributing set (all risk signals + priority_load) is a
#     SUPERSET of `risk_intelligence/<type>`'s (signals of that type only);
#   * min over a superset <= min over a subset;
#   * `risk_intelligence/<type>` always exists for a present risk signal (its
#     generator has no threshold) and always cites it, so it is always relevant;
#   * `_intelligence_bias` takes the MAX confidence among relevant facts.
#
# => the type-specific insight always ties or beats the footprint. This test
# pins the chain so a change to either side cannot quietly reverse it.

def test_the_org_wide_footprint_can_never_set_the_bias_for_a_risk_decision():
    from navis.intelligence.service import project_values
    from navis.orchestration.brief import _intelligence_view
    from tests.intelligence.fakes import make_view as make_learning_view
    from tests.intelligence.fakes import rich_reality

    org = uuid.uuid4()
    rows = project_values(
        make_learning_view(org_id=org, learning_version=0),
        rich_reality(org_id=org, state_watermark="wm-1", reasoning_version=4),
    )
    facts = {f.key: f for f in _intelligence_view(org, [SimpleNamespace(**r) for r in rows]).facts}
    footprint = facts["footprint"]
    risk_facts = [f for f in facts.values() if f.insight_type == "risk_intelligence"]
    assert risk_facts, "fixture must contain at least one risk insight"

    for risk in risk_facts:
        assert footprint.confidence <= risk.confidence, (
            f"footprint ({footprint.confidence}) outranks the specific "
            f"{risk.key} insight ({risk.confidence}) — an org-wide counts "
            "recitation would set the bias for a real obligation"
        )
        # ...and both are relevant to the same signal, so max() sees both.
        assert set(risk.supporting_reasoning_signal_ids) <= set(
            footprint.supporting_reasoning_signal_ids
        ), "footprint no longer folds every risk signal; the superset argument breaks"


def test_footprint_only_ever_corroborates_risk_category_signals():
    """It cites risk signals + priority_load and nothing else, so it cannot
    reach a WAITING_REPLY / FOLLOW_UP / DECISION_OWED decision at all.

    The two sets are maintained in different layers and must stay identical: if
    Intelligence widened its risk vocabulary, footprint would start corroborating
    decisions for which no type-specific insight exists — and then it COULD set
    the bias, because the superset argument above would no longer have a
    same-signal competitor.
    """
    from navis.intelligence.insights import _RISK_SIGNAL_TYPES

    decision_risk_types = {
        st.value for st in SignalType if category_for(st) is DecisionCategory.RISK
    }
    assert _RISK_SIGNAL_TYPES == decision_risk_types, (
        "Intelligence's risk vocabulary drifted from Decision's RISK category; "
        "footprint can now reach a decision with no specific insight to bound it"
    )
