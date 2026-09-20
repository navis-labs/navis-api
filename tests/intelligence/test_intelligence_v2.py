"""Intelligence V2: determinism, replay parity, lineage, the six generators,
Learning-only dependency, and the no-clock/no-uuid4 determinism guards.
"""
from __future__ import annotations

import ast
import pathlib
from dataclasses import fields as dataclasses_fields

import pytest

from navis.intelligence import insights as gen
from navis.intelligence.insights import AggRow
from navis.intelligence.models import InsightType, compute_insight_id
from navis.intelligence.replay import IntelligenceReplayService
from navis.intelligence.service import IntelligenceService, project_values
from tests.intelligence.fakes import (
    AS_OF,
    ORG_A,
    ORG_B,
    FakeIntelligenceRepo,
    make_reality,
    make_view,
    reality_signal,
    rich_reality,
    rich_view,
    snapshot,
)

PKG = pathlib.Path(__file__).resolve().parent.parent.parent / "navis" / "intelligence"
PIPELINE_DOMAINS = {
    "signals", "memory", "state", "reasoning",
    "decisions", "execution", "observation", "learning",
    # Phase 4.0 gave Intelligence a second input (current organizational
    # reality). The tempting shortcut for that is to reach for State, Reasoning
    # or ML directly; `ml` is listed explicitly because it is the one an
    # "insight needs semantic context" change would grab first, and Intelligence
    # narrating through a model would make the LLM the source of truth for
    # insight_type / confidence / sample_size.
    "ml",
}


# ---- 1. deterministic insight ids --------------------------------------------

def test_insight_id_is_deterministic_and_versioned():
    a = compute_insight_id(ORG_A, "signal_intelligence", "sig-A", 12)
    assert a == compute_insight_id(ORG_A, "signal_intelligence", "sig-A", 12)
    # New learning version => new immutable id (append-only, never mutate).
    assert a != compute_insight_id(ORG_A, "signal_intelligence", "sig-A", 13)
    # Different subject => different id.
    assert a != compute_insight_id(ORG_A, "signal_intelligence", "sig-B", 12)


# ---- 2. same Learning in => same insights out --------------------------------

def test_same_view_same_projection():
    view = rich_view()
    assert snapshot(project_values(view)) == snapshot(project_values(view))


# ---- 3 + 14. delete-and-rebuild parity ---------------------------------------

async def test_delete_and_rebuild_is_identical():
    repo = FakeIntelligenceRepo(rich_view())
    await IntelligenceService(repo).project(ORG_A)
    before = snapshot([r.__dict__ for r in await repo.get_insights(ORG_A)])

    await IntelligenceReplayService(repo=repo).rebuild(ORG_A)  # drop + re-project
    after = snapshot([r.__dict__ for r in await repo.get_insights(ORG_A)])
    assert after == before and len(after) > 0


# ---- 16. incremental projection == idempotent (no duplicates) ----------------

async def test_project_is_idempotent_append_only():
    repo = FakeIntelligenceRepo(rich_view())
    first = await IntelligenceService(repo).project(ORG_A)
    n_after_first = len(repo.insights)
    second = await IntelligenceService(repo).project(ORG_A)  # redeliver
    assert snapshot(first) == snapshot(second)
    assert len(repo.insights) == n_after_first  # ON CONFLICT DO NOTHING


# ---- 4. multi-tenant isolation -----------------------------------------------

async def test_multi_tenant_ids_and_isolation():
    repo_a = FakeIntelligenceRepo(rich_view(org_id=ORG_A))
    repo_b = FakeIntelligenceRepo(rich_view(org_id=ORG_B))
    await IntelligenceService(repo_a).project(ORG_A)
    await IntelligenceService(repo_b).project(ORG_B)
    ids_a = {str(r.insight_id) for r in await repo_a.get_insights(ORG_A)}
    ids_b = {str(r.insight_id) for r in await repo_b.get_insights(ORG_B)}
    assert ids_a and ids_b and ids_a.isdisjoint(ids_b)  # org_id in the id namespace


async def test_empty_learning_yields_no_insights():
    # version-0 (nothing learned yet) short circuits at the service layer.
    repo = FakeIntelligenceRepo(make_view(learning_version=0))
    rows = await IntelligenceService(repo).project(ORG_A)
    assert rows == [] and repo.insights == {}


# ---- 5. Learning is the ONLY upstream dependency -----------------------------

def test_no_forbidden_pipeline_imports():
    violations = []
    for py in PKG.rglob("*.py"):
        tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        mods = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods |= {a.name for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods.add(node.module)
        for mod in mods:
            parts = mod.split(".")
            if len(parts) >= 2 and parts[0] == "navis" and parts[1] in PIPELINE_DOMAINS:
                if parts[1] != "learning":
                    violations.append(f"{py.name} imports {mod}")
    assert not violations, "Intelligence may only depend on Learning:\n" + "\n".join(violations)


# ---- 6 + 7. determinism guards: no wall clock, no uuid4/random ---------------

def test_no_nondeterministic_calls():
    banned = {"now", "utcnow", "uuid4", "uuid1", "random", "randint", "choice", "shuffle"}
    offenders = []
    for py in PKG.rglob("*.py"):
        tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in banned:
                    offenders.append(f"{py.name}:{node.lineno} {node.func.attr}()")
    assert not offenders, "non-deterministic calls in Intelligence:\n" + "\n".join(offenders)


def test_created_at_is_logical_time_not_wall_clock():
    rows = project_values(rich_view())
    assert rows and all(r["created_at"] == AS_OF for r in rows)


# ---- 8..13. the six generators -----------------------------------------------

def test_signal_intelligence_states_fact_and_skips_thin():
    drafts = gen.signal_intelligence(rich_view())
    keys = {d.key for d in drafts}
    assert keys == {"sig-A", "sig-B"}  # sig-thin (2 samples) excluded
    a = next(d for d in drafts if d.key == "sig-A")
    assert a.insight_type is InsightType.SIGNAL
    assert a.evidence["success_rate"] == 0.8 and a.sample_size == 10
    assert a.supporting == ["e1", "e2", "e3"]  # lineage carried


def test_decision_intelligence():
    drafts = gen.decision_intelligence(rich_view())
    assert len(drafts) == 1
    d = drafts[0]
    assert d.insight_type is InsightType.DECISION and d.key == "dec-1"
    assert d.evidence["success_rate"] == 0.6 and d.sample_size == 10


def test_team_intelligence_from_patterns():
    drafts = gen.team_intelligence(rich_view())
    assert len(drafts) == 1
    d = drafts[0]
    assert d.insight_type is InsightType.TEAM
    assert d.evidence == {"reliable": 1, "total_patterns": 2}


def test_customer_intelligence_engagement_rate():
    drafts = gen.customer_intelligence(rich_view())
    assert len(drafts) == 1
    d = drafts[0]
    assert d.insight_type is InsightType.CUSTOMER
    # (reply 7 + meeting 3) / 20 executed = 0.5
    assert d.evidence == {"engaged": 10, "attempts": 20, "engagement_rate": 0.5}


def test_organizational_intelligence_outcome_profile():
    drafts = gen.organizational_intelligence(rich_view())
    assert len(drafts) == 1
    d = drafts[0]
    assert d.insight_type is InsightType.ORGANIZATIONAL
    # positive = 7+3 = 10; negative = action_failed 5 + no_response 8 = 13; 10/23
    assert d.evidence["positive"] == 10 and d.evidence["negative"] == 13


def test_adaptation_intelligence_picks_strongest_signal():
    drafts = gen.adaptation_intelligence(rich_view())
    assert len(drafts) == 1
    d = drafts[0]
    assert d.insight_type is InsightType.ADAPTATION
    assert d.evidence["signal_key"] == "sig-A"  # best decisive success rate
    # Stated as a learned fact, never a recommendation.
    low = (d.title + d.summary).lower()
    assert "should" not in low and "recommend" not in low


# ---- 15. lineage completeness ------------------------------------------------

def test_lineage_is_capped_sorted_and_traceable():
    big = [f"e{i:03d}" for i in range(120)]
    view = make_view(
        signal_aggs=[AggRow("sig-X", successes=60, failures=10, neutrals=0, total=70)],
        lineage={"sig-X": big},
    )
    row = project_values(view)[0]
    ids = row["supporting_learning_event_ids"]
    assert ids == sorted(big)[:50]  # capped at 50, sorted, real learning ids
    assert all(i in big for i in ids)


# ---- no recommendations / scores anywhere ------------------------------------

def test_insight_vocabulary_is_closed_and_split_by_clock():
    """The whole vocabulary, and which clock each half is versioned against.

    Adding a type is a contract change, so it fails here first. The split is the
    load-bearing part: `service._version_key` reads `REALITY_INSIGHT_TYPES` to
    decide whether an insight's identity moves with `learning_version` or with
    `state_watermark`, and a type in neither half — or in both — would be
    versioned against the wrong clock and silently frozen or duplicated.
    """
    from navis.intelligence.insights import REALITY_INSIGHT_TYPES

    learning = {
        InsightType.SIGNAL, InsightType.DECISION, InsightType.TEAM,
        InsightType.CUSTOMER, InsightType.ORGANIZATIONAL, InsightType.ADAPTATION,
    }
    reality = {
        InsightType.RISK, InsightType.OPPORTUNITY, InsightType.RELATIONSHIP,
        InsightType.COMMITMENT, InsightType.OBLIGATION, InsightType.STATUS,
        InsightType.TREND,
        # Folded from the current snapshot's entity graph and the conclusions
        # Reasoning reached over it — so it moves when State moves, exactly like
        # every other reality family, and is versioned by
        # `state_watermark|reasoning_version`. It is emphatically NOT Learning-
        # derived: it measures nothing about past outcomes.
        InsightType.SITUATION,
    }
    assert set(InsightType) == learning | reality
    assert learning.isdisjoint(reality)
    assert REALITY_INSIGHT_TYPES == reality


def test_no_recommendation_or_score_columns():
    from navis.intelligence.models import IntelligenceInsight

    cols = set(IntelligenceInsight.__table__.columns.keys())
    forbidden = {"recommendation", "action", "score", "risk", "priority", "advice"}
    assert cols & forbidden == set()


# ---- the trigger + the evidence chain (Phase 3F) -----------------------------
#
# Intelligence was fully implemented and never ran: `project()` was reachable
# only from POST /intelligence/project, so `intelligence_insights` held 0 rows
# and Adaptive — which reads those insights — had nothing to propose from. The
# loop was wired end to end and never turned over. Nothing in the code looked
# broken, which is exactly why it needs a test.


def test_learning_updated_triggers_the_projection():
    """Drop the registry import and the loop silently stops closing again."""
    from navis.platform.composition import ensure_subscribers
    from navis.platform.events import EventType, event_bus

    ensure_subscribers()
    handlers = [
        getattr(h, "__name__", "") for h in event_bus._handlers.get(EventType.LEARNING_UPDATED, [])
    ]
    assert "on_learning_updated" in handlers, (
        "nothing re-projects Intelligence when Learning changes; insights will "
        "stay empty and Adaptive will never receive them"
    )


def _signal_row(rows: list[dict]) -> dict:
    """The signal_intelligence row — a view also drives the adaptation engine."""
    return next(r for r in rows if r["insight_type"] == "signal_intelligence")


def test_insight_evidence_carries_the_state_watermark():
    """An insight traceable to its learning events but not to the State snapshot
    they described cannot answer "was this true of the org, and when?"."""
    view = make_view(
        signal_aggs=[AggRow("sig-A", successes=6, failures=1, neutrals=0, total=7)],
        state_watermark="wm-42",
    )
    row = _signal_row(project_values(view))
    assert row["evidence"]["state_watermark"] == "wm-42"
    # Provenance may never overwrite a measurement.
    assert row["evidence"]["successes"] == 6


def test_evidence_omits_provenance_it_does_not_have():
    """Absence is honest; a null would read as "we looked and there was none"."""
    view = make_view(
        signal_aggs=[AggRow("sig-A", successes=6, failures=1, neutrals=0, total=7)],
    )
    row = _signal_row(project_values(view))
    assert "state_watermark" not in row["evidence"]
    assert "supporting_reasoning_signal_ids" not in row["evidence"]


def test_insight_traces_back_to_the_reasoning_that_surfaced_it():
    view = make_view(
        decision_aggs=[AggRow("dec-1", successes=6, failures=2, neutrals=0, total=8)],
        signal_lineage={"dec-1": ["sig-b", "sig-a"]},
    )
    row = next(r for r in project_values(view) if r["key"] == "dec-1")
    # Sorted, so the same evidence always serializes identically.
    assert row["evidence"]["supporting_reasoning_signal_ids"] == ["sig-a", "sig-b"]


# ---- Phase 4.0: the organizational-reality input contract --------------------
#
# Intelligence consumed Learning alone, so it described historical execution
# attempts and never "what is true of this organization now". State and Reasoning
# hold that answer and Intelligence may not import either, so it arrives as a
# primitive-only projection built by the composition root.
#
# This phase adds the CONTRACT, not an algorithm: no generator reads a reality
# view yet. The single behavioural change is which state_watermark an insight is
# stamped with. Everything below exists to prove the existing Learning-only
# projection is untouched by it.


def test_existing_learning_insight_ids_are_byte_identical():
    """The id formula, pinned literally.

    `compute_insight_id` was generalised from `learning_version: int` to
    `version_key: int | str` so reality insights can be versioned by watermark.
    If that generalisation had changed the key string by so much as a separator,
    every insight ever written would have been orphaned — the table is
    append-only with ON CONFLICT DO NOTHING, so the old rows would simply never
    be matched again and would sit there forever alongside their duplicates.
    Recomputing the uuid5 here rather than calling the function is the point.
    """
    import uuid as _uuid

    from navis.intelligence.models import _INSIGHT_NAMESPACE

    expected = _uuid.uuid5(
        _INSIGHT_NAMESPACE, f"{ORG_A}|signal_intelligence|sig-A|12"
    )
    assert compute_insight_id(ORG_A, "signal_intelligence", "sig-A", 12) == expected
    # An int and its string render identically — that equivalence is what makes
    # the pre-existing call sites safe.
    assert compute_insight_id(ORG_A, "signal_intelligence", "sig-A", "12") == expected


def test_version_key_separates_learning_from_reality_identity():
    """A watermark and a learning version are different version dimensions."""
    by_learning = compute_insight_id(ORG_A, "signal_intelligence", "k", 7)
    by_watermark = compute_insight_id(ORG_A, "signal_intelligence", "k", "wm-7")
    assert by_learning != by_watermark
    # Reality moves when State moves: a new watermark is a new immutable row.
    assert by_watermark != compute_insight_id(ORG_A, "signal_intelligence", "k", "wm-8")


def test_reality_is_optional_and_changes_nothing_by_default():
    """Both existing callers (the subscriber, POST /project) pass no reality."""
    view = rich_view()
    assert snapshot(project_values(view)) == snapshot(project_values(view, None))


def test_reality_supplies_the_authoritative_watermark():
    """State's watermark beats the one inherited from a LearningEvent.

    `LearningView.state_watermark` is whatever an execution happened to record on
    its way past — in production two of three outcomes recorded NULL, so the
    newest-non-null pick can stamp an insight with an unrelated action's
    watermark. When State is available, it is the answer.
    """
    view = make_view(
        signal_aggs=[AggRow("sig-A", successes=6, failures=1, neutrals=0, total=7)],
        state_watermark="wm-from-learning",
    )
    row = _signal_row(project_values(view, make_reality(state_watermark="wm-from-state")))
    assert row["state_watermark"] == "wm-from-state"
    # One notion of "which snapshot is this about" — column and evidence agree.
    assert row["evidence"]["state_watermark"] == "wm-from-state"
    # Reality never touches a measurement.
    assert row["evidence"]["successes"] == 6
    # ...and never the identity: these rows are still Learning-derived folds.
    assert row["insight_id"] == compute_insight_id(
        ORG_A, "signal_intelligence", "sig-A", str(view.learning_version)
    )


def test_reality_without_a_watermark_fabricates_nothing():
    """An org with no State snapshot yet must not invent one."""
    view = make_view(
        signal_aggs=[AggRow("sig-A", successes=6, failures=1, neutrals=0, total=7)],
    )
    row = _signal_row(project_values(view, make_reality(state_watermark=None)))
    assert row["state_watermark"] is None
    assert "state_watermark" not in row["evidence"]  # absence, not null


def test_engine_identity_is_stamped_on_every_row():
    """"Would this generator still emit this today?" needs an answer."""
    from navis.intelligence.insights import (
        LEARNING_ENGINE_NAME,
        LEARNING_ENGINE_VERSION,
    )

    rows = project_values(rich_view())
    assert rows
    assert all(r["engine_name"] == LEARNING_ENGINE_NAME for r in rows)
    assert all(r["engine_version"] == LEARNING_ENGINE_VERSION for r in rows)


def test_same_learning_and_reality_produce_an_identical_projection():
    view, reality = rich_view(), make_reality()
    first, second = project_values(view, reality), project_values(view, reality)
    assert snapshot(first) == snapshot(second)
    assert [r["state_watermark"] for r in first] == [r["state_watermark"] for r in second]


def test_reality_view_is_frozen_and_deterministic():
    """The contract is a value, not a handle: no ORM row, no session, no clock."""
    import dataclasses

    a, b = make_reality(), make_reality()
    assert a == b  # same inputs => same value
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.state_watermark = "mutated"  # type: ignore[misc]
    assert isinstance(a.signals, tuple)  # a frozen dataclass around a list is not


async def test_reality_from_another_org_is_refused():
    """Ids hash org_id, so a mismatch would not collide — it would silently
    attribute Org A's state to Org B's insight. Louder is safer."""
    repo = FakeIntelligenceRepo(rich_view(org_id=ORG_A))
    with pytest.raises(ValueError, match="reality view is for org"):
        await IntelligenceService(repo).project(ORG_A, reality=make_reality(org_id=ORG_B))
    assert repo.insights == {}  # nothing written on the refusal path


def test_reality_carries_no_orm_or_domain_objects():
    """Primitives only — the reason Intelligence can accept it without importing
    State or Reasoning."""
    reality = make_reality()
    allowed = (str, int, float, bool, dict, type(None))
    for row in reality.signals:
        for f in dataclasses_fields(row):
            assert isinstance(getattr(row, f.name), allowed), f.name


# ══════════════════════════════════════════════════════════════════════════════
# Phase 4.1 — the reality engines
# ══════════════════════════════════════════════════════════════════════════════


def _reality_rows(reality=None, view=None) -> list[dict]:
    rows = project_values(view or make_view(learning_version=0), reality)
    return [r for r in rows if r["insight_type"] in {t.value for t in REALITY_TYPES}]


REALITY_TYPES = {
    InsightType.RISK, InsightType.OPPORTUNITY, InsightType.RELATIONSHIP,
    InsightType.COMMITMENT, InsightType.OBLIGATION, InsightType.STATUS,
    InsightType.TREND, InsightType.SITUATION,
}


def _of_type(rows: list[dict], t: InsightType) -> list[dict]:
    return [r for r in rows if r["insight_type"] == t.value]


# ---- signal-type literals must match the real Reasoning vocabulary -----------

def test_reality_signal_type_literals_are_real_signal_types():
    """Intelligence cannot import Reasoning, so its signal types are literals.

    The test CAN import both, which is what stops the two sides drifting into
    an engine that silently matches nothing and emits no insight forever.
    """
    from navis.intelligence import insights as ins
    from navis.reasoning.v2.contract import SignalType

    known = {t.value for t in SignalType}
    referenced = (
        ins._RISK_SIGNAL_TYPES
        | ins._OPPORTUNITY_SIGNAL_TYPES
        | ins._COMMITMENT_SIGNAL_TYPES
        | {ins._RELATIONSHIP_COOLING, ins._OPPORTUNITY, ins._ORG_DRIFT, ins._PRIORITY_LOAD}
    )
    assert referenced <= known, f"not real SignalTypes: {sorted(referenced - known)}"


# WHICH DECISION-WORTHY CONCLUSIONS INTELLIGENCE CAN SEE — and which it cannot.
#
# Measured, not assumed: of the nine SignalTypes that can become a Decision, the
# reality generators consume seven. Two are consumed by NO generator, so a brief
# card built from either travels
#
#     Signal -> Memory -> State -> Reasoning -> Decision -> Today
#
# with Intelligence contributing nothing — not an insight, not corroboration, not
# an entry in `supporting_insights`. They are also, between them, two of the three
# categories `execution.proposer` can turn into a real action, so the MOST
# actionable cards on Today are exactly the ones the Intelligence layer is silent
# about.
#
# This is a COVERAGE GAP, not an incorrect claim. `_RISK_SIGNAL_TYPES` is
# byte-identical to the set `decisions.base` maps to `DecisionCategory.RISK`
# (asserted below), so `status_intelligence`'s "N carrying open risk" counts
# exactly what Decision also calls risk — the two layers agree, and neither
# overstates.
#
# THE GAP IS NOW CLOSED. `obligation_intelligence` gave `waiting_on_user_reply`
# and `decision_owed` their own family rather than folding them into RISK, so
# every decision-worthy conclusion now has an Intelligence representation and the
# risk vocabularies still match. This stays pinned at EMPTY: a new decision-worthy
# SignalType added without a generator re-opens the blind spot, and this is where
# that shows up.
_UNCOVERED_BY_INTELLIGENCE: set[str] = set()


def test_every_decision_worthy_conclusion_reaches_intelligence():
    from navis.decisions.base import _CATEGORY_FOR_SIGNAL
    from navis.intelligence import insights as ins

    consumed = (
        ins._RISK_SIGNAL_TYPES
        | ins._OPPORTUNITY_SIGNAL_TYPES
        | ins._COMMITMENT_SIGNAL_TYPES
        | ins._OBLIGATION_SIGNAL_TYPES
        | {ins._RELATIONSHIP_COOLING, ins._OPPORTUNITY, ins._ORG_DRIFT, ins._PRIORITY_LOAD}
    )
    decision_worthy = {t.value for t in _CATEGORY_FOR_SIGNAL}

    assert decision_worthy - consumed == _UNCOVERED_BY_INTELLIGENCE, (
        "a decision-worthy conclusion can reach Today with no Intelligence "
        "behind it; give it a generator or update _UNCOVERED_BY_INTELLIGENCE "
        "deliberately"
    )


def test_every_obligation_type_has_a_name_the_organization_would_use():
    """A signal type in the set but not the noun map would render as a raw
    engine identifier in a founder-facing summary."""
    from navis.intelligence import insights as ins

    assert set(ins._OBLIGATION_SIGNAL_TYPES) == set(ins._OBLIGATION_NOUNS)


def test_intelligence_and_decision_mean_the_same_thing_by_risk():
    """Two layers, one vocabulary. If these drift, `status_intelligence` starts
    counting a different population than the brief calls risk, and the same
    organization gets two different risk counts on one screen."""
    from navis.decisions.base import _CATEGORY_FOR_SIGNAL, DecisionCategory
    from navis.intelligence import insights as ins

    decision_risk = {
        t.value for t, c in _CATEGORY_FOR_SIGNAL.items() if c is DecisionCategory.RISK
    }
    assert decision_risk == set(ins._RISK_SIGNAL_TYPES)


# ---- 1. positive case: every engine that can fire, fires ---------------------

def test_reality_engines_produce_insights_from_real_signal_shapes():
    rows = _reality_rows(rich_reality())
    produced = {r["insight_type"] for r in rows}
    assert produced == {
        InsightType.RISK.value, InsightType.OPPORTUNITY.value,
        InsightType.RELATIONSHIP.value, InsightType.COMMITMENT.value,
        InsightType.STATUS.value,
    }
    # TREND stays silent: no org_drift signal, because nothing passes a prior
    # snapshot to Reasoning in production yet.
    assert _of_type(rows, InsightType.TREND) == []


def test_risk_is_keyed_by_kind_not_by_entity():
    """Bounded by the vocabulary (<=6), not by how many entities are affected."""
    rows = _of_type(_reality_rows(rich_reality()), InsightType.RISK)
    assert {r["key"] for r in rows} == {"relationship_cooling", "commitment_slipping"}
    cooling = next(r for r in rows if r["key"] == "relationship_cooling")
    assert cooling["sample_size"] == 2
    assert cooling["evidence"]["affected"] == 2
    # Peak severity, not an average: exposure is set by the worst instance.
    assert cooling["evidence"]["peak_severity"] == 0.6
    # Composite confidence is the MINIMUM — a claim about all of them can be no
    # stronger than its weakest member (0.75 and 0.89 here).
    assert cooling["confidence"] == 0.75
    labels = {e["label"] for e in cooling["evidence"]["entities"]}
    assert labels == {"amy@acme.com", "bo@beta.io"}  # read from evidence, not invented


def test_commitment_reports_only_deadlines_that_exist():
    row = _of_type(_reality_rows(rich_reality()), InsightType.COMMITMENT)[0]
    assert row["evidence"]["open_obligations"] == 2
    assert row["evidence"]["dated"] == 1 and row["evidence"]["undated"] == 1
    assert row["evidence"]["max_overdue_days"] == 11.5  # quoted from the message
    assert "deadline" not in row["summary"] or "stated deadline" in row["summary"]


def test_commitment_without_any_dated_promise_invents_no_deadline():
    reality = make_reality(signals=(
        reality_signal("s1", signal_type="commitment_slipping",
                       entity_type="commitment", entity_id="c1",
                       evidence={"age_days": 9.0}),
    ))
    row = _of_type(_reality_rows(reality), InsightType.COMMITMENT)[0]
    assert "max_overdue_days" not in row["evidence"]
    assert row["evidence"]["dated"] == 0
    assert "carries no stated deadline" in row["summary"]

    # ...and the plural path says the same thing without inventing a deadline.
    two = make_reality(signals=(
        reality_signal("s1", signal_type="commitment_slipping",
                       entity_type="commitment", entity_id="c1",
                       evidence={"age_days": 9.0}),
        reality_signal("s2", signal_type="commitment_slipping",
                       entity_type="commitment", entity_id="c2",
                       evidence={"age_days": 14.0}),
    ))
    plural = _of_type(_reality_rows(two), InsightType.COMMITMENT)[0]
    assert "none of them carries a stated deadline" in plural["summary"]
    assert "max_overdue_days" not in plural["evidence"]


def test_relationship_portfolio_counts_come_from_signals_not_recomputation():
    row = _of_type(_reality_rows(rich_reality()), InsightType.RELATIONSHIP)[0]
    # `tracked` is State's count; `cooling`/`warm` are counted from the signals
    # Reasoning emitted, never re-derived from timestamps here.
    assert row["evidence"]["tracked"] == 16
    assert row["evidence"]["cooling"] == 2
    assert row["evidence"]["warm"] == 1
    assert row["sample_size"] == 16
    assert {e["label"] for e in row["evidence"]["warm_entities"]} == {"dev@delta.co"}


def test_status_states_counts_and_never_a_health_score():
    row = _of_type(_reality_rows(rich_reality()), InsightType.STATUS)[0]
    assert row["evidence"]["counts"]["relationship"] == 16
    assert row["evidence"]["open_risks"] == 3
    assert row["evidence"]["pressure"] == {"cooling": 2, "slipping": 2, "stalled": 0}
    blob = (row["title"] + row["summary"] + str(row["evidence"])).lower()
    for banned in ("health", "score", "%", "grade", "rating"):
        assert banned not in blob, f"status insight leaked a {banned}"


def test_prose_is_grammatical_for_singular_and_plural():
    """These strings are rendered to founders. Live output read "12 companys",
    "1 company connect multiple known people" and "open past their threshold"
    for a single commitment."""
    from navis.intelligence.insights import _plural

    assert _plural(2, "company") == "companies"
    assert _plural(2, "person") == "people"
    assert _plural(1, "company") == "company"
    assert _plural(2, "relationship") == "relationships"

    rows = _reality_rows(rich_reality())
    blob = " ".join(r["title"] + " " + r["summary"] for r in rows)
    for broken in ("companys", "persons", "companies connects", "company connect ",
                   "1 commitments", "their threshold"):
        assert broken not in blob, f"ungrammatical output: {broken!r}"

    single_hub = make_reality(signals=(
        reality_signal("s-lev", signal_type="leverage_point", entity_type="company",
                       entity_id="co-1", evidence={"people": 3, "domain": "acme.com"}),
    ))
    row = _of_type(_reality_rows(single_hub), InsightType.OPPORTUNITY)[0]
    assert row["title"] == "1 hub company"
    assert "company connects multiple" in row["summary"]


def test_a_commitment_is_labelled_by_its_speaker_not_a_uuid():
    """`commitment_slipping` carries only `speaker_email`; without it in the
    label preference list every slipping commitment rendered as a raw UUID."""
    reality = make_reality(signals=(
        reality_signal("s-c", signal_type="commitment_slipping",
                       entity_type="commitment", entity_id="f33778d8-a54e-50bb",
                       evidence={"age_days": 9.0, "speaker_email": "amy@acme.com"}),
    ))
    row = _of_type(_reality_rows(reality), InsightType.COMMITMENT)[0]
    assert row["evidence"]["entities"][0]["label"] == "amy@acme.com"


def test_reality_insights_state_facts_never_recommendations():
    for row in _reality_rows(rich_reality()):
        low = (row["title"] + " " + row["summary"]).lower()
        for banned in ("should", "recommend", "you must", "we suggest", "consider "):
            assert banned not in low, f"{row['key']}: {banned!r}"


# ---- 2. insufficient evidence => no insight (the no-fabrication rule) --------

def test_nothing_tracked_and_nothing_concluded_means_no_insight_at_all():
    assert _reality_rows(make_reality(entity_counts={}, signals=())) == []


def test_entities_without_conclusions_yield_the_footprint_and_nothing_else():
    """"We are tracking 3 people" is a grounded fact about the organization and
    is the one thing State alone can support. Every other reality engine needs a
    reasoning conclusion and correctly stays silent."""
    rows = _reality_rows(make_reality(entity_counts={"person": 3}, signals=()))
    assert [r["insight_type"] for r in rows] == [InsightType.STATUS.value]
    row = rows[0]
    assert row["evidence"]["counts"] == {"person": 3}
    assert row["evidence"]["open_risks"] == 0
    assert "pressure" not in row["evidence"]  # no PriorityEngine signal to carry
    # Nothing was concluded, so nothing is cited — and no signal id is invented.
    assert "supporting_reasoning_signal_ids" not in row["evidence"]
    # The snapshot it describes is still recorded: that is State, not a conclusion.
    assert row["state_watermark"] == "wm-state-1"


def test_relationships_with_no_conclusions_produce_no_portfolio_insight():
    """Tracked relationships but nothing concluded about them is an absence of
    analysis, not a finding of "0 cooling"."""
    reality = make_reality(
        entity_counts={"relationship": 40},
        signals=(reality_signal("s1", signal_type="leverage_point",
                                entity_type="company", entity_id="co-1",
                                evidence={"people": 3, "domain": "x.com"}),),
    )
    assert _of_type(_reality_rows(reality), InsightType.RELATIONSHIP) == []


def test_zero_customers_produces_no_customer_risk():
    rows = _reality_rows(rich_reality())
    assert not [r for r in rows if r["key"] == "customer_engagement_drop"]


def test_a_single_snapshot_is_never_a_trend():
    """TREND requires an org_drift signal, which requires a PRIOR snapshot."""
    assert _of_type(_reality_rows(rich_reality()), InsightType.TREND) == []
    drift = make_reality(signals=(
        reality_signal("s-drift", signal_type="org_drift",
                       entity_type=None, entity_id=None, confidence=0.6, severity=0.3,
                       engine_name="trend", engine_version="2.0.0",
                       evidence={"delta": 4, "current": 12}),
    ))
    row = _of_type(_reality_rows(drift), InsightType.TREND)[0]
    assert row["evidence"]["delta"] == 4
    assert row["evidence"]["current"] == 12
    assert row["evidence"]["direction"] == "up"
    assert row["sample_size"] == 4
    assert row["evidence"]["supporting_reasoning_signal_ids"] == ["s-drift"]


def test_no_forecast_or_behaviour_insight_is_ever_produced():
    """Neither has an evidence source. Absence is the designed output."""
    values = {t.value for t in InsightType}
    assert not [v for v in values if "forecast" in v or "behavio" in v]


def test_reality_without_a_watermark_projects_nothing():
    """The watermark IS the version key; without one, rows could not be told
    apart from the next snapshot's."""
    assert _reality_rows(rich_reality(state_watermark=None)) == []


# ---- 3 + 4 + 5 + 6 + 7. determinism, isolation, provenance, versioning ------

def test_reality_projection_is_byte_identical_for_the_same_inputs():
    view, reality = rich_view(), rich_reality()
    assert snapshot(project_values(view, reality)) == snapshot(project_values(view, reality))
    a, b = project_values(view, reality), project_values(view, reality)
    assert [r["evidence"] for r in a] == [r["evidence"] for r in b]


def test_reality_insight_identity_moves_with_the_state_watermark():
    at_1 = _reality_rows(rich_reality(state_watermark="wm-1"))
    at_2 = _reality_rows(rich_reality(state_watermark="wm-2"))
    ids_1 = {str(r["insight_id"]) for r in at_1}
    ids_2 = {str(r["insight_id"]) for r in at_2}
    assert ids_1 and ids_1.isdisjoint(ids_2)  # a new snapshot is a new immutable row
    assert {r["key"] for r in at_1} == {r["key"] for r in at_2}  # same conclusions


def test_reality_insight_identity_also_moves_with_the_reasoning_version():
    """State and Reasoning are two independent clocks over the same facts.

    THE LIVE FAILURE this pins: the v4 relevance fix changed what Reasoning
    concludes about an UNCHANGED snapshot, so the projection recomputed the same
    `insight_id`s and ON CONFLICT DO NOTHING discarded every corrected row. The
    stale "4 open relationship cooling risks" — built entirely from newsletters —
    stayed served until an unrelated memory record moved the watermark.
    """
    v3 = _reality_rows(rich_reality(state_watermark="wm-1", reasoning_version=3))
    v4 = _reality_rows(rich_reality(state_watermark="wm-1", reasoning_version=4))
    ids_v3 = {str(r["insight_id"]) for r in v3}
    ids_v4 = {str(r["insight_id"]) for r in v4}
    assert ids_v3 and ids_v3.isdisjoint(ids_v4)
    # Same snapshot, so the watermark column still records the same State.
    assert {r["state_watermark"] for r in v3} == {"wm-1"}
    assert {r["state_watermark"] for r in v4} == {"wm-1"}


async def test_a_reasoning_version_bump_appends_corrected_rows():
    """The end-to-end shape of the live failure, at the service boundary."""
    repo = FakeIntelligenceRepo(rich_view())
    svc = IntelligenceService(repo)
    await svc.project(ORG_A, reality=rich_reality(state_watermark="wm-1", reasoning_version=3))
    before = dict(repo.insights)
    await svc.project(ORG_A, reality=rich_reality(state_watermark="wm-1", reasoning_version=4))
    after = repo.insights
    assert len(after) > len(before), "corrected conclusions were silently discarded"
    for k, v in before.items():  # history stays immutable
        assert after[k].title == v.title


def test_reality_insights_are_org_isolated():
    a = _reality_rows(rich_reality(org_id=ORG_A))
    b = _reality_rows(rich_reality(org_id=ORG_B), view=make_view(org_id=ORG_B, learning_version=0))
    assert {str(r["insight_id"]) for r in a}.isdisjoint({str(r["insight_id"]) for r in b})
    assert all(r["organization_id"] == ORG_A for r in a)


def test_every_reality_insight_traces_back_to_its_reasoning_signals():
    for row in _reality_rows(rich_reality()):
        cited = row["evidence"].get("supporting_reasoning_signal_ids")
        assert cited, f"{row['key']} cites no reasoning signal"
        assert cited == sorted(cited)
        assert all(c.startswith("sig-") for c in cited)  # real ids from the view
        assert row["state_watermark"] == "wm-state-1"
        # Reality folds are not derived from learning events, and say so.
        assert row["supporting_learning_event_ids"] == []


def test_each_reality_engine_stamps_its_own_identity():
    engines = {r["insight_type"]: (r["engine_name"], r["engine_version"])
               for r in _reality_rows(rich_reality())}
    assert engines[InsightType.RISK.value] == ("intelligence.reality.risk", "1")
    assert engines[InsightType.STATUS.value] == ("intelligence.reality.status", "1")
    assert all(name.startswith("intelligence.reality.") for name, _ in engines.values())
    # ...and never the Learning engine's identity.
    assert "intelligence.learning" not in {name for name, _ in engines.values()}


def test_learning_insights_keep_the_learning_engine_identity():
    rows = [r for r in project_values(rich_view(), rich_reality())
            if r["insight_type"] not in {t.value for t in REALITY_TYPES}]
    assert rows
    assert all(r["engine_name"] == "intelligence.learning" for r in rows)


def test_reality_created_at_is_states_logical_clock():
    for row in _reality_rows(rich_reality()):
        assert row["created_at"] == AS_OF  # reality.as_of, never wall clock


# ---- 8 + 9. Learning stays intact; history stays immutable ------------------

def test_adding_reality_does_not_change_any_learning_insight():
    """The six Learning generators are untouched by Phase 4.1."""
    without = [r for r in project_values(rich_view())
               if r["insight_type"] not in {t.value for t in REALITY_TYPES}]
    with_reality = [r for r in project_values(rich_view(), rich_reality())
                    if r["insight_type"] not in {t.value for t in REALITY_TYPES}]
    assert snapshot(without) == snapshot(with_reality)
    assert [r["insight_id"] for r in without] == [r["insight_id"] for r in with_reality]


async def test_reality_projection_is_append_only_and_idempotent():
    repo = FakeIntelligenceRepo(rich_view())
    svc = IntelligenceService(repo)
    first = await svc.project(ORG_A, reality=rich_reality())
    n = len(repo.insights)
    second = await svc.project(ORG_A, reality=rich_reality())  # redeliver
    assert snapshot(first) == snapshot(second)
    assert len(repo.insights) == n  # ON CONFLICT DO NOTHING


async def test_a_new_watermark_appends_and_never_mutates_history():
    repo = FakeIntelligenceRepo(rich_view())
    svc = IntelligenceService(repo)
    await svc.project(ORG_A, reality=rich_reality(state_watermark="wm-1"))
    before = {k: (v.title, v.evidence, v.state_watermark) for k, v in repo.insights.items()}
    await svc.project(ORG_A, reality=rich_reality(state_watermark="wm-2"))
    after = {k: (v.title, v.evidence, v.state_watermark) for k, v in repo.insights.items()}
    # Every prior row is present and completely unchanged.
    for k, v in before.items():
        assert after[k] == v
    assert len(after) > len(before)


async def test_an_org_with_no_learning_still_gets_reality_insights():
    """State exists and Reasoning concluded things long before the org's first
    executed action. Gating reality on Learning would have starved every new org."""
    repo = FakeIntelligenceRepo(make_view(learning_version=0))
    rows = await IntelligenceService(repo).project(ORG_A, reality=rich_reality())
    assert rows and all(r["insight_type"] in {t.value for t in REALITY_TYPES} for r in rows)


async def test_no_learning_and_no_reality_still_projects_nothing():
    repo = FakeIntelligenceRepo(make_view(learning_version=0))
    assert await IntelligenceService(repo).project(ORG_A) == []


# ---- Phase 4.1.2 — the READ decides which projection is current --------------
#
# Appending corrected rows (above) fixed the WRITE. The read still served every
# projection an org had ever run, so v3's "4 of 20 tracked relationships are
# cooling" was returned beside v4's corrected "0 of 20" — same watermark, same
# learning_version, same logical time, both live in the API response.


def _types(rows) -> set[str]:
    return {r.insight_type for r in rows}


def _by_key(rows):
    return {(r.insight_type, r.key): r for r in rows}


async def _projected(*realities, view=None, org_id=ORG_A):
    """Project one org through several passes, oldest first, then read it back."""
    repo = FakeIntelligenceRepo(view or rich_view(org_id=org_id))
    svc = IntelligenceService(repo)
    for reality in realities:
        await svc.project(org_id, reality=reality)
    return repo, await repo.get_insights(org_id)


# 1. same (insight_type, key) at two reasoning versions -> only the newest.

async def test_a_superseded_projection_is_not_served_beside_the_current_one():
    """THE live defect, org df7d1c0a: reasoning v3 and v4 folded the SAME State
    snapshot, so both rows carry watermark wm-1 and both reached the read API."""
    repo, rows = await _projected(
        rich_reality(state_watermark="wm-1", reasoning_version=3),
        rich_reality(state_watermark="wm-1", reasoning_version=4),
    )
    keys = [(r.insight_type, r.key) for r in rows]
    assert len(keys) == len(set(keys)), f"a superseded projection is still served: {keys}"

    v4 = {
        v["insight_id"]
        for v in project_values(rich_view(), rich_reality(state_watermark="wm-1",
                                                          reasoning_version=4))
    }
    reality = [r for r in rows if r.insight_type in {t.value for t in REALITY_TYPES}]
    assert reality and all(r.insight_id in v4 for r in reality)


# 1b. the conclusion a corrected projection DROPPED must stop being served.

async def test_a_fact_the_newest_projection_no_longer_states_is_not_served():
    """Org df7d1c0a's actual symptom, and why a per-key rule is not enough.

    v4 emitted NO `risk_intelligence/relationship_cooling` at all — the four
    cooling risks were newsletter noise that organizational relevance removed. A
    rule that kept "the newest row per key" would find no newer row for that key
    and keep serving the v3 one indefinitely. A reality projection is a snapshot:
    the newest pass is the whole current answer.
    """
    cooling_only = rich_reality(state_watermark="wm-1", reasoning_version=3)
    corrected = make_reality(
        state_watermark="wm-1",
        reasoning_version=4,
        entity_counts=cooling_only.entity_counts,
        signals=tuple(
            s for s in cooling_only.signals if s.signal_type != "relationship_cooling"
        ),
    )
    repo, rows = await _projected(cooling_only, corrected)

    served = _by_key(rows)
    assert (InsightType.RISK.value, "relationship_cooling") not in served
    portfolio = served[(InsightType.RELATIONSHIP.value, "portfolio")]
    assert portfolio.evidence["cooling"] == 0
    # ...and the superseded row was never touched, only stopped being served.
    stale = [
        r for r in await repo.get_insights(ORG_A, include_superseded=True)
        if (r.insight_type, r.key) == (InsightType.RISK.value, "relationship_cooling")
    ]
    assert len(stale) == 1 and stale[0].evidence["affected"] == 2


# 2. the two insight families are versioned by different clocks and stay independent.

async def test_learning_insights_are_versioned_per_key_not_per_snapshot():
    """A Learning insight is an accumulating log: each key's newest
    `learning_version` is current, and a key restated later never hides a
    different key that was not restated."""
    repo = FakeIntelligenceRepo(rich_view(learning_version=30))
    await IntelligenceService(repo).project(ORG_A)
    older = await repo.get_insights(ORG_A)

    repo.view = rich_view(learning_version=31)
    await IntelligenceService(repo).project(ORG_A)
    newer = await repo.get_insights(ORG_A)

    assert _by_key(older).keys() == _by_key(newer).keys()  # same logical insights
    assert all(r.learning_version == 31 for r in newer)
    assert len(repo.insights) == len(older) + len(newer)  # both versions stored


async def test_reality_and_learning_insights_do_not_supersede_each_other():
    _repo, rows = await _projected(
        rich_reality(state_watermark="wm-1", reasoning_version=3),
        rich_reality(state_watermark="wm-2", reasoning_version=4),
    )
    learning = {t.value for t in InsightType} - {t.value for t in REALITY_TYPES}
    assert _types(rows) & learning, "reality superseded the Learning insights"
    assert _types(rows) & {t.value for t in REALITY_TYPES}
    # Every reality row is from the newest snapshot; no Learning row was dropped.
    assert {r.state_watermark for r in rows if r.insight_type in
            {t.value for t in REALITY_TYPES}} == {"wm-2"}


# 3. nothing is deleted — the append-only log is intact and still queryable.

async def test_history_is_preserved_and_reachable_through_the_explicit_path():
    repo, current = await _projected(
        rich_reality(state_watermark="wm-1", reasoning_version=3),
        rich_reality(state_watermark="wm-1", reasoning_version=4),
        rich_reality(state_watermark="wm-2", reasoning_version=4),
    )
    everything = await repo.get_insights(ORG_A, include_superseded=True)
    assert len(everything) == len(repo.insights) > len(current)
    assert {r.insight_id for r in current} <= {r.insight_id for r in everything}
    # Three passes' worth of watermarks are all still stored.
    assert {r.state_watermark for r in everything} >= {"wm-1", "wm-2"}


# 4. deterministic ordering — the same stored rows always read back identically.

async def test_the_current_view_is_deterministic():
    from navis.intelligence.repository import current_insights

    repo, first = await _projected(
        rich_reality(state_watermark="wm-1", reasoning_version=3),
        rich_reality(state_watermark="wm-1", reasoning_version=4),
    )
    stored = list(repo.insights.values())
    for permutation in (stored, list(reversed(stored)), sorted(stored, key=lambda r: str(r.key))):
        got = current_insights(permutation)
        assert [r.insight_id for r in got] == [r.insight_id for r in first]
    assert [(r.insight_type, r.key) for r in first] == sorted(
        (r.insight_type, r.key) for r in first
    )


def test_a_stored_rows_reasoning_version_is_recovered_from_its_own_identity():
    """The read boundary's only way to tell v3 from v4: nothing else differs.

    Recovery is a proof (reproduce the id) rather than an inference, so a row
    minted before the reasoning version joined the identity — the live v3 rows,
    keyed on the bare watermark — is recognised as exactly that, not guessed at.
    """
    from types import SimpleNamespace

    from navis.intelligence.models import PRE_VERSIONED_REALITY, recover_reasoning_version

    row = project_values(
        make_view(learning_version=0),
        rich_reality(state_watermark="wm-1", reasoning_version=7),
    )[0]
    stored = SimpleNamespace(**row)
    assert recover_reasoning_version(stored) == 7

    bare = SimpleNamespace(**{**row, "insight_id": compute_insight_id(
        ORG_A, row["insight_type"], row["key"], "wm-1")})
    assert recover_reasoning_version(bare) == PRE_VERSIONED_REALITY

    unknown = SimpleNamespace(**{**row, "insight_id": compute_insight_id(
        ORG_A, row["insight_type"], row["key"], "some-other-scheme")})
    assert recover_reasoning_version(unknown) is None  # unknown, never "old"


# 5. organization isolation.

async def test_the_current_view_never_crosses_organizations():
    _repo_a, rows_a = await _projected(
        rich_reality(org_id=ORG_A, state_watermark="wm-a", reasoning_version=4),
        org_id=ORG_A,
    )
    _repo_b, rows_b = await _projected(
        rich_reality(org_id=ORG_B, state_watermark="wm-b", reasoning_version=3),
        view=rich_view(org_id=ORG_B),
        org_id=ORG_B,
    )
    assert rows_a and rows_b
    assert all(r.organization_id == ORG_A for r in rows_a)
    assert all(r.organization_id == ORG_B for r in rows_b)
    assert {r.insight_id for r in rows_a}.isdisjoint({r.insight_id for r in rows_b})
    # B's older reasoning version is not superseded by A's newer one.
    assert {r.state_watermark for r in rows_b if r.state_watermark} == {"wm-b"}


# 8. an org with a single projection reads back exactly what it did before.

async def test_a_single_projection_is_returned_unchanged():
    repo = FakeIntelligenceRepo(rich_view())
    projected = await IntelligenceService(repo).project(ORG_A, reality=rich_reality())
    rows = await repo.get_insights(ORG_A)
    assert {r.insight_id for r in rows} == {v["insight_id"] for v in projected}
    assert rows == await repo.get_insights(ORG_A, include_superseded=True)


# 9. empty results.

async def test_an_org_with_no_insights_reads_back_empty():
    repo = FakeIntelligenceRepo(make_view(learning_version=0))
    assert await repo.get_insights(ORG_A) == []
    assert await repo.get_insights(ORG_B) == []


# 10. the API response contract is untouched.

async def test_the_serialized_response_contract_is_unchanged():
    from navis.intelligence.router import _serialize

    _repo, rows = await _projected(
        rich_reality(state_watermark="wm-1", reasoning_version=3),
        rich_reality(state_watermark="wm-1", reasoning_version=4),
    )
    payload = [_serialize(r) for r in rows]
    assert payload and set(payload[0]) == {
        "insight_id", "insight_type", "key", "title", "summary", "evidence",
        "sample_size", "confidence", "supporting_learning_event_ids",
        "learning_version", "created_at",
    }
    reality = [p for p in payload if p["insight_type"] in {t.value for t in REALITY_TYPES}]
    assert reality and all(
        p["evidence"]["state_watermark"] == "wm-1"
        and p["evidence"]["supporting_reasoning_signal_ids"]
        for p in reality
    ), "provenance was lost at the read boundary"


# ══════════════════════════════════════════════════════════════════════════════
# CONTEXT AS A PIPELINE STAGE, and OBLIGATION as an insight family
# ══════════════════════════════════════════════════════════════════════════════
#
# Before this, Context (`state.v2.scope.scope_for`) had exactly two callers — the
# `/context` endpoint and Ask Navis — and none between State and Intelligence. So
# an insight could name an affected entity and could not say which remembered
# fact or which captured business-system event it was known from:
# `service._provenance` documents that break in its own docstring. Context held
# the answer the whole time.


def _obligation_reality(*, context=None):
    """A reality view whose signals are the two obligation types."""
    from tests.intelligence.fakes import context_row, reality_signal

    waiting = reality_signal(
        "sig-wait", signal_type="waiting_on_user_reply", entity_type="thread",
        entity_id="thr-1", confidence=0.9, severity=0.65,
        evidence={"person_email": "nick@acme.com", "thread_id": "19fdb"},
    )
    owed = reality_signal(
        "sig-owed", signal_type="decision_owed", entity_type="decision",
        entity_id="dec-1", confidence=0.8, severity=0.5,
        evidence={"person_email": "amy@beta.io", "decision": "ship or hold?"},
    )
    default = (
        context_row(entity_type="thread", entity_id="thr-1", rank=0,
                    reason="most recently active",
                    memory_ids=("mem-thr-1",), signal_event_ids=("evt-1",)),
        context_row(entity_type="decision", entity_id="dec-1", rank=1,
                    reason="open, oldest activity",
                    memory_ids=("mem-dec-1",), signal_event_ids=("evt-2",)),
    )
    return make_reality(
        signals=(waiting, owed),
        context=default if context is None else context,
    )


# ---- 9 + 10. the two blind-spot signal types now produce Intelligence --------

def test_waiting_on_user_reply_now_produces_intelligence():
    """THE regression this family exists for. A waiting card used to reach Today
    with `Intelligence = []`."""
    [row] = _of_type(_reality_rows(_obligation_reality()), InsightType.OBLIGATION) and [
        r for r in _of_type(_reality_rows(_obligation_reality()), InsightType.OBLIGATION)
        if r["key"] == "waiting_on_user_reply"
    ]
    assert row["title"] == "1 outstanding reply obligation"
    assert row["evidence"]["outstanding"] == 1
    assert row["evidence"]["signal_type"] == "waiting_on_user_reply"
    assert row["supporting_learning_event_ids"] == []  # a reality fold, not Learning
    assert row["evidence"]["supporting_reasoning_signal_ids"] == ["sig-wait"]


def test_decision_owed_now_produces_intelligence():
    [row] = [
        r for r in _of_type(_reality_rows(_obligation_reality()), InsightType.OBLIGATION)
        if r["key"] == "decision_owed"
    ]
    assert row["title"] == "1 outstanding decision obligation"
    assert row["evidence"]["supporting_reasoning_signal_ids"] == ["sig-owed"]


def test_each_obligation_kind_stays_its_own_insight():
    """"4 people are waiting" must never hide inside a total with "1 decision is
    owed" — they are different debts and a founder acts on them differently."""
    rows = _of_type(_reality_rows(_obligation_reality()), InsightType.OBLIGATION)
    assert sorted(r["key"] for r in rows) == ["decision_owed", "waiting_on_user_reply"]


def test_no_obligation_signals_produces_no_obligation_claim():
    """Absence of evidence is absence of insight — the rule every reality engine
    is gated on."""
    assert _of_type(_reality_rows(rich_reality()), InsightType.OBLIGATION) == []


def test_obligation_is_not_folded_into_the_risk_count():
    """`status_intelligence`'s "N carrying open risk" counts `_RISK_SIGNAL_TYPES`
    and nothing else. If obligations leaked in, Intelligence and Decision would
    report two different risk numbers for one organization."""
    rows = _reality_rows(_obligation_reality())
    [status] = _of_type(rows, InsightType.STATUS)
    assert status["evidence"]["open_risks"] == 0


def test_obligation_confidence_is_the_weakest_contributing_detection():
    """A claim about several obligations can be no stronger than its weakest
    member — the same rule every other composite insight applies."""
    from tests.intelligence.fakes import reality_signal

    reality = make_reality(signals=(
        reality_signal("s1", signal_type="waiting_on_user_reply", entity_type="thread",
                       entity_id="t1", confidence=0.9, severity=0.4),
        reality_signal("s2", signal_type="waiting_on_user_reply", entity_type="thread",
                       entity_id="t2", confidence=0.55, severity=0.8),
    ))
    [row] = _of_type(_reality_rows(reality), InsightType.OBLIGATION)
    assert row["confidence"] == 0.55           # min, never the mean
    assert row["evidence"]["peak_severity"] == 0.8   # exposure is set by the worst


# ---- 8 + 11. Intelligence receives Context, and cites it ---------------------

def test_an_insight_cites_the_memory_and_business_event_behind_each_entity():
    """THE hop the provenance chain was missing. Before Context reached
    Intelligence, an insight named an entity and stopped there."""
    [row] = [
        r for r in _of_type(_reality_rows(_obligation_reality()), InsightType.OBLIGATION)
        if r["key"] == "waiting_on_user_reply"
    ]
    [entity] = row["evidence"]["entities"]

    assert entity["entity_type"] == "thread" and entity["entity_id"] == "thr-1"
    assert entity["memory_ids"] == ["mem-thr-1"]
    assert entity["signal_event_ids"] == ["evt-1"]
    # ... and WHY Context considered it relevant, in Context's own words.
    assert entity["context_rank"] == 0
    assert entity["context_reason"] == "most recently active"


def test_an_entity_context_did_not_select_carries_no_invented_lineage():
    """Context returns a top-N selection, so an insight may legitimately name an
    entity it did not select. Absence is honest; an empty list would read as "we
    looked and there was none"."""
    from tests.intelligence.fakes import reality_signal

    reality = make_reality(
        signals=(reality_signal("s-x", signal_type="waiting_on_user_reply",
                                entity_type="thread", entity_id="unselected"),),
        context=(),
    )
    [row] = _of_type(_reality_rows(reality), InsightType.OBLIGATION)
    [entity] = row["evidence"]["entities"]

    assert "memory_ids" not in entity and "signal_event_ids" not in entity
    assert "context_rank" not in entity and "context_reason" not in entity


def test_context_lineage_never_crosses_entities():
    """One entity's citations must never be attached to another's — that would be
    fabricated provenance pointing at a real row, the worst kind."""
    rows = _of_type(_reality_rows(_obligation_reality()), InsightType.OBLIGATION)
    by_key = {r["key"]: r for r in rows}

    [thread] = by_key["waiting_on_user_reply"]["evidence"]["entities"]
    [decision] = by_key["decision_owed"]["evidence"]["entities"]
    assert thread["memory_ids"] == ["mem-thr-1"]
    assert decision["memory_ids"] == ["mem-dec-1"]


def test_a_reality_view_without_context_projects_exactly_as_before():
    """Context is additive. An org whose Context could not be built loses lineage
    depth and gains nothing fabricated — every other field is byte-identical."""
    with_ctx = _reality_rows(_obligation_reality())
    without = _reality_rows(_obligation_reality(context=()))

    def strip(rows):
        out = []
        for r in rows:
            e = dict(r["evidence"])
            if "entities" in e:
                e["entities"] = [
                    {k: v for k, v in ent.items()
                     if k not in {"memory_ids", "signal_event_ids",
                                  "context_rank", "context_reason"}}
                    for ent in e["entities"]
                ]
            out.append({**r, "evidence": e})
        return snapshot(out)

    assert strip(with_ctx) == strip(without)


def test_context_does_not_change_insight_identity():
    """`compute_insight_id` hashes (org, type, key, watermark|reasoning_version).
    Context deepens an insight's evidence; it must never fork its id, or every
    projection with a differently-capped Context would mint a parallel row."""
    a = {r["key"]: r["insight_id"] for r in _reality_rows(_obligation_reality())}
    b = {r["key"]: r["insight_id"] for r in _reality_rows(_obligation_reality(context=()))}
    assert a == b


# ---- 20. no fake ML / LLM path ----------------------------------------------

def test_no_generator_is_backed_by_a_model():
    """Every reality insight names a deterministic engine. A model-backed
    producer would break this table's core invariant — "drop the table,
    re-project, get identical rows" — and there is no column for a model id, a
    prompt version, or a wall-clock generated_at to make it safe."""
    engines = {r["engine_name"] for r in _reality_rows(_obligation_reality())}
    assert engines and all(e.startswith("intelligence.reality.") for e in engines)
    for banned in ("llm", "model", "gpt", "claude", "gemini", "azure", "openai"):
        assert not any(banned in e.lower() for e in engines)


# ══════════════════════════════════════════════════════════════════════════════
# 7.5.7 — Situation Intelligence: the engine that connects instead of counting
#
# Every other reality engine buckets signals by TYPE ("2 open commitment
# slipping risks"), which restates Reasoning one altitude up and connects
# nothing. These pin the one engine that reads State's persisted entity graph —
# and, just as importantly, pin that it refuses to connect what State did not.
# ══════════════════════════════════════════════════════════════════════════════

from tests.intelligence.fakes import (  # noqa: E402
    connected_counterparty_context,
    context_link,
    context_row,
)


def _connected_reality(org_id=ORG_A, **over):
    """The AJITESH case: a slipping commitment and a cooling relationship that
    State links to ONE counterparty."""
    signals = (
        reality_signal(
            "sig-commit-1", signal_type="commitment_slipping",
            entity_type="commitment", entity_id="com-1",
            confidence=0.9, severity=0.84,
            evidence={"age_days": 13.0, "speaker_email": "ajitesh@example.com"},
        ),
        reality_signal(
            "sig-cool-1", signal_type="relationship_cooling",
            entity_type="relationship", entity_id="rel-1",
            confidence=0.85, severity=0.6,
            evidence={"days_since_contact": 10.0, "person_email": "ajitesh@example.com"},
        ),
    )
    over.setdefault("context", connected_counterparty_context())
    return make_reality(org_id=org_id, signals=signals, **over)


def test_a_connected_counterparty_produces_one_situation():
    rows = _of_type(_reality_rows(_connected_reality()), InsightType.SITUATION)
    assert len(rows) == 1


def test_the_situation_names_the_counterparty_and_both_conclusions():
    """The headline must be the connected sentence, not a bucket count."""
    row = _of_type(_reality_rows(_connected_reality()), InsightType.SITUATION)[0]
    assert "ajitesh@example.com" in row["title"]
    assert "unresolved commitment" in row["title"]
    assert "cooling relationship" in row["title"]


def test_the_situation_cites_both_reasoning_signals():
    """This is what makes ONE organizational sentence reach BOTH decisions —
    `what_matters_today._relevant_facts` matches on the cited signal id."""
    row = _of_type(_reality_rows(_connected_reality()), InsightType.SITUATION)[0]
    assert set(row["evidence"]["supporting_reasoning_signal_ids"]) == {
        "sig-commit-1", "sig-cool-1"
    }


def test_the_situation_carries_the_grounded_representation():
    row = _of_type(_reality_rows(_connected_reality()), InsightType.SITUATION)[0]
    ev = row["evidence"]
    assert ev["situation_type"] == "commitment_slipping+relationship_cooling"
    assert ev["primary_entity"]["entity_id"] == "rel-1"
    assert ev["linked_entity_counts"] == {"commitment": 1, "meeting": 2, "person": 1}
    assert {r["entity_type"] for r in ev["related_entities"]} == {
        "commitment", "meeting", "person"
    }
    # Memory/business-system lineage, carried as POINTERS (Intelligence may not
    # read Memory) — the hop the provenance chain used to stop short of.
    assert ev["memory_ids"] and ev["signal_event_ids"]


def test_every_related_entity_names_the_edge_that_put_it_there():
    """"Why is this meeting here?" must be answerable from the stored row."""
    row = _of_type(_reality_rows(_connected_reality()), InsightType.SITUATION)[0]
    for related in row["evidence"]["related_entities"]:
        assert related["via_role"] in {
            "commitments", "meetings", "relationships_of", "owner", "owner_of"
        }


def test_the_implication_states_the_link_and_never_a_cause():
    row = _of_type(_reality_rows(_connected_reality()), InsightType.SITUATION)[0]
    implication = row["evidence"]["organizational_implication"].lower()
    assert "connects them through" in implication
    for causal in ("because", "caused", "due to", "led to", "resulted in"):
        assert causal not in implication


# ---- the refusals: what it must NOT connect ---------------------------------


def test_unrelated_entities_are_never_joined_into_one_situation():
    """TWO counterparties, each with their own conclusion, NO edge between them.

    This is the false join the engine exists to avoid: without the link rule it
    would be trivial to bundle any two signals that happen to co-occur.
    """
    signals = (
        reality_signal(
            "sig-commit-x", signal_type="commitment_slipping",
            entity_type="commitment", entity_id="com-x",
            evidence={"speaker_email": "x@example.com"},
        ),
        reality_signal(
            "sig-cool-y", signal_type="relationship_cooling",
            entity_type="relationship", entity_id="rel-y",
            evidence={"person_email": "y@example.com"},
        ),
    )
    context = (
        context_row(entity_type="commitment", entity_id="com-x", rank=0, links=()),
        context_row(entity_type="relationship", entity_id="rel-y", rank=1, links=()),
    )
    rows = _of_type(
        _reality_rows(make_reality(signals=signals, context=context)),
        InsightType.SITUATION,
    )
    assert rows == []


def test_a_shared_company_does_not_merge_two_counterparties():
    """`people` edges (company -> person, channel -> person) are deliberately NOT
    traversable. A company with two contacts must stay two counterparties, or one
    "situation" would silently span an entire customer account."""
    signals = (
        reality_signal(
            "sig-cool-a", signal_type="relationship_cooling",
            entity_type="relationship", entity_id="rel-a",
            evidence={"person_email": "a@acme.com"},
        ),
        reality_signal(
            "sig-cool-b", signal_type="relationship_cooling",
            entity_type="relationship", entity_id="rel-b",
            evidence={"person_email": "b@acme.com"},
        ),
    )
    context = (
        context_row(entity_type="relationship", entity_id="rel-a", rank=0, links=()),
        context_row(entity_type="relationship", entity_id="rel-b", rank=1, links=()),
        context_row(
            entity_type="company", entity_id="co-acme", rank=2,
            links=(
                context_link(role="people", entity_type="person", entity_id="per-a"),
                context_link(role="people", entity_type="person", entity_id="per-b"),
            ),
        ),
    )
    rows = _of_type(
        _reality_rows(make_reality(signals=signals, context=context)),
        InsightType.SITUATION,
    )
    assert rows == []


def test_one_conclusion_alone_is_not_a_situation():
    """A single conclusion about a single entity is a ReasoningSignal. Repeating
    it as a "situation" would be exactly the restatement this engine replaces."""
    signals = (
        reality_signal(
            "sig-cool-1", signal_type="relationship_cooling",
            entity_type="relationship", entity_id="rel-1",
            evidence={"person_email": "solo@example.com"},
        ),
    )
    rows = _of_type(
        _reality_rows(
            make_reality(signals=signals, context=connected_counterparty_context())
        ),
        InsightType.SITUATION,
    )
    assert rows == []


def test_without_context_no_situation_is_emitted():
    """Context is the only source of edges. No Context => no claim about them,
    and every other reality engine keeps working exactly as before."""
    rows = _reality_rows(_connected_reality(context=()))
    assert _of_type(rows, InsightType.SITUATION) == []
    assert _of_type(rows, InsightType.RISK), "aggregation engines must still fire"


def test_a_situation_is_versioned_by_state_and_reasoning_like_every_reality_row():
    first = _of_type(_reality_rows(_connected_reality()), InsightType.SITUATION)[0]
    moved = _of_type(
        _reality_rows(_connected_reality(state_watermark="wm-state-2")),
        InsightType.SITUATION,
    )[0]
    assert first["insight_id"] != moved["insight_id"]
    assert first["key"] == moved["key"]


def test_a_situation_is_deterministic():
    a = _of_type(_reality_rows(_connected_reality()), InsightType.SITUATION)[0]
    b = _of_type(_reality_rows(_connected_reality()), InsightType.SITUATION)[0]
    assert a == b


def test_a_situation_cannot_cross_tenants():
    a = _of_type(_reality_rows(_connected_reality(org_id=ORG_A)), InsightType.SITUATION)[0]
    b = _of_type(
        _reality_rows(_connected_reality(org_id=ORG_B), view=make_view(
            org_id=ORG_B, learning_version=0)),
        InsightType.SITUATION,
    )[0]
    assert a["insight_id"] != b["insight_id"]
    assert a["organization_id"] == ORG_A
    assert b["organization_id"] == ORG_B


def test_the_situation_confidence_is_the_weakest_contributing_signal():
    """A composite claim can be no stronger than its weakest member — the same
    rule every other composite insight follows."""
    row = _of_type(_reality_rows(_connected_reality()), InsightType.SITUATION)[0]
    assert row["confidence"] == 0.85
    assert row["sample_size"] == 2
