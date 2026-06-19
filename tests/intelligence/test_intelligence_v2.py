"""Intelligence V2: determinism, replay parity, lineage, the six generators,
Learning-only dependency, and the no-clock/no-uuid4 determinism guards.
"""
from __future__ import annotations

import ast
import pathlib

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
    make_view,
    rich_view,
    snapshot,
)

PKG = pathlib.Path(__file__).resolve().parent.parent.parent / "navis" / "intelligence"
PIPELINE_DOMAINS = {
    "signals", "memory", "state", "reasoning",
    "decisions", "execution", "observation", "learning",
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
        tree = ast.parse(py.read_text(), filename=str(py))
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
        tree = ast.parse(py.read_text(), filename=str(py))
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

def test_insight_types_are_exactly_the_six_facts():
    assert {t.value for t in InsightType} == {
        "signal_intelligence", "decision_intelligence", "team_intelligence",
        "customer_intelligence", "organizational_intelligence", "adaptation_intelligence",
    }


def test_no_recommendation_or_score_columns():
    from navis.intelligence.models import IntelligenceInsight

    cols = set(IntelligenceInsight.__table__.columns.keys())
    forbidden = {"recommendation", "action", "score", "risk", "priority", "advice"}
    assert cols & forbidden == set()
