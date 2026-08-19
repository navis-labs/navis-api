"""Adaptive V1: determinism, replay parity, governance, safety, multi-tenancy,
Learning/Intelligence-only dependency, and default-parity with live constants.
"""
from __future__ import annotations

import ast
import uuid
import pathlib

import pytest

from navis.adaptive import policy as P
from navis.adaptive import proposals
from navis.adaptive.contract import ProposalStatus, compute_proposal_id
from navis.adaptive.governance import NotAuthorized, require_approver, validate_proposal
from navis.adaptive.replay import AdaptiveReplayService
from navis.adaptive.service import AdaptiveService
from tests.adaptive.fakes import (
    AS_OF,
    AT,
    FOUNDER,
    ORG_A,
    ORG_B,
    FakeAdaptiveRepo,
    make_view,
    org_insight,
    proposal_snapshot,
)

PKG = pathlib.Path(__file__).resolve().parent.parent.parent / "navis" / "adaptive"
FORBIDDEN_DOMAINS = {"signals", "memory", "state", "reasoning", "decisions", "execution", "observation"}


# ---- defaults EXACTLY match the live hardcoded constants (no behavior change) ----

def test_defaults_match_live_constants():
    from navis.decisions.what_matters_today import WhatMattersTodayEngine
    from navis.execution.config import MAX_EXECUTION_ATTEMPTS
    from navis.reasoning.v2.engines import meta, risk

    eng = WhatMattersTodayEngine()
    assert P.DEFAULT_POLICY["commitment_slipping_days"] == risk.COMMITMENT_SLIPPING_DAYS
    assert P.DEFAULT_POLICY["relationship_cooling_days"] == risk.RELATIONSHIP_COOLING_DAYS
    assert P.DEFAULT_POLICY["project_stalled_days"] == risk.PROJECT_STALLED_DAYS
    assert P.DEFAULT_POLICY["customer_drop_days"] == risk.CUSTOMER_DROP_DAYS
    assert P.DEFAULT_POLICY["meeting_overload_count"] == risk.MEETING_OVERLOAD_COUNT
    assert P.DEFAULT_POLICY["leverage_min_people"] == meta.LEVERAGE_MIN_PEOPLE
    assert P.DEFAULT_POLICY["brief_max_items"] == eng.max_items
    assert P.DEFAULT_POLICY["brief_min_confidence"] == eng.min_confidence
    assert P.DEFAULT_POLICY["execution_retry_limit"] == MAX_EXECUTION_ATTEMPTS


# ---- deterministic proposal ids + same-in-same-out ----

def test_proposal_id_is_deterministic_and_versioned():
    a = compute_proposal_id(ORG_A, 20)
    assert a == compute_proposal_id(ORG_A, 20)
    assert a != compute_proposal_id(ORG_A, 21)
    assert a != compute_proposal_id(ORG_B, 20)


def test_same_view_same_proposal():
    view = make_view()
    d1, d2 = proposals.propose(view), proposals.propose(view)
    assert d1 is not None
    assert d1.proposal_id == d2.proposal_id
    assert d1.proposed_payload == d2.proposed_payload
    assert [c.to_value for c in d1.changes] == [c.to_value for c in d2.changes]


def test_ambiguous_track_record_proposes_nothing():
    # positive rate exactly 0.5 -> neither trust nor distrust -> no proposal.
    view = make_view(insights=[org_insight(positive=10, negative=10)])
    assert proposals.propose(view) is None


# ---- Phase 4.1 safety: reality insights are INFORMATIONAL, never governing ----
#
# Intelligence gained six reality-derived insight types. They land in the same
# append-only table Adaptive reads, so `AdaptiveView.insights` now carries rows
# that were never meant to move a policy threshold. Nothing in `proposals.py`
# was changed to accommodate them — these tests exist to prove that the
# admissibility gate (`_org_insight`, which selects on insight_type) is what
# keeps them out, rather than luck about ordering or counts.


def _reality_insight(insight_type: str, *, sample_size=50, learning_version=20):
    """A reality-derived insight row as AdaptiveRepository would load it."""
    from navis.adaptive.contract import InsightRow

    return InsightRow(
        insight_id=str(uuid.uuid5(uuid.NAMESPACE_OID, f"reality|{insight_type}")),
        insight_type=insight_type,
        key="portfolio",
        # Deliberately shaped like the one Adaptive DOES admit: if the gate were
        # counting evidence instead of checking the type, this would pass it.
        evidence={"positive": 40, "negative": 2, "cooling": 6},
        sample_size=sample_size,
        confidence=0.9,
        supporting_learning_event_ids=[],
        learning_version=learning_version,
    )


_REALITY_TYPES = [
    "risk_intelligence", "opportunity_intelligence", "relationship_intelligence",
    "commitment_intelligence", "organizational_status", "trend_intelligence",
]


@pytest.mark.parametrize("insight_type", _REALITY_TYPES)
def test_a_reality_insight_alone_can_never_propose_policy(insight_type):
    """No admissible evidence => no proposal, however rich the reality is."""
    view = make_view(insights=[_reality_insight(insight_type)])
    assert proposals.propose(view) is None


def test_reality_insights_do_not_alter_an_existing_proposal():
    """Adding every reality type must leave the proposal byte-identical."""
    baseline = proposals.propose(make_view())
    assert baseline is not None
    with_reality = proposals.propose(
        make_view(insights=[org_insight()] + [_reality_insight(t) for t in _REALITY_TYPES])
    )
    assert with_reality is not None
    assert with_reality.proposal_id == baseline.proposal_id
    assert with_reality.proposed_payload == baseline.proposed_payload
    assert [c.to_value for c in with_reality.changes] == [c.to_value for c in baseline.changes]
    # Only the admitted organizational_intelligence insight is ever cited.
    assert with_reality.supporting_insight_ids == baseline.supporting_insight_ids


def test_the_admissible_insight_type_is_a_closed_set():
    """A new insight type must not become policy-bearing by simply existing."""
    from navis.intelligence.models import InsightType

    assert proposals._ORG_INSIGHT == InsightType.ORGANIZATIONAL.value
    admissible = {proposals._ORG_INSIGHT}
    assert admissible < {t.value for t in InsightType}  # strict subset


# ---- replay parity (delete + rebuild identical) ----

async def test_delete_and_rebuild_is_identical():
    repo = FakeAdaptiveRepo(make_view())
    await AdaptiveService(repo).propose(ORG_A)
    before = proposal_snapshot(repo.proposals.values())
    assert before  # something was proposed

    await AdaptiveReplayService(repo=repo).rebuild(ORG_A)
    after = proposal_snapshot(repo.proposals.values())
    assert after == before


async def test_propose_is_idempotent_append_only():
    repo = FakeAdaptiveRepo(make_view())
    svc = AdaptiveService(repo)
    await svc.propose(ORG_A)
    n = len(repo.proposals)
    await svc.propose(ORG_A)  # redeliver
    assert len(repo.proposals) == n  # ON CONFLICT DO NOTHING


# ---- multi-tenant isolation ----

async def test_multi_tenant_isolation():
    ra = FakeAdaptiveRepo(make_view(org_id=ORG_A))
    rb = FakeAdaptiveRepo(make_view(org_id=ORG_B))
    await AdaptiveService(ra).propose(ORG_A)
    await AdaptiveService(rb).propose(ORG_B)
    ids_a = set(ra.proposals.keys())
    ids_b = set(rb.proposals.keys())
    assert ids_a and ids_b and ids_a.isdisjoint(ids_b)


# ---- approval workflow + governance ----

async def test_only_founder_can_approve_and_it_activates():
    repo = FakeAdaptiveRepo(make_view())
    svc = AdaptiveService(repo)
    prop = await svc.propose(ORG_A)
    pid = prop["proposal_id"]

    with pytest.raises(NotAuthorized):
        await svc.approve(ORG_A, pid, approver_id=FOUNDER, role="employee", at=AT)

    await svc.approve(ORG_A, pid, approver_id=FOUNDER, role="founder", at=AT)
    assert repo.proposals[pid].status == ProposalStatus.APPROVED.value
    active = await repo.get_active_snapshot(ORG_A)
    assert active is not None and active.activated_at == AT
    assert len(repo.approvals) == 1


async def test_reject_blocks_activation():
    repo = FakeAdaptiveRepo(make_view())
    svc = AdaptiveService(repo)
    prop = await svc.propose(ORG_A)
    await svc.reject(ORG_A, prop["proposal_id"], approver_id=FOUNDER, role="founder", at=AT)
    assert repo.proposals[prop["proposal_id"]].status == ProposalStatus.REJECTED.value
    assert await repo.get_active_snapshot(ORG_A) is None  # nothing active


# ---- rollback + kill switch ----

async def test_rollback_reactivates_prior_payload():
    repo = FakeAdaptiveRepo(make_view())
    svc = AdaptiveService(repo)
    prop = await svc.propose(ORG_A)
    await svc.approve(ORG_A, prop["proposal_id"], approver_id=FOUNDER, role="founder", at=AT)
    v1 = await svc.get_active_policy(ORG_A)
    v1_policy_id = (await repo.get_active_snapshot(ORG_A)).policy_id

    # kill switch -> defaults active
    await svc.kill_switch(ORG_A, approver_id=FOUNDER, role="founder", at=AT)
    assert await svc.get_active_policy(ORG_A) == P.DEFAULT_POLICY

    # rollback to v1 -> v1 payload active again
    await svc.rollback(ORG_A, v1_policy_id, approver_id=FOUNDER, role="founder", at=AT)
    assert await svc.get_active_policy(ORG_A) == v1


async def test_kill_switch_requires_founder():
    repo = FakeAdaptiveRepo(make_view())
    with pytest.raises(NotAuthorized):
        await AdaptiveService(repo).kill_switch(ORG_A, approver_id=FOUNDER, role="admin", at=AT)


# ---- safety: bounds + magnitude caps ----

def test_bounds_clamp():
    assert P.clamp("relationship_cooling_days", 9999) == 60.0
    assert P.clamp("relationship_cooling_days", -5) == 1.0
    assert P.clamp("brief_max_items", 1000) == 25


def test_change_magnitude_capped_to_25pct():
    val, capped = P.cap_change("relationship_cooling_days", 8.0, 100.0)
    assert capped and val == 10.0  # 8 * 1.25
    val2, capped2 = P.cap_change("relationship_cooling_days", 8.0, 9.0)
    assert not capped2 and val2 == 9.0


def test_validate_change_rejects_big_jump():
    with pytest.raises(P.PolicyViolation):
        P.validate_change("brief_min_confidence", 0.45, 0.90)  # +100% > 25%


def test_validate_change_rejects_out_of_bounds():
    with pytest.raises(P.PolicyViolation):
        P.validate_change("execution_retry_limit", 5, 50)


def test_proposed_changes_never_exceed_caps():
    # Even with a wildly positive org, capping keeps each change within 25%.
    view = make_view(insights=[org_insight(positive=999, negative=1)])
    draft = proposals.propose(view)
    assert draft is not None
    for c in draft.changes:
        old, new = c.from_value, c.to_value
        assert abs(new - old) <= abs(old) * P.MAX_CHANGE_RATIO + 1e-9
    validate_proposal(draft, view.active_policy)  # independent gate passes


def test_require_approver():
    require_approver("founder")  # ok
    for bad in ("ceo", "coo", "admin", "employee", "owner"):
        with pytest.raises(NotAuthorized):
            require_approver(bad)


# ---- no forbidden imports (Learning + Intelligence only) ----

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
            if len(parts) >= 2 and parts[0] == "navis" and parts[1] in FORBIDDEN_DOMAINS:
                violations.append(f"{py.name} imports {mod}")
    assert not violations, "Adaptive may only depend on Intelligence + Learning:\n" + "\n".join(violations)


# ---- Intelligence is read through ITS OWN boundary, never queried directly ----
#
# `intelligence_insights` is append-only. A bare `select(IntelligenceInsight)` —
# which this module did until Phase 4.1.3 — returns every projection the org has
# ever run, so Adaptive could propose POLICY from a superseded conclusion while
# GET /intelligence/insights served the corrected one. The behavioural proof runs
# against Postgres in tests/integration/test_intelligence_read_path_postgres.py;
# this is the DB-free guard that protects the boundary everywhere else.


def test_adaptive_does_not_import_the_intelligence_orm_model():
    """Importing the model is what makes a private query possible at all.

    Structural rather than textual: Adaptive may name Intelligence in prose, but
    it must reach the rows through `navis.intelligence.repository`.
    """
    offenders = []
    for py in PKG.rglob("*.py"):
        tree = ast.parse(py.read_text(), filename=str(py))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "navis.intelligence.models":
                offenders.append(f"{py.name}:{node.lineno} imports {node.module}")
            elif isinstance(node, ast.Import):
                offenders += [
                    f"{py.name}:{node.lineno} imports {a.name}"
                    for a in node.names
                    if a.name == "navis.intelligence.models"
                ]
    assert not offenders, (
        "Adaptive must read Intelligence through IntelligenceRepository."
        "get_insights, not by querying the table itself:\n" + "\n".join(offenders)
    )


def test_build_view_uses_the_intelligence_read_boundary():
    import inspect

    from navis.adaptive.repository import AdaptiveRepository

    src = inspect.getsource(AdaptiveRepository.build_view)
    assert "IntelligenceRepository(self.db).get_insights(" in src, (
        "build_view stopped using Intelligence's read boundary; it will serve "
        "superseded insights to policy again"
    )
    assert "select(IntelligenceInsight" not in src


# ---- determinism guard: no wall clock / uuid4 / random in the projection ----

def test_no_nondeterministic_calls_in_projection():
    projection = ["policy.py", "contract.py", "proposals.py", "models.py", "governance.py"]
    banned = {"now", "utcnow", "uuid4", "uuid1", "random", "randint", "choice", "shuffle"}
    offenders = []
    for name in projection:
        tree = ast.parse((PKG / name).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in banned:
                    offenders.append(f"{name}:{node.lineno} .{node.func.attr}()")
            if isinstance(node, ast.Import) and any(a.name == "random" for a in node.names):
                offenders.append(f"{name}: import random")
    assert not offenders, "non-deterministic calls in Adaptive projection:\n" + "\n".join(offenders)


async def test_created_at_is_logical_time():
    repo = FakeAdaptiveRepo(make_view())
    row = await AdaptiveService(repo).propose(ORG_A)
    assert row["created_at"] == AS_OF  # logical, from Intelligence


# ---- auditability: every change cites its evidence ----

def test_every_change_cites_insight_and_learning():
    draft = proposals.propose(make_view())
    assert draft is not None
    for c in draft.changes:
        assert c.supporting_insight_ids and c.supporting_learning_ids
        assert c.reason
