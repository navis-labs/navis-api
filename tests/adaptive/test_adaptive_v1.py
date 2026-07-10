"""Adaptive V1: determinism, replay parity, governance, safety, multi-tenancy,
Learning/Intelligence-only dependency, and default-parity with live constants.
"""
from __future__ import annotations

import ast
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


def test_merge_defaults_is_total_and_fail_safe():
    """Regression: merge_defaults is the READ seam producing the effective policy the
    pipeline consumes (resolve_policy documents "never raises"). Sanctioned writers all
    validate, but a corrupt or historical stored value must DEGRADE, never propagate:
    out-of-bounds numbers clamp to the schema's declared safe bounds and un-castable
    values fall back to the field default. Without this, one bad stored value either
    crashes brief assembly for the org (ValueError) or silently empties its briefs
    (brief_min_confidence=5.0 filters every signal)."""
    # Valid payload: identical to the pre-hardening cast behavior.
    ok = P.merge_defaults({"brief_max_items": 9, "brief_min_confidence": 0.5})
    assert ok["brief_max_items"] == 9 and ok["brief_min_confidence"] == 0.5

    # Out-of-bounds numerics clamp to the declared safe bounds.
    clamped = P.merge_defaults({"brief_min_confidence": 5.0, "brief_max_items": -3})
    assert clamped["brief_min_confidence"] == 0.95  # can no longer filter every signal
    assert clamped["brief_max_items"] == 1

    # Un-castable values degrade to the safe code default instead of raising.
    corrupt = P.merge_defaults({"brief_min_confidence": "abc", "brief_max_items": None})
    assert corrupt["brief_min_confidence"] == P.DEFAULT_POLICY["brief_min_confidence"]
    assert corrupt["brief_max_items"] == P.DEFAULT_POLICY["brief_max_items"]

    # Unknown fields are still dropped (unchanged behavior).
    assert "mystery" not in P.merge_defaults({"mystery": 1})


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
