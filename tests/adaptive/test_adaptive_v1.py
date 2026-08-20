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


# ---- Adaptive reads no Learning TABLE -----------------------------------------

def test_adaptive_does_not_read_learning_aggregates_directly():
    """`AdaptiveRepository._aggs` is gone, and must not come back by accident.

    It ran three unbounded `learning_aggregates` reads per view and filled three
    `AdaptiveView` fields nothing ever read — on a 100k-decision organization,
    100k rows fetched and discarded. Deleting it removed Adaptive's only direct
    read of a Learning table.

    An AST check rather than a behavioural one, because the regression is a
    single convenience import ("Adaptive needs success rates now") and that would
    reintroduce the unbounded read without failing anything else. When Adaptive
    does need Learning, it goes through `navis.learning.metrics.learning_snapshot`
    — the contract Intelligence and ML already consume — and this test should be
    updated to allow THAT, not a table.
    """
    import ast
    from pathlib import Path

    import navis.adaptive as pkg

    violations = []
    for py in Path(pkg.__file__).parent.rglob("*.py"):
        tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        for node in ast.walk(tree):
            module = None
            if isinstance(node, ast.ImportFrom) and node.module:
                module = node.module
            elif isinstance(node, ast.Import):
                module = next(
                    (a.name for a in node.names if a.name.startswith("navis.learning")),
                    None,
                )
            if module and module.startswith("navis.learning"):
                # `navis.learning.metrics` IS the contract — the same one
                # Intelligence and ML consume. Everything else under
                # navis.learning is Learning's private storage.
                if module != "navis.learning.metrics":
                    violations.append(f"{py.name}:{node.lineno} imports {module}")
    assert not violations, (
        "Adaptive reached past the Learning contract into its storage; the only "
        "permitted import is navis.learning.metrics:\n" + "\n".join(violations)
    )


def test_adaptive_view_carries_no_unread_aggregate_fields():
    """The fields are gone from the contract, not merely unpopulated."""
    import dataclasses

    from navis.adaptive.contract import AdaptiveView
    from navis.adaptive.repository import AdaptiveRepository

    names = {f.name for f in dataclasses.fields(AdaptiveView)}
    assert names == {
        "org_id", "adaptive_version", "as_of", "active_policy", "insights", "learning",
    }
    assert not hasattr(AdaptiveRepository, "_aggs"), "the dead aggregate read is back"

    # Every field is READ by the proposal fold — that is what distinguishes
    # `learning` (added with a consumer) from the aggregate fields (added
    # without one, and deleted for it).
    import inspect

    from navis.adaptive import proposals

    source = inspect.getsource(proposals)
    for field in names - {"org_id", "as_of"}:  # both are carried into the draft
        assert f"view.{field}" in source or f".{field}" in source, (
            f"AdaptiveView.{field} is carried but never read"
        )


# ---- the governed vocabulary declares WHAT it governs -------------------------

def test_every_policy_field_declares_the_layer_it_tunes():
    """`target` is what makes "Adaptive tunes thinking, not permission" testable."""
    from navis.adaptive.policy import POLICY_FIELDS, PolicyTarget, fields_for

    assert POLICY_FIELDS, "the governed vocabulary is empty"
    for name, spec in POLICY_FIELDS.items():
        assert isinstance(spec.target, PolicyTarget), f"{name} declares no target"

    # Every field is reachable through exactly one target, so the partition is total.
    partitioned = [n for t in PolicyTarget for n in fields_for(t)]
    assert sorted(partitioned) == sorted(POLICY_FIELDS)
    assert len(partitioned) == len(set(partitioned))


def test_no_policy_field_can_grant_capability_or_permission():
    """THE SAFETY INVARIANT, as an assertion rather than a convention.

    Adaptive governs numeric thresholds. It must never acquire a field that
    grants a capability, widens an entitlement, touches a credential, or decides
    whether an action may execute — those live in `navis.organization`
    (entitlements/flags) and Execution's authorization path, and a policy field
    is approved by a founder reviewing a NUMBER, not a permission grant.

    Every field is also numeric and bounded: a bool or a string here would mean
    the founder is approving something other than a magnitude.
    """
    from navis.adaptive.policy import POLICY_FIELDS, PolicyTarget

    banned = (
        "capability", "capabilities", "permission", "entitlement", "credential",
        "token", "scope", "role", "auth", "authorize", "authorization", "grant",
        "allow", "enable", "admin", "bypass", "override",
    )
    for name, spec in POLICY_FIELDS.items():
        lowered = name.lower()
        for word in banned:
            assert word not in lowered, f"policy field {name!r} looks like a grant"
        assert spec.kind in {"int", "float"}, f"{name} is not a magnitude"
        assert spec.min < spec.max, f"{name} has no usable range"

    # And no field targets anything but the three tuning layers.
    assert {s.target for s in POLICY_FIELDS.values()} <= set(PolicyTarget)


# ---- resolution fails CLOSED -------------------------------------------------

def test_a_corrupted_policy_row_falls_back_to_defaults():
    """`merge_defaults` is the last thing between a stored row and the runtime.

    `cast()` rounds but does not clamp, so an out-of-bounds stored value used to
    be served verbatim. Proposals are bounds-checked on write AND at approval, so
    a row that arrives here out of range is damage — and damage must not become
    policy.
    """
    from navis.adaptive.policy import DEFAULT_POLICY, merge_defaults

    corrupt = {
        "brief_max_items": 10_000,        # far above max=25
        "brief_min_confidence": -3.0,     # below min=0.0
        "meeting_overload_count": "many", # not a number
        "leverage_min_people": None,      # not a number
        "grant_admin": True,              # unknown field
    }
    resolved = merge_defaults(corrupt)

    assert resolved["brief_max_items"] == DEFAULT_POLICY["brief_max_items"]
    assert resolved["brief_min_confidence"] == DEFAULT_POLICY["brief_min_confidence"]
    assert resolved["meeting_overload_count"] == DEFAULT_POLICY["meeting_overload_count"]
    assert resolved["leverage_min_people"] == DEFAULT_POLICY["leverage_min_people"]
    assert "grant_admin" not in resolved, "an unknown field entered the policy"
    assert set(resolved) == set(DEFAULT_POLICY)


def test_a_valid_policy_row_is_still_honoured_exactly():
    """Failing closed must not mean ignoring legitimate policy."""
    from navis.adaptive.policy import DEFAULT_POLICY, merge_defaults

    resolved = merge_defaults({"brief_max_items": 5, "brief_min_confidence": 0.6})
    assert resolved["brief_max_items"] == 5
    assert resolved["brief_min_confidence"] == 0.6
    # Untouched fields keep the code default.
    assert resolved["execution_retry_limit"] == DEFAULT_POLICY["execution_retry_limit"]


def test_boundary_values_are_accepted_not_rejected():
    """min and max are inclusive; a founder may approve the extreme."""
    from navis.adaptive.policy import POLICY_FIELDS, merge_defaults

    spec = POLICY_FIELDS["brief_max_items"]
    assert merge_defaults({"brief_max_items": spec.min})["brief_max_items"] == spec.min
    assert merge_defaults({"brief_max_items": spec.max})["brief_max_items"] == spec.max


def test_no_active_policy_resolves_to_todays_behaviour():
    """The whole contract is inert until a founder activates something."""
    from navis.adaptive.policy import DEFAULT_POLICY, merge_defaults

    assert merge_defaults(None) == DEFAULT_POLICY
    assert merge_defaults({}) == DEFAULT_POLICY


# ---- 6.3.1: LEARNED EVIDENCE -> CANDIDATE ------------------------------------
#
# The rules above fold Intelligence insights. These cover the bridge that folds
# Learning's OWN counters — specifically how often a founder rejected what Navis
# proposed, which no insight expresses.


def _evidence(*, executed: int, rejected: int, org_id=ORG_A, learning_version=20):
    from navis.adaptive.contract import LearningEvidence, OutcomeEvidence

    outcomes = []
    if executed:
        outcomes.append(OutcomeEvidence(
            "action_executed", "delivery", 0, 0, executed, executed,
            tuple(f"le-exec-{i}" for i in range(min(executed, 3))),
        ))
    if rejected:
        outcomes.append(OutcomeEvidence(
            "action_rejected", "judgement", 0, rejected, 0, rejected,
            tuple(f"le-rej-{i}" for i in range(min(rejected, 3))),
        ))
    return LearningEvidence(
        org_id=org_id, learning_version=learning_version, as_of=AS_OF,
        outcomes=tuple(outcomes),
    )


def _learning_only_view(evidence, **kw):
    """A view with NO admissible insight, so only the learned rule can fire."""
    return make_view(insights=[], **kw).__class__(
        org_id=ORG_A,
        adaptive_version=evidence.learning_version,
        as_of=AS_OF,
        active_policy=P.merge_defaults(None),
        insights=[],
        learning=evidence,
    )


def test_learned_rejection_evidence_produces_a_candidate():
    """A founder rejecting most proposals is evidence Navis surfaces too freely."""
    draft = proposals.propose(_learning_only_view(_evidence(executed=2, rejected=18)))
    assert draft is not None, "strong learned evidence produced no candidate"
    fields = {c.field for c in draft.changes}
    assert fields <= {"brief_min_confidence", "brief_max_items"}
    by_field = {c.field: c for c in draft.changes}
    # Tighten: raise the confidence floor, lower the item cap.
    if "brief_min_confidence" in by_field:
        assert by_field["brief_min_confidence"].direction == "increase"
    if "brief_max_items" in by_field:
        assert by_field["brief_max_items"].direction == "decrease"


def test_a_candidate_cites_the_learning_events_that_caused_it():
    draft = proposals.propose(_learning_only_view(_evidence(executed=2, rejected=18)))
    assert draft is not None
    assert draft.supporting_learning_ids, "candidate asserts a change with no evidence"
    assert any(i.startswith("le-rej-") for i in draft.supporting_learning_ids)


def test_thin_evidence_produces_no_candidate():
    """Below Learning's own MIN_SAMPLES there is nothing to conclude."""
    from navis.learning.metrics import MIN_SAMPLES

    assert proposals.propose(
        _learning_only_view(_evidence(executed=1, rejected=MIN_SAMPLES - 2))
    ) is None


def test_a_good_acceptance_rate_loosens_nothing():
    """Asymmetric on purpose: absence of rejection is not consent."""
    assert proposals.propose(_learning_only_view(_evidence(executed=40, rejected=1))) is None


def test_missing_or_absent_evidence_produces_no_candidate():
    from navis.adaptive.contract import LearningEvidence

    assert proposals.propose(_learning_only_view(_evidence(executed=0, rejected=0))) is None
    empty = LearningEvidence(org_id=ORG_A, learning_version=9, as_of=AS_OF)
    assert proposals.propose(_learning_only_view(empty)) is None


def test_evidence_from_another_organization_is_refused():
    """Fail closed: a view carrying another org's evidence yields no candidate."""
    foreign = _evidence(executed=2, rejected=18, org_id=ORG_B)
    view = _learning_only_view(foreign)          # view.org_id is ORG_A
    assert proposals.propose(view) is None


def test_same_evidence_produces_the_same_candidate_identity():
    a = proposals.propose(_learning_only_view(_evidence(executed=2, rejected=18)))
    b = proposals.propose(_learning_only_view(_evidence(executed=2, rejected=18)))
    assert a is not None and b is not None
    assert a.proposal_id == b.proposal_id
    assert a.proposed_payload == b.proposed_payload
    assert [c.to_value for c in a.changes] == [c.to_value for c in b.changes]


def test_a_new_learning_version_is_a_new_candidate_identity():
    """Identity must move with the evidence, or a changed conclusion would be
    silently discarded by the append-only ON CONFLICT DO NOTHING."""
    a = proposals.propose(_learning_only_view(_evidence(executed=2, rejected=18, learning_version=20)))
    b = proposals.propose(_learning_only_view(_evidence(executed=2, rejected=18, learning_version=21)))
    assert a is not None and b is not None
    assert a.proposal_id != b.proposal_id


def test_conflicting_evidence_changes_nothing():
    """Intelligence says "trust earned, surface more"; Learning says the founder
    rejects most of it. Acting on either would be a guess, so neither fires."""
    view = make_view()                     # org insight with a GOOD track record
    baseline = proposals.propose(view)
    assert baseline is not None and baseline.changes

    conflicted = view.__class__(
        org_id=view.org_id,
        adaptive_version=view.adaptive_version,
        as_of=view.as_of,
        active_policy=view.active_policy,
        insights=view.insights,
        learning=_evidence(executed=2, rejected=18),
    )
    draft = proposals.propose(conflicted)
    contested = {"brief_min_confidence", "brief_max_items"}
    changed = {c.field for c in draft.changes} if draft else set()
    assert not (changed & contested), (
        "a contested field was changed on conflicting evidence"
    )


def test_a_learned_candidate_only_touches_governed_tuning_fields():
    """It can never reach authorization, capability or credential space."""
    from navis.adaptive.policy import POLICY_FIELDS, PolicyTarget

    draft = proposals.propose(_learning_only_view(_evidence(executed=2, rejected=18)))
    assert draft is not None
    for change in draft.changes:
        spec = POLICY_FIELDS[change.field]          # KeyError if invented
        assert spec.target in {PolicyTarget.REASONING, PolicyTarget.DECISION}
    assert set(draft.proposed_payload) == set(POLICY_FIELDS)


async def test_a_learned_candidate_stays_pending_until_approved():
    """Generation creates a PENDING proposal and activates nothing."""
    repo = FakeAdaptiveRepo(_learning_only_view(_evidence(executed=2, rejected=18)))
    result = await AdaptiveService(repo).propose(ORG_A)
    assert result is not None
    assert result["status"] == ProposalStatus.PENDING_APPROVAL.value
    assert await repo.get_active_snapshot(ORG_A) is None, "a candidate self-activated"


async def test_repeated_generation_is_idempotent():
    repo = FakeAdaptiveRepo(_learning_only_view(_evidence(executed=2, rejected=18)))
    svc = AdaptiveService(repo)
    await svc.propose(ORG_A)
    n_proposals, n_snapshots = len(repo.proposals), len(repo.snapshots)
    await svc.propose(ORG_A)
    await svc.propose(ORG_A)
    assert (len(repo.proposals), len(repo.snapshots)) == (n_proposals, n_snapshots)


# ---- 6.3.1.1: the RUNTIME path ----------------------------------------------


def test_resolved_policy_carries_the_version_that_produced_it():
    """REGRESSION. `ScopedView.policy_version` was unconditionally None.

    `scope.py` reads `policy["policy_version"]`, but `merge_defaults` drops every
    key outside POLICY_FIELDS — so the lookup could never succeed and no brief
    was attributable to a governing version. `resolve_policy` stamps it.
    """
    import inspect

    from navis.orchestration import runtime_policy

    src = inspect.getsource(runtime_policy.resolve_policy)
    assert '"policy_version"' in src, "resolve_policy stopped stamping the version"
    # And the vocabulary itself must stay clean — the stamp is NOT a policy field,
    # or it would leak into every stored proposal's payload.
    assert "policy_version" not in P.POLICY_FIELDS
    assert "policy_version" not in P.merge_defaults({"policy_version": 9})


def test_scope_reads_the_stamp_resolve_policy_writes():
    """The two halves must agree; they are in different packages."""
    from navis.state.v2.scope import MorningBrief, scope_for
    from navis.state.v2.models import OrganizationalState, Watermark

    state = OrganizationalState(org_id=ORG_A, as_of=AS_OF, watermark=Watermark())
    stamped = {**P.merge_defaults(None), "policy_version": 7}
    view = scope_for(state, MorningBrief(), stamped, as_of=AS_OF)
    assert view.policy_version == 7
    # No active policy => no version, and that stays distinguishable from 0.
    assert scope_for(state, MorningBrief(), {**P.merge_defaults(None), "policy_version": None},
                     as_of=AS_OF).policy_version is None


def test_a_malformed_snapshot_payload_degrades_instead_of_raising():
    """A JSONB column can hold a list or a string. `(payload or {}).items()`
    raised AttributeError, which would fail the brief rather than degrade it."""
    for junk in ([1, 2, 3], "brief_max_items=99", 42, True):
        assert P.merge_defaults(junk) == P.DEFAULT_POLICY


def test_non_finite_values_never_become_runtime_policy():
    """NaN and infinity survive float() and must be refused explicitly."""
    resolved = P.merge_defaults({
        "brief_min_confidence": float("nan"),
        "brief_max_items": float("inf"),
        "meeting_overload_count": float("-inf"),
    })
    assert resolved == P.DEFAULT_POLICY


def test_adaptive_version_moves_when_either_evidence_source_moves():
    """REGRESSION for the staleness window.

    `proposal_id` is uuid5(org, adaptive_version) and proposals append with
    ON CONFLICT DO NOTHING, so evidence that does not move this number is
    silently discarded. It must be the high-water mark of BOTH inputs.
    """
    import inspect

    from navis.adaptive.repository import AdaptiveRepository

    src = inspect.getsource(AdaptiveRepository.build_view)
    assert "max(" in src and "evidence.learning_version" in src, (
        "adaptive_version stopped tracking Learning; a newer rejection can now "
        "be lost to a stale proposal id"
    )


def test_a_lagging_intelligence_projection_cannot_hide_new_evidence():
    """Insights stuck at v20 while Learning reaches v21 must still re-key."""
    insight_versions = [20, 20]
    for learning_version, expected in ((20, 20), (21, 21), (0, 20)):
        assert max(max(insight_versions, default=0), learning_version) == expected


# ---- 6.3.2: the ACTIVATION path -----------------------------------------------


def test_one_active_policy_per_org_is_enforced_by_the_schema():
    """REGRESSION. `uq_ps_one_active` shipped in migration 0012 but was never
    declared on the model, so `create_all` built a schema WITHOUT it — the
    invariant was real in production and absent everywhere it could be tested.
    `get_active_snapshot` uses `db.scalar()`, which needs at-most-one row to be
    deterministic."""
    from navis.adaptive.models import PolicySnapshot

    partial = [
        i for i in PolicySnapshot.__table__.indexes
        if i.unique and i.dialect_options["postgresql"].get("where") is not None
    ]
    assert partial, "no partial unique index guards 'one ACTIVE per org'"
    idx = partial[0]
    assert [c.name for c in idx.columns] == ["organization_id"]
    assert "active" in str(idx.dialect_options["postgresql"]["where"])


async def test_approval_of_a_proposal_with_no_draft_does_not_retire_the_active_policy():
    """REGRESSION for a silent fail-open.

    `supersede_active` ran before anything checked there was a snapshot to
    activate, and `set_snapshot_status` no-ops on a missing row — so a proposal
    with a null `policy_id` retired the org's policy and activated nothing,
    dropping it to code defaults on what looked like a successful approval.
    """
    repo = FakeAdaptiveRepo(make_view())
    svc = AdaptiveService(repo)

    # An org running on a real active policy.
    await svc._activate_payload(
        ORG_A, {"brief_max_items": 4}, created_by="seed",
        approver_id=FOUNDER, role="founder", at=AT, kind="draft")
    before = await repo.get_active_payload(ORG_A)
    assert before["brief_max_items"] == 4

    prop = await svc.propose(ORG_A)
    repo.proposals[prop["proposal_id"]].policy_id = None      # dangling proposal

    with pytest.raises(ValueError):
        await svc.approve(ORG_A, prop["proposal_id"],
                          approver_id=FOUNDER, role="founder", at=AT)

    after = await repo.get_active_payload(ORG_A)
    assert after == before, "the active policy was retired by a failed approval"
    assert (await repo.get_active_snapshot(ORG_A)) is not None


async def test_a_stale_proposal_cannot_exceed_the_cap_from_the_current_baseline():
    """REGRESSION. The 25% magnitude cap is relative to the ACTIVE policy, and
    was only checked at propose() time. A rollback/kill switch between proposal
    and approval moves that baseline, so the stored change had to be re-validated
    on the path that actually activates — which `governance` documented itself as
    doing but never did."""
    from navis.adaptive.governance import PolicyViolation, validate_changes

    # Proposed when the floor was 0.45 (a legal +25% step to 0.5625)...
    changes = [{"field": "brief_min_confidence", "from": 0.45, "to": 0.5625}]
    validate_changes(changes, {"brief_min_confidence": 0.45})   # fine then

    # ...but a kill switch has since moved the baseline far below it.
    with pytest.raises(PolicyViolation):
        validate_changes(changes, {"brief_min_confidence": 0.1})


def test_a_malformed_stored_change_is_refused_not_skipped():
    """Fail closed: a record that cannot be shown safe must not activate."""
    from navis.adaptive.governance import PolicyViolation, validate_changes

    for junk in ([{"to": 5}], [{"field": "brief_max_items"}], ["not-a-dict"],
                 [{"field": "grant_admin", "from": 1, "to": 2}]):
        with pytest.raises(PolicyViolation):
            validate_changes(junk, P.DEFAULT_POLICY)


async def test_an_already_decided_proposal_cannot_be_approved_again():
    """Replay safety: approve twice, reject-then-approve — both refused."""
    repo = FakeAdaptiveRepo(make_view())
    svc = AdaptiveService(repo)
    prop = await svc.propose(ORG_A)
    pid = prop["proposal_id"]

    await svc.approve(ORG_A, pid, approver_id=FOUNDER, role="founder", at=AT)
    with pytest.raises(ValueError):                       # replayed approval
        await svc.approve(ORG_A, pid, approver_id=FOUNDER, role="founder", at=AT)

    repo2 = FakeAdaptiveRepo(make_view())
    svc2 = AdaptiveService(repo2)
    p2 = (await svc2.propose(ORG_A))["proposal_id"]
    await svc2.reject(ORG_A, p2, approver_id=FOUNDER, role="founder", at=AT)
    with pytest.raises(ValueError):                       # rejected then approved
        await svc2.approve(ORG_A, p2, approver_id=FOUNDER, role="founder", at=AT)
    assert await repo2.get_active_snapshot(ORG_A) is None


async def test_a_proposal_cannot_be_approved_from_another_organization():
    """The org check is on the persisted row, not on the caller's claim."""
    repo = FakeAdaptiveRepo(make_view(org_id=ORG_A))
    svc = AdaptiveService(repo)
    pid = (await svc.propose(ORG_A))["proposal_id"]

    with pytest.raises(LookupError):
        await svc.approve(ORG_B, pid, approver_id=FOUNDER, role="founder", at=AT)
    assert await repo.get_active_snapshot(ORG_A) is None


# ---- 6.3.3: Adaptive -> DECISION runtime ---------------------------------------


def test_only_the_two_live_fields_reach_a_runtime_consumer():
    """D. Dormant fields must stay explicitly dormant, not appear to work.

    `run_reasoning_v2(state, *, prior_state, evidence)` takes no policy, and the
    execution worker reads `MAX_EXECUTION_ATTEMPTS` directly — so seven of the
    nine governed fields are proposable and inert. This pins that split so a
    future change either wires a field AND updates this list, or is caught.
    """
    import inspect

    from navis.decisions.service import DecisionService
    from navis.reasoning.v2.service import run_reasoning_v2
    from navis.state.v2 import scope

    live = {"brief_max_items", "brief_min_confidence"}
    assert live <= set(P.POLICY_FIELDS)

    consumed = inspect.getsource(DecisionService.decide) + inspect.getsource(scope)
    for field in live:
        assert field in consumed, f"{field} lost its runtime consumer"

    dormant = set(P.POLICY_FIELDS) - live
    assert "policy" not in inspect.signature(run_reasoning_v2).parameters, (
        "Reasoning grew a policy seam; the six reasoning fields may no longer be dormant"
    )
    for field in dormant:
        assert field not in consumed, f"{field} became live without updating this test"


def test_a_brief_records_the_policy_version_that_governed_it():
    """E. A brief built under an approved policy was byte-indistinguishable from
    one built on code defaults."""
    import inspect

    from navis.decisions.models import DailyBrief
    from navis.decisions.service import DecisionService

    assert "policy_version" in DailyBrief.__table__.columns
    assert DailyBrief.__table__.columns["policy_version"].nullable, (
        "null must stay available: it means 'ran on code defaults'"
    )
    src = inspect.getsource(DecisionService.decide)
    assert 'policy.get("policy_version")' in src
    # Not part of the identity: a policy change replaces a generation's brief.
    from navis.decisions.base import compute_brief_id

    assert "policy" not in inspect.getsource(compute_brief_id)


def test_adaptive_never_reaches_intelligence():
    """The loop is Adaptive -> Decision and Adaptive -> Context. NOT Intelligence.

    Intelligence projects Learning and is forbidden from importing Adaptive
    (its AST invariant), so an Adaptive policy cannot change what the
    organization believes about itself — only what is surfaced from it.
    """
    import ast
    from pathlib import Path

    import navis.intelligence as pkg

    for py in Path(pkg.__file__).parent.rglob("*.py"):
        tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        for node in ast.walk(tree):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods = [node.module]
            assert not any(m.startswith("navis.adaptive") for m in mods), (
                f"{py.name} imports Adaptive; policy would steer Intelligence"
            )
