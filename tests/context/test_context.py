"""Context layer — Phase 0 invariants.

Covers: FULL_ORG parity, determinism/replay, lossless serialization, no-reasoning-keys,
RBAC default-deny + ABAC, token budgeting, scoped link pruning, multi-tenant isolation,
cache hit, provenance preservation, unknown-task failure.
"""
from __future__ import annotations

import uuid

import pytest

from navis.context import (
    ActorRef,
    ContextRequest,
    ContextService,
    ContextTask,
    PermissionScope,
    TokenBudget,
)
from navis.context.contracts import ContextBundle, compute_context_id
from navis.context.exceptions import UnknownContextTask
from navis.context.models import ContextScope
from navis.memory.envelope import EntityType
from navis.state.v2.builders import build_projection
from tests.context.fakes import FakeStateReader, MemoryContextCache
from tests.state.fakes import ORG_A, ORG_B, corpus


def test_public_surface_is_importable_and_matches_all():
    """The package __all__ is the public contract: every name resolves, and a caller can
    catch the whole layer via the exported ContextError (build() raises its subclasses)."""
    import navis.context as ctx

    assert set(ctx.__all__) == {
        "ContextService", "ContextRequest", "ContextTask", "ContextBundle",
        "ActorRef", "PermissionScope", "TokenBudget", "ContextError",
    }
    for name in ctx.__all__:
        assert getattr(ctx, name) is not None  # no dangling/broken re-export
    assert issubclass(UnknownContextTask, ctx.ContextError)  # the layer is catchable as one type


def _svc(*states, cache=None):
    return ContextService(state_service=FakeStateReader(*states), cache=cache)


def _full_state():
    return build_projection(ORG_A, corpus())


# --- FULL_ORG parity (the migration guarantee) ------------------------------

async def test_full_org_bundle_state_equals_state():
    state = _full_state()
    bundle = await _svc(state).build(ContextRequest.full_org(ORG_A))
    assert bundle.state.to_dict() == state.to_dict()          # byte-identical scoped state
    assert bundle.budget_report.dropped == 0                  # unbounded → nothing dropped


# --- Determinism / replay ---------------------------------------------------

async def test_context_id_and_bundle_are_deterministic():
    state = _full_state()
    svc = _svc(state)
    a = await svc.build(ContextRequest.full_org(ORG_A))
    b = await svc.build(ContextRequest.full_org(ORG_A))
    assert a.context_id == b.context_id
    assert a.to_dict() == b.to_dict()                          # replay → identical bytes


async def test_context_id_changes_with_state_watermark():
    s1 = build_projection(ORG_A, corpus())
    s2 = build_projection(ORG_A, corpus()[:5])                 # different watermark
    req = ContextRequest.full_org(ORG_A)
    assert compute_context_id(req, s1) != compute_context_id(req, s2)


async def test_bundle_serialization_is_lossless():
    bundle = await _svc(_full_state()).build(ContextRequest.full_org(ORG_A))
    assert ContextBundle.from_dict(bundle.to_dict()).to_dict() == bundle.to_dict()


# --- Context computes NO intelligence ---------------------------------------

_FORBIDDEN = {
    "risk", "score", "scores", "priority", "urgency", "health", "recommendation",
    "recommendations", "prediction", "predictions", "sentiment", "probability", "ranking_score",
}


def _all_keys(obj):
    keys = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            keys.add(str(k).lower())
            keys |= _all_keys(v)
    elif isinstance(obj, list):
        for v in obj:
            keys |= _all_keys(v)
    return keys


async def test_bundle_has_no_reasoning_keys():
    bundle = await _svc(_full_state()).build(ContextRequest.full_org(ORG_A))
    assert not (_all_keys(bundle.to_dict()) & _FORBIDDEN)


# --- RBAC / ABAC ------------------------------------------------------------

def _scoped_request(allowed_types=(), predicates=()):
    return ContextRequest(
        org_id=ORG_A, task=ContextTask.FULL_ORG,
        actor=ActorRef(actor_id=uuid.uuid4(), kind="user", roles=("viewer",)),
        permissions=PermissionScope(
            allow_all=False, allowed_entity_types=tuple(allowed_types),
            attribute_predicates=tuple(predicates),
        ),
    )


async def test_rbac_default_deny_yields_empty_bundle():
    bundle = await _svc(_full_state()).build(_scoped_request(allowed_types=()))
    assert bundle.scope.entries == ()
    assert bundle.state.counts() == {et.value: 0 for et in EntityType}


async def test_rbac_admits_only_permitted_types():
    bundle = await _svc(_full_state()).build(_scoped_request(allowed_types=(EntityType.PERSON.value,)))
    types = {e.entity_type for e in bundle.scope.entries}
    assert types == {EntityType.PERSON.value}
    assert bundle.state.counts()[EntityType.PERSON.value] >= 1
    assert bundle.state.counts()[EntityType.COMMITMENT.value] == 0


async def test_abac_attribute_predicate_filters():
    # Only people whose role == "from" survive.
    bundle = await _svc(_full_state()).build(
        _scoped_request(allowed_types=(EntityType.PERSON.value,), predicates=(("role", "from"),))
    )
    for p in bundle.state.people():
        assert p.role == "from"
    assert bundle.state.counts()[EntityType.PERSON.value] >= 1


# --- Token budgeting --------------------------------------------------------

async def test_budget_drops_tail_and_stays_within_limit():
    state = _full_state()
    bundle = await ContextService(state_service=FakeStateReader(state)).build(
        ContextRequest(
            org_id=ORG_A, task=ContextTask.FULL_ORG,
            actor=ActorRef.system(), permissions=PermissionScope.system(),
            budget=TokenBudget(max_tokens=10),
        )
    )
    assert bundle.budget_report.dropped > 0
    assert bundle.budget_report.admitted < 8
    assert bundle.budget_report.token_estimate <= 10


async def test_unbounded_budget_admits_everything():
    bundle = await _svc(_full_state()).build(ContextRequest.full_org(ORG_A))
    assert bundle.budget_report.admitted == 8 and bundle.budget_report.dropped == 0


# --- Scoped task link pruning ----------------------------------------------

async def test_scoped_task_prunes_dangling_links():
    bundle = await _svc(_full_state()).build(
        ContextRequest(
            org_id=ORG_A, task=ContextTask.COMMITMENTS,
            actor=ActorRef.system(), permissions=PermissionScope.system(),
        )
    )
    admitted = set(bundle.scope.entity_ids())
    for bucket in bundle.state.entities.values():
        for raw in bucket.values():
            for targets in (raw.get("links") or {}).values():
                assert all(t in admitted for t in targets)  # no link points outside the scope


# --- Multi-tenant isolation -------------------------------------------------

async def test_multi_tenant_isolation():
    sa = build_projection(ORG_A, corpus(ORG_A))
    sb = build_projection(ORG_B, corpus(ORG_B))
    svc = _svc(sa, sb)
    ba = await svc.build(ContextRequest.full_org(ORG_A))
    bb = await svc.build(ContextRequest.full_org(ORG_B))
    assert ba.org_id != bb.org_id
    assert set(ba.scope.entity_ids()).isdisjoint(set(bb.scope.entity_ids()))
    assert ba.context_id != bb.context_id


# --- Cache ------------------------------------------------------------------

def _scoped_cacheable_request():
    # A scoped task is a strict subset of the org → cacheable (worth persisting).
    return ContextRequest(org_id=ORG_A, task=ContextTask.COMMITMENTS,
                          actor=ActorRef.system(), permissions=PermissionScope.system())


async def test_cache_hit_returns_stored_bundle():
    cache = MemoryContextCache()
    svc = _svc(_full_state(), cache=cache)
    first = await svc.build(_scoped_cacheable_request())
    second = await svc.build(_scoped_cacheable_request())
    assert cache.puts == 1 and cache.hits == 1
    assert second.to_dict() == first.to_dict()


async def test_full_org_bundle_is_not_persisted(cache=None):
    # A full-org bundle embeds a whole copy of State → never worth caching (B8).
    cache = MemoryContextCache()
    svc = _svc(_full_state(), cache=cache)
    await svc.build(ContextRequest.full_org(ORG_A))
    assert cache.puts == 0


# --- Provenance preserved ---------------------------------------------------

async def test_provenance_preserved_for_admitted_entities():
    bundle = await _svc(_full_state()).build(ContextRequest.full_org(ORG_A))
    assert set(bundle.evidence) == set(bundle.scope.entity_ids())
    for ev in bundle.evidence.values():
        assert "memory_ids" in ev and "signal_ids" in ev


# --- Unknown task fails closed ---------------------------------------------

async def test_unknown_task_raises(monkeypatch):
    from navis.context import tasks as tmod
    monkeypatch.delitem(tmod.TASK_SPECS, ContextTask.FULL_ORG)
    with pytest.raises(UnknownContextTask):
        await _svc(_full_state()).build(ContextRequest.full_org(ORG_A))
