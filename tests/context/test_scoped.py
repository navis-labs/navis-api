"""Phase 3: scoped tasks — link expansion, deep-copy isolation, budget invariants,
concurrency determinism.
"""
from __future__ import annotations

import asyncio

from navis.context import (
    ActorRef,
    ContextRequest,
    ContextService,
    ContextTask,
    PermissionScope,
    TokenBudget,
)
from navis.context.models import SelectionReason
from navis.context.selectors import DeterministicSelector
from navis.context.tasks import TaskSpec
from navis.memory.envelope import EntityType
from navis.state.v2.builders import build_projection
from tests.context.fakes import FakeStateReader
from tests.state.fakes import ORG_A, corpus


def _state():
    return build_projection(ORG_A, corpus())


def _svc(state=None):
    return ContextService(state_service=FakeStateReader(state or _state()))


# --- Link expansion (selector unit) -----------------------------------------

def test_link_expansion_pulls_in_neighbours():
    state = _state()
    spec = TaskSpec(entity_types=frozenset({EntityType.COMMITMENT.value}), expand_links=True)
    scope = DeterministicSelector().select(state, spec)
    types = {e.entity_type for e in scope.entries}
    assert EntityType.COMMITMENT.value in types
    # the commitment's owner (a person) is reached purely via State's link set
    person_entries = [e for e in scope.entries if e.entity_type == EntityType.PERSON.value]
    assert person_entries and any(
        e.reason == SelectionReason.LINK_EXPANSION.value for e in person_entries
    )


def test_no_expansion_keeps_only_task_types():
    state = _state()
    spec = TaskSpec(entity_types=frozenset({EntityType.COMMITMENT.value}), expand_links=False)
    scope = DeterministicSelector().select(state, spec)
    assert {e.entity_type for e in scope.entries} == {EntityType.COMMITMENT.value}


def test_selection_is_deterministic():
    state = _state()
    spec = TaskSpec(entity_types=frozenset({EntityType.COMMITMENT.value}), expand_links=True)
    a = DeterministicSelector().select(state, spec)
    b = DeterministicSelector().select(state, spec)
    assert a.to_dict() == b.to_dict()


# --- Scoped state is an isolated deep copy ----------------------------------

async def test_scoped_state_is_isolated_from_source():
    state = _state()
    before = state.to_dict()
    bundle = await _svc(state).build(
        ContextRequest(org_id=ORG_A, task=ContextTask.COMMITMENTS,
                       actor=ActorRef.system(), permissions=PermissionScope.system())
    )
    # The COMMITMENTS scope is a strict subset → the bundle owns a deep copy.
    assert bundle.state.to_dict() != before
    for bucket in bundle.state.entities.values():
        for raw in bucket.values():
            raw["fields"]["__mutated__"] = True
    assert state.to_dict() == before  # source projection untouched


# --- Budget invariants (property-style) -------------------------------------

async def test_budget_invariant_holds_across_limits():
    svc = _svc()
    total = sum(len(b) for b in _state().entities.values())
    for limit in (1, 3, 7, 15, 40, 100, 10_000):
        bundle = await svc.build(
            ContextRequest(org_id=ORG_A, task=ContextTask.FULL_ORG,
                           actor=ActorRef.system(), permissions=PermissionScope.system(),
                           budget=TokenBudget(max_tokens=limit))
        )
        r = bundle.budget_report
        assert r.token_estimate <= limit                 # never exceeds the budget
        assert r.admitted + r.dropped == total            # every entity accounted for
        assert len(bundle.scope.entries) == r.admitted    # scope == admitted


# --- Concurrency determinism ------------------------------------------------

async def test_concurrent_builds_are_identical():
    svc = _svc()
    results = await asyncio.gather(
        *[svc.build(ContextRequest.full_org(ORG_A)) for _ in range(25)]
    )
    assert len({r.context_id for r in results}) == 1
    first = results[0].to_dict()
    assert all(r.to_dict() == first for r in results)
