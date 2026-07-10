"""Regression tests for navis/context/selectors.py hardening: one-hop (non-recursive)
expansion, dangling/malformed-link safety, no-duplicate, FULL_ORG fast path, and the
O(scope)-not-O(org) scalability invariant.
"""
from __future__ import annotations

from datetime import datetime, timezone

from navis.context.selectors import DeterministicSelector
from navis.context.tasks import TaskSpec
from navis.memory.envelope import EntityType
from navis.state.v2.models import OrganizationalState, Watermark
from tests.state.fakes import ORG_A

_P = EntityType.PERSON.value
_R = EntityType.RELATIONSHIP.value
_C = EntityType.COMMITMENT.value


def _raw(eid, *, links=None):
    return {"entity_id": eid, "first_seen": None, "last_seen": None,
            "fields": {}, "lists": {}, "links": links or {},
            "provenance": {"memory_ids": [], "signal_ids": []}}


def _state(buckets):
    return OrganizationalState(
        org_id=ORG_A, as_of=datetime(2026, 6, 1, tzinfo=timezone.utc),
        watermark=Watermark(), entities=buckets,
    )


def _select(buckets, types, *, expand=True):
    spec = TaskSpec(entity_types=(None if types is None else frozenset(types)), expand_links=expand)
    scope = DeterministicSelector().select(_state(buckets), spec)
    return {e.entity_id: e.reason for e in scope.entries}


# --- One-hop, non-recursive -------------------------------------------------

def test_expansion_is_one_hop_not_recursive():
    buckets = {
        _P: {"P1": _raw("P1", links={"relationships": ["R1"]})},
        _R: {"R1": _raw("R1", links={"commitments": ["C1"]})},
        _C: {"C1": _raw("C1")},
    }
    result = _select(buckets, {_P})
    assert result == {"P1": "task_relevant", "R1": "link_expansion"}  # C1 NOT followed


# --- Dangling + malformed links handled safely ------------------------------

def test_dangling_link_is_skipped_not_fabricated():
    buckets = {_C: {"C1": _raw("C1", links={"owner": ["GHOST"]})}}
    assert _select(buckets, {_C}) == {"C1": "task_relevant"}  # GHOST never appears


def test_malformed_link_targets_are_skipped():
    buckets = {
        _C: {"C1": _raw("C1", links={"owner": ["", None, 123, "P1"]})},
        _P: {"P1": _raw("P1")},
    }
    assert _select(buckets, {_C}) == {"C1": "task_relevant", "P1": "link_expansion"}


# --- No duplicate when a link target is already selected --------------------

def test_no_duplicate_when_target_already_selected():
    buckets = {
        _P: {"P1": _raw("P1", links={"relationships": ["R1"]})},
        _R: {"R1": _raw("R1")},
    }
    result = _select(buckets, {_P, _R})
    assert result == {"P1": "task_relevant", "R1": "task_relevant"}  # R1 not re-tagged


# --- FULL_ORG: every entity directly selected, expansion adds nothing --------

def test_full_org_is_all_task_relevant():
    buckets = {
        _P: {"P1": _raw("P1", links={"relationships": ["R1"]})},
        _R: {"R1": _raw("R1")},
    }
    result = _select(buckets, None)
    assert result == {"P1": "task_relevant", "R1": "task_relevant"}


# --- Scalability: scoped selection is O(scope), not O(org) -------------------

def test_scoped_selection_does_not_pull_whole_org():
    persons = {f"P{i}": _raw(f"P{i}") for i in range(50_000)}
    buckets = {
        _P: persons,
        _C: {"C1": _raw("C1", links={"owner": ["P0"]})},
    }
    # COMMITMENT scope + one-hop pulls ONLY the one linked person, never all 50k.
    result = _select(buckets, {_C})
    assert result == {"C1": "task_relevant", "P0": "link_expansion"}


# --- Determinism ------------------------------------------------------------

def test_selection_is_deterministic_under_input_reordering():
    a = {_C: {"C1": _raw("C1", links={"owner": ["P1"]})}, _P: {"P1": _raw("P1"), "P2": _raw("P2")}}
    b = {_P: {"P2": _raw("P2"), "P1": _raw("P1")}, _C: {"C1": _raw("C1", links={"owner": ["P1"]})}}
    s1 = DeterministicSelector().select(_state(a), TaskSpec(frozenset({_C}), True))
    s2 = DeterministicSelector().select(_state(b), TaskSpec(frozenset({_C}), True))
    assert s1.to_dict() == s2.to_dict()
