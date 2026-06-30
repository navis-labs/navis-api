"""Regression tests for navis/context/ranking.py: fail-safe recency (no crash on bad
timestamps), structural type-centrality order, unknown-type/last, completeness, determinism.
"""
from __future__ import annotations

from datetime import datetime, timezone

from navis.context.models import ContextScope, ScopeEntry, SelectionReason
from navis.context.ranking import StructuralRanker, _TYPE_PRIORITY
from navis.memory.envelope import EntityType
from navis.state.v2.models import OrganizationalState, Watermark
from tests.state.fakes import ORG_A

_PERSON = EntityType.PERSON.value
_COMMIT = EntityType.COMMITMENT.value
_TR = SelectionReason.TASK_RELEVANT.value
_AS_OF = datetime(2026, 6, 2, tzinfo=timezone.utc)


def _raw(eid, *, last_seen=None, links=None):
    return {"entity_id": eid, "first_seen": None, "last_seen": last_seen,
            "fields": {}, "lists": {}, "links": links or {}, "provenance": {}}


def _state(buckets, *, as_of=_AS_OF):
    return OrganizationalState(org_id=ORG_A, as_of=as_of, watermark=Watermark(), entities=buckets)


def _scope(*entries: ScopeEntry):
    return ContextScope(tuple({e.entity_type for e in entries}), tuple(entries))


def _order(state, scope):
    return [e.entity_id for e in StructuralRanker().rank(state, scope)]


# --- Fail-safe recency (the headline fix: naive last_seen used to crash the sort) --------

def test_naive_last_seen_does_not_crash_and_ranks_last():
    state = _state({_PERSON: {
        "P1": _raw("P1", last_seen="2026-06-01T00:00:00+00:00"),  # aware, recent
        "P2": _raw("P2", last_seen="2026-06-01T12:00:00"),        # NAIVE (old code: TypeError)
    }})
    assert _order(state, _scope(ScopeEntry("P1", _PERSON, _TR), ScopeEntry("P2", _PERSON, _TR))) == ["P1", "P2"]


def test_missing_and_invalid_recency_rank_last():
    state = _state({_PERSON: {
        "GOOD": _raw("GOOD", last_seen="2026-06-01T00:00:00+00:00"),
        "MISSING": _raw("MISSING", last_seen=None),
        "INVALID": _raw("INVALID", last_seen="not-a-date"),
    }})
    order = _order(state, _scope(
        ScopeEntry("GOOD", _PERSON, _TR), ScopeEntry("MISSING", _PERSON, _TR), ScopeEntry("INVALID", _PERSON, _TR)))
    assert order[0] == "GOOD"  # dated entity first; the two +inf ones follow (id-tiebroken)
    assert set(order[1:]) == {"INVALID", "MISSING"}


def test_future_timestamp_ranks_as_freshest():
    state = _state({_PERSON: {
        "PAST": _raw("PAST", last_seen="2026-06-01T00:00:00+00:00"),
        "FUTURE": _raw("FUTURE", last_seen="2026-06-10T00:00:00+00:00"),  # after as_of (scheduled)
    }})
    assert _order(state, _scope(ScopeEntry("PAST", _PERSON, _TR), ScopeEntry("FUTURE", _PERSON, _TR))) == ["FUTURE", "PAST"]


# --- Structural type centrality ---------------------------------------------

def test_type_centrality_orders_person_before_commitment():
    state = _state({_PERSON: {"P1": _raw("P1")}, _COMMIT: {"C1": _raw("C1")}})
    order = StructuralRanker().rank(state, _scope(ScopeEntry("C1", _COMMIT, _TR), ScopeEntry("P1", _PERSON, _TR)))
    assert [e.entity_type for e in order][0] == _PERSON  # priority 0 < 5


def test_unknown_type_ranks_last():
    state = _state({_PERSON: {"P1": _raw("P1")}})
    order = _order(state, _scope(ScopeEntry("P1", _PERSON, _TR), ScopeEntry("X1", "mystery", _TR)))
    assert order == ["P1", "X1"]  # unknown type -> _UNKNOWN_TYPE -> after known types


# --- Completeness + determinism ---------------------------------------------

def test_priority_table_covers_every_entity_type():
    assert {e.value for e in EntityType} <= set(_TYPE_PRIORITY)


def test_ranking_is_deterministic_total_order():
    persons = {"P2": _raw("P2", last_seen="2026-06-01T00:00:00+00:00"),
               "P1": _raw("P1", last_seen="2026-06-01T00:00:00+00:00")}  # identical recency
    state = _state({_PERSON: persons})
    a = _order(state, _scope(ScopeEntry("P1", _PERSON, _TR), ScopeEntry("P2", _PERSON, _TR)))
    b = _order(state, _scope(ScopeEntry("P2", _PERSON, _TR), ScopeEntry("P1", _PERSON, _TR)))
    assert a == b == ["P1", "P2"]  # entity_id tiebreak → identical regardless of input order


# --- Freeze pass: injectable ordering provider + throw-proof key ------------

def test_injected_type_priority_overrides_default_ordering():
    state = _state({_PERSON: {"P1": _raw("P1")}, _COMMIT: {"C1": _raw("C1")}})
    ranker = StructuralRanker(type_priority={_COMMIT: 0, _PERSON: 1})  # deployment override
    order = ranker.rank(state, _scope(ScopeEntry("P1", _PERSON, _TR), ScopeEntry("C1", _COMMIT, _TR)))
    assert [e.entity_type for e in order][0] == _COMMIT  # injected ordering wins, deterministically


def test_injected_type_priority_validates_int_values():
    import pytest
    with pytest.raises(ValueError):
        StructuralRanker(type_priority={_PERSON: "high"})  # type: ignore[dict-item]
    with pytest.raises(ValueError):
        StructuralRanker(type_priority={_PERSON: True})  # bool is not an int priority


def test_malformed_links_do_not_crash_ranking():
    state = _state({_PERSON: {
        "P1": _raw("P1", links=["not", "a", "dict"]),  # corrupt: links is a list
        "P2": _raw("P2", links={"x": 5}),               # corrupt: non-sized link value
    }})
    order = _order(state, _scope(ScopeEntry("P1", _PERSON, _TR), ScopeEntry("P2", _PERSON, _TR)))
    assert set(order) == {"P1", "P2"}  # no crash; both ranked (degree treated as 0)
