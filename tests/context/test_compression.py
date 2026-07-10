"""Regression tests for navis/context/compression.py: greedy skip-and-continue, budget /
accounting / ordering invariants, fail-safe token estimation, determinism.
"""
from __future__ import annotations

from datetime import datetime, timezone

from navis.context.compression import GreedyCompressor
from navis.context.contracts import TokenBudget
from navis.context.models import ScopeEntry, SelectionReason
from navis.state.v2.models import OrganizationalState, Watermark
from tests.state.fakes import ORG_A

_P = "person"
_TR = SelectionReason.TASK_RELEVANT.value


def _raw(eid, *, fields=None):
    return {"entity_id": eid, "first_seen": None, "last_seen": None,
            "fields": fields or {}, "lists": {}, "links": {}, "provenance": {}}


def _big(eid):
    return _raw(eid, fields={"data": "x " * 80})  # ~80 word-tokens → large cost


def _state(persons):
    return OrganizationalState(org_id=ORG_A, as_of=datetime(2026, 6, 1, tzinfo=timezone.utc),
                              watermark=Watermark(), entities={_P: persons})


def _ranked(*ids):
    return [ScopeEntry(i, _P, _TR) for i in ids]


def _compress(persons, ranked, budget):
    return GreedyCompressor().compress(ranked, _state(persons), budget)


# --- Greedy skip-and-continue -----------------------------------------------

def test_skip_and_continue_admits_smaller_lower_ranked():
    # BIG ranked first but doesn't fit; SMALL ranked second fills the leftover budget.
    persons = {"BIG": _big("BIG"), "SMALL": _raw("SMALL")}
    admitted, dropped, _ = _compress(persons, _ranked("BIG", "SMALL"), TokenBudget(5))
    assert [e.entity_id for e in admitted] == ["SMALL"]
    assert [e.entity_id for e in dropped] == ["BIG"]


# --- Invariants --------------------------------------------------------------

def test_budget_invariant_used_within_budget():
    persons = {f"P{i}": _raw(f"P{i}", fields={"n": "w " * i}) for i in range(20)}
    for limit in (1, 5, 20, 100, 1000):
        _, _, report = _compress(persons, _ranked(*persons), TokenBudget(limit))
        assert report.token_estimate <= limit


def test_accounting_admitted_plus_dropped_equals_ranked():
    persons = {"A": _raw("A"), "B": _big("B"), "GHOST_NOT_IN_STATE": None}  # placeholder
    persons.pop("GHOST_NOT_IN_STATE")
    ranked = _ranked("A", "B", "GHOST")  # GHOST absent from State
    admitted, dropped, report = _compress(persons, ranked, TokenBudget(3))
    assert len(admitted) + len(dropped) == len(ranked)
    assert report.admitted == len(admitted) and report.dropped == len(dropped)


def test_ordering_is_preserved_subsequence():
    persons = {"A": _raw("A"), "B": _big("B"), "C": _raw("C")}
    admitted, _, _ = _compress(persons, _ranked("A", "B", "C"), TokenBudget(3))
    ids = [e.entity_id for e in admitted]
    assert ids == ["A", "C"]  # B dropped (too big); A,C stay in ranked order (no reorder)


# --- Unbounded fast path -----------------------------------------------------

def test_unbounded_admits_all_present_zero_estimate():
    persons = {"A": _big("A"), "B": _raw("B")}
    admitted, dropped, report = _compress(persons, _ranked("A", "B"), TokenBudget.unbounded())
    assert {e.entity_id for e in admitted} == {"A", "B"} and dropped == []
    assert report.token_estimate == 0  # fast path skips serialization


def test_missing_entity_is_dropped_unbounded_and_bounded():
    persons = {"A": _raw("A")}
    for budget in (TokenBudget.unbounded(), TokenBudget(100)):
        admitted, dropped, _ = _compress(persons, _ranked("A", "GHOST"), budget)
        assert [e.entity_id for e in admitted] == ["A"]
        assert [e.entity_id for e in dropped] == ["GHOST"]


# --- Fail-safe token estimation (never crash on malformed data) -------------

def test_malformed_entity_is_sized_not_crashed():
    # Budget estimation is now fail-safe (budget.py): mixed-type field keys are canonicalized
    # deterministically and SIZED rather than throwing — the entity is budgeted like any other
    # (no crash, no special-case drop). Corruption is a State data-quality concern, not budget's.
    persons = {"BAD": _raw("BAD", fields={1: "a", "b": "c"}), "OK": _raw("OK")}
    admitted, dropped, report = _compress(persons, _ranked("BAD", "OK"), TokenBudget(100))
    assert {e.entity_id for e in admitted} == {"BAD", "OK"}  # both fit; neither crashes
    assert dropped == [] and report.admitted == 2


# --- Determinism -------------------------------------------------------------

def test_compression_is_deterministic():
    persons = {"A": _raw("A", fields={"n": "w w"}), "B": _big("B"), "C": _raw("C")}
    a = _compress(persons, _ranked("A", "B", "C"), TokenBudget(4))
    b = _compress(persons, _ranked("A", "B", "C"), TokenBudget(4))
    assert ([e.entity_id for e in a[0]], [e.entity_id for e in a[1]], a[2].to_dict()) == \
           ([e.entity_id for e in b[0]], [e.entity_id for e in b[1]], b[2].to_dict())
