"""Unit tests for navis/context/models.py hardening: validation, canonicalization,
deterministic ordering/equality, and serialization round-trips.
"""
from __future__ import annotations

import pytest

from navis.context.models import BudgetReport, ContextScope, ScopeEntry, SelectionReason

_TR = SelectionReason.TASK_RELEVANT.value
_LE = SelectionReason.LINK_EXPANSION.value


# --- ScopeEntry validation --------------------------------------------------

def test_empty_entity_id_rejected():
    with pytest.raises(ValueError):
        ScopeEntry(entity_id="", entity_type="person", reason=_TR)


def test_empty_entity_type_rejected():
    with pytest.raises(ValueError):
        ScopeEntry(entity_id="e1", entity_type="", reason=_TR)


def test_invalid_selection_reason_rejected():
    with pytest.raises(ValueError):
        ScopeEntry(entity_id="e1", entity_type="person", reason="because_i_said_so")


def test_valid_scope_entry():
    e = ScopeEntry(entity_id="e1", entity_type="person", reason=_TR)
    assert e.reason == SelectionReason.TASK_RELEVANT.value  # str-enum compares to its value


# --- ContextScope: duplicates + canonical ordering --------------------------

def test_duplicate_entity_ids_rejected():
    with pytest.raises(ValueError):
        ContextScope(
            entity_types=("person",),
            entries=(ScopeEntry("e1", "person", _TR), ScopeEntry("e1", "company", _TR)),
        )


def test_entries_are_canonically_sorted():
    scope = ContextScope(
        entity_types=("person",),
        entries=(ScopeEntry("e3", "person", _TR), ScopeEntry("e1", "person", _TR), ScopeEntry("e2", "person", _LE)),
    )
    assert scope.entity_ids() == ("e1", "e2", "e3")  # sorted regardless of input order


def test_entity_types_are_canonical():
    scope = ContextScope(entity_types=("person", "company", "person"), entries=())
    assert scope.entity_types == ("company", "person")  # sorted + deduped


def test_deterministic_equality_independent_of_construction_order():
    a = ContextScope(("b", "a"), (ScopeEntry("e2", "person", _TR), ScopeEntry("e1", "company", _LE)))
    b = ContextScope(("a", "b"), (ScopeEntry("e1", "company", _LE), ScopeEntry("e2", "person", _TR)))
    assert a == b
    assert a.to_dict() == b.to_dict()


# --- BudgetReport validation ------------------------------------------------

@pytest.mark.parametrize("kwargs", [
    {"token_estimate": -1, "admitted": 0, "dropped": 0, "max_tokens": None},
    {"token_estimate": 0, "admitted": -1, "dropped": 0, "max_tokens": None},
    {"token_estimate": 0, "admitted": 0, "dropped": -5, "max_tokens": None},
    {"token_estimate": 0, "admitted": 0, "dropped": 0, "max_tokens": 0},
    {"token_estimate": 0, "admitted": 0, "dropped": 0, "max_tokens": -10},
    {"token_estimate": 100, "admitted": 1, "dropped": 0, "max_tokens": 10},  # estimate > budget
])
def test_invalid_budget_report_rejected(kwargs):
    with pytest.raises(ValueError):
        BudgetReport(**kwargs)


def test_valid_budget_reports():
    assert BudgetReport(0, 0, 0, None).max_tokens is None
    assert BudgetReport(8, 5, 3, 10).admitted == 5


# --- Serialization round-trips ----------------------------------------------

def test_scope_entry_round_trip():
    e = ScopeEntry("e1", "person", _LE)
    assert ScopeEntry.from_dict(e.to_dict()) == e


def test_context_scope_round_trip_is_lossless_and_deterministic():
    scope = ContextScope(
        ("person", "company"),
        (ScopeEntry("e2", "company", _LE), ScopeEntry("e1", "person", _TR)),
    )
    rt = ContextScope.from_dict(scope.to_dict())
    assert rt == scope and rt.to_dict() == scope.to_dict()


def test_budget_report_round_trip():
    r = BudgetReport(8, 5, 3, 10)
    assert BudgetReport.from_dict(r.to_dict()) == r
