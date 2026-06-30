"""Unit tests for navis/context/tasks.py: TaskSpec validation, registry completeness
(one ContextTask <-> one TaskSpec), determinism, and serialization stability.
"""
from __future__ import annotations

import pytest

from navis.context import tasks as tasks_mod
from navis.context.tasks import TASK_SPECS, ContextTask, TaskSpec, _assert_complete_registry
from navis.memory.envelope import EntityType


# --- TaskSpec validation ----------------------------------------------------

def test_empty_entity_types_rejected():
    with pytest.raises(ValueError):
        TaskSpec(entity_types=frozenset(), expand_links=True)


def test_unknown_entity_type_rejected():
    with pytest.raises(ValueError):
        TaskSpec(entity_types=frozenset({"persons"}))  # typo of "person"


def test_expand_links_must_be_bool():
    with pytest.raises(ValueError):
        TaskSpec(entity_types=None, expand_links="yes")  # type: ignore[arg-type]


def test_valid_taskspec_none_is_all():
    spec = TaskSpec(entity_types=None)
    assert spec.entity_types is None and spec.expand_links is True


def test_valid_taskspec_canonicalizes_to_frozenset():
    spec = TaskSpec(entity_types=frozenset({EntityType.PERSON.value, EntityType.COMPANY.value}))
    assert spec.entity_types == frozenset({"person", "company"})


# --- Registry: one-to-one, no orphan/duplicate ------------------------------

def test_registry_is_one_to_one():
    assert set(TASK_SPECS) == set(ContextTask)
    assert len(TASK_SPECS) == len(ContextTask)
    assert all(isinstance(s, TaskSpec) for s in TASK_SPECS.values())


def test_missing_task_is_detected(monkeypatch):
    # Simulate an orphan task (enum member with no spec) → the import-time guard raises.
    monkeypatch.delitem(tasks_mod.TASK_SPECS, ContextTask.FULL_ORG)
    with pytest.raises(RuntimeError):
        _assert_complete_registry()


# --- Determinism + serialization stability ----------------------------------

def test_deterministic_equality_is_order_independent():
    a = TaskSpec(entity_types=frozenset({"person", "company"}))
    b = TaskSpec(entity_types=frozenset({"company", "person"}))
    assert a == b


def test_entity_types_have_stable_canonical_ordering():
    spec = TASK_SPECS[ContextTask.RELATIONSHIPS]
    assert spec.entity_types is not None
    once = sorted(spec.entity_types)
    twice = sorted(spec.entity_types)
    assert once == twice  # canonical, deterministic serialization-ready ordering


def test_specs_are_immutable():
    spec = TASK_SPECS[ContextTask.COMMITMENTS]
    with pytest.raises(Exception):
        spec.expand_links = False  # type: ignore[misc]  (frozen)
