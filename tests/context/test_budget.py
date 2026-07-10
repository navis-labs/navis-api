"""Regression tests for navis/context/budget.py: deterministic, fail-safe, provider-
independent token estimation. Canonicalization must NEVER crash and must be replay-stable.
"""
from __future__ import annotations

import datetime
import decimal
import inspect
import uuid

from navis.context.budget import _canonical, entity_tokens, estimate_tokens


def _raw(fields=None, lists=None, links=None):
    return {"fields": fields or {}, "lists": lists or {}, "links": links or {}}


# --- Determinism / replay ---------------------------------------------------

def test_estimate_is_deterministic_and_min_one():
    assert estimate_tokens("") == 1
    assert estimate_tokens("a b c") == estimate_tokens("a b c")


def test_canonical_is_deterministic_for_valid_entity():
    raw = _raw(fields={"email": "a@b.com", "name": "Alice Smith", "role": "from"})
    assert _canonical(raw) == _canonical(raw)
    assert entity_tokens(raw) == entity_tokens(raw)


def test_canonical_key_order_is_stable():
    a = _raw(fields={"b": 1, "a": 2, "c": 3})
    b = _raw(fields={"c": 3, "a": 2, "b": 1})  # different insertion order
    assert _canonical(a) == _canonical(b)  # sort_keys → identical bytes


# --- Canonicalization can NEVER crash (and stays deterministic) -------------

def test_mixed_type_keys_do_not_crash():
    raw = _raw(fields={1: "a", "b": "c", 2.5: "d"})  # would raise sort_keys TypeError
    assert entity_tokens(raw) == entity_tokens(raw) >= 1  # deterministic, no crash


def test_set_value_is_deterministic_sorted():
    raw = _raw(fields={"tags": {3, 1, 2}})
    # set iteration order is hash-seed dependent; canonicalization sorts → stable bytes.
    assert _canonical(raw) == _canonical(_raw(fields={"tags": {2, 3, 1}}))
    assert "[1,2,3]" in _canonical(raw)


def test_exotic_value_types_do_not_crash():
    for value in (
        datetime.datetime(2026, 6, 1, 12, 0),
        uuid.UUID(int=7),
        decimal.Decimal("1.5"),
        b"\x00\x01\x02",
        object(),
    ):
        raw = _raw(fields={"v": value})
        assert entity_tokens(raw) == entity_tokens(raw) >= 1  # deterministic, no crash


def test_circular_reference_does_not_recurse_forever():
    circ: dict = {}
    circ["self"] = circ
    assert entity_tokens(_raw(fields=circ)) >= 1  # depth-bounded, no RecursionError


def test_malformed_raw_is_safe():
    # raw missing keys / non-dict members must not crash.
    assert entity_tokens({}) >= 1
    assert entity_tokens({"fields": None, "lists": None, "links": None}) >= 1


# --- Provider independence (NOT a vendor tokenizer) -------------------------

def test_no_provider_or_tokenizer_dependencies():
    # Check actual IMPORTS (via AST), not prose — the module is a pure stdlib transform.
    import ast
    import navis.context.budget as budget_mod
    tree = ast.parse(inspect.getsource(budget_mod))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    assert roots <= {"__future__", "json", "re", "typing"}, f"unexpected imports: {roots}"
