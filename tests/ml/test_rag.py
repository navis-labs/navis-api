"""Regression tests for navis/ml/rag/grounding.py — the immutability fix on the ONLY exported
contract, plus the invariants that must hold for a grounding layer: evidence is immutable,
every rendered fact carries provenance, citations survive rendering, and rendering is
deterministic. GroundedContext is pure (no retrieval/provider/Memory dependency).
"""
from __future__ import annotations

import ast
import dataclasses
import os

import pytest

from navis.ml.rag import grounding as grounding_mod
from navis.ml.rag.grounding import GroundedContext, Grounding


def _ctx():
    return GroundedContext(
        org_id="o", question="Q",
        facts=[Grounding("memory", "m1", "Applied to YC"), Grounding("state", "s1", "Deal open")],
    )


# --- immutability (the fix) --------------------------------------------------

def test_facts_are_coerced_to_immutable_tuple():
    ctx = _ctx()
    assert isinstance(ctx.facts, tuple)          # list input coerced to a tuple
    with pytest.raises(AttributeError):
        ctx.facts.append(Grounding("x", "y", "z"))  # tuple has no append — evidence can't grow
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.facts = ()                            # frozen — can't reassign


def test_empty_default_is_immutable():
    ctx = GroundedContext(org_id="o", question="q")
    assert ctx.facts == ()  # no shared-mutable-default; each instance is independent


# --- provenance + citations survive rendering -------------------------------

def test_every_rendered_fact_carries_provenance():
    rendered = _ctx().render()
    assert "Question: Q" in rendered
    assert "[1] (memory:m1) Applied to YC" in rendered   # citation + (kind:source) + text
    assert "[2] (state:s1) Deal open" in rendered
    for line in rendered.splitlines():
        if line.startswith("["):
            assert "(" in line and ":" in line and ")" in line  # provenance on every cited line


def test_sources_preserve_lineage_in_order():
    assert _ctx().sources() == ["m1", "s1"]


# --- determinism -------------------------------------------------------------

def test_render_is_deterministic():
    facts = [Grounding("memory", "m1", "a"), Grounding("memory", "m2", "b")]
    a = GroundedContext("o", "q", facts).render()
    b = GroundedContext("o", "q", facts).render()
    assert a == b


# --- purity: grounding never imports retrieval / providers / Memory / State --

def test_grounding_module_is_pure():
    src = os.path.join(os.path.dirname(grounding_mod.__file__), "grounding.py")
    tree = ast.parse(open(src).read())
    navis_imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("navis")
    }
    assert navis_imports == set()  # pure: no retrieval / provider / Memory / State dependency
