"""Decision V2: determinism, replay, provenance, categories, tenancy, no-LLM/clock."""
from __future__ import annotations

import ast
import pathlib

from navis.decisions.base import DecisionCategory, compute_brief_id, compute_decision_id
from navis.decisions.what_matters_today import WhatMattersTodayEngine
from navis.memory.envelope import EntityType
from navis.reasoning.v2 import REASONING_VERSION, run_reasoning_v2
from navis.state.v2.builders import build_projection
from tests.state.fakes import ORG_A, ORG_B, corpus, rec


def _signals(org=ORG_A):
    recs = corpus(org) + [
        rec(EntityType.PERSON, "hb@acme.com", {"email": "hb@acme.com", "role": "from"},
            offset=40 * 86400, tag="hb", org_id=org)
    ]
    state = build_projection(org, recs)
    return run_reasoning_v2(state), state


def _decide(signals, state, org=ORG_A):
    wm = str(state.watermark.last_memory_id)
    return WhatMattersTodayEngine().decide(
        signals, org_id=org, reasoning_version=REASONING_VERSION,
        state_watermark=wm, as_of=state.as_of,
    )


def test_same_signals_produce_identical_decisions():
    signals, state = _signals()
    a = [d.to_dict() for d in _decide(signals, state)]
    b = [d.to_dict() for d in _decide(signals, state)]
    assert a == b  # deterministic incl decision_id + created_at → replay-safe


def test_decision_id_is_deterministic_and_traceable():
    signals, state = _signals()
    decisions = _decide(signals, state)
    wm = str(state.watermark.last_memory_id)
    for d in decisions:
        sig_id = d.reasoning_signal_ids[0]
        import uuid
        assert d.decision_id == compute_decision_id(ORG_A, REASONING_VERSION, wm, uuid.UUID(sig_id))


def test_full_contract_on_every_decision():
    signals, state = _signals()
    decisions = _decide(signals, state)
    assert decisions
    for d in decisions:
        assert d.org_id == ORG_A
        assert d.decision_id and d.title and d.summary and d.why
        assert d.priority_rank >= 1
        assert d.reasoning_signal_ids and d.reasoning_version == REASONING_VERSION
        assert d.state_watermark == str(state.watermark.last_memory_id)
        assert d.created_at == state.as_of  # logical time, never now()
        assert isinstance(d.category, DecisionCategory)


def test_priority_ranks_are_dense_and_ordered():
    signals, state = _signals()
    decisions = _decide(signals, state)
    assert [d.priority_rank for d in decisions] == list(range(1, len(decisions) + 1))


def test_categories_map_from_signal_types():
    signals, state = _signals()
    by_cat = {d.category for d in _decide(signals, state)}
    # The stale corpus yields risk decisions.
    assert DecisionCategory.RISK in by_cat
    # PRIORITY / OPPORTUNITY / DELEGATION are no longer reachable.
    #
    # This assertion used to read `DecisionCategory.PRIORITY in by_cat`,
    # which pinned the bug: the only signal types that mapped to those
    # categories (CONTEXT_SUMMARY, PRIORITY_LOAD, ORG_DRIFT, OPPORTUNITY,
    # LEVERAGE_POINT) are interpretation, not decision points, and were
    # being promoted into founders' briefs as "Organizational context"
    # and "Active inbound engagement". See tests/decisions/
    # test_decision_semantics.py for the full boundary.
    assert DecisionCategory.PRIORITY not in by_cat
    assert DecisionCategory.OPPORTUNITY not in by_cat
    assert DecisionCategory.DELEGATION not in by_cat


def test_multi_tenant_isolation():
    sa, st_a = _signals(ORG_A)
    sb, st_b = _signals(ORG_B)
    da = _decide(sa, st_a, ORG_A)
    db = _decide(sb, st_b, ORG_B)
    assert all(d.org_id == ORG_A for d in da)
    assert all(d.org_id == ORG_B for d in db)
    assert {d.decision_id for d in da}.isdisjoint({d.decision_id for d in db})


def test_brief_id_is_deterministic():
    a = compute_brief_id(ORG_A, REASONING_VERSION, "wm-1")
    b = compute_brief_id(ORG_A, REASONING_VERSION, "wm-1")
    c = compute_brief_id(ORG_A, REASONING_VERSION, "wm-2")
    assert a == b and a != c


def test_min_confidence_filters_and_max_items_caps():
    signals, state = _signals()
    engine = WhatMattersTodayEngine(max_items=2, min_confidence=0.99)
    out = engine.decide(signals, org_id=ORG_A, reasoning_version=REASONING_VERSION,
                        state_watermark="w", as_of=state.as_of)
    assert len(out) <= 2  # capped
    # Only very-high-confidence signals (e.g. context) survive the 0.99 filter.


_FORBIDDEN_CALLS = {"now", "uuid4"}


def test_decision_domain_has_no_clock_or_uuid4():
    """No Python wall-clock / uuid4 in decision LOGIC. SQLAlchemy `func.now()` for
    the cache-row `created_at` is allowed (server-side metadata, not decision content)."""
    root = pathlib.Path(__file__).resolve().parent.parent.parent / "navis" / "decisions"
    for py in root.rglob("*.py"):
        tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr not in _FORBIDDEN_CALLS:
                    continue
                # Allow SQLAlchemy func.now() (server-side cache timestamp).
                if node.func.attr == "now" and isinstance(node.func.value, ast.Name) and node.func.value.id == "func":
                    continue
                raise AssertionError(f"{py.name} calls forbidden {node.func.attr}()")
