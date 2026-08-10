"""Reasoning V2: determinism, provenance, tenancy, no-LLM/no-clock, coverage."""
from __future__ import annotations

import pathlib

from navis.reasoning.v2 import run_reasoning_v2
from navis.reasoning.v2.contract import REASONING_VERSION, SignalType
from navis.reasoning.v2.engines import ENGINES
from navis.state.v2.builders import build_projection
from navis.state.v2.models import OrganizationalState
from navis.memory.envelope import EntityType
from tests.state.fakes import ORG_A, ORG_B, corpus, rec


def _state(org=ORG_A, *, stale: bool = True) -> OrganizationalState:
    records = corpus(org)
    if stale:
        # A late "heartbeat" pushes as_of ~40 days ahead so risk engines fire.
        records = records + [
            rec(EntityType.PERSON, "heartbeat@acme.com",
                {"email": "heartbeat@acme.com", "name": "HB", "role": "from"},
                offset=40 * 86400, tag="heartbeat", org_id=org)
        ]
    return build_projection(org, records)


def test_same_state_produces_identical_signals():
    state = _state()
    a = [s.to_dict() for s in run_reasoning_v2(state)]
    b = [s.to_dict() for s in run_reasoning_v2(state)]
    assert a == b  # byte-identical incl signal_id + created_at → replay-safe


def test_created_at_is_logical_time_not_wallclock():
    state = _state()
    for s in run_reasoning_v2(state):
        assert s.created_at == state.as_of  # deterministic logical clock, never now()


def test_every_signal_is_fully_provenanced():
    state = _state()
    signals = run_reasoning_v2(state)
    assert signals
    wm = str(state.watermark.last_memory_id)
    for s in signals:
        assert s.org_id == state.org_id
        assert s.signal_id is not None
        assert s.engine_name and s.engine_version
        assert s.state_as_of == state.as_of
        assert s.state_watermark == wm
        assert s.reasoning_version == REASONING_VERSION
        assert s.why and s.title
        assert 0.0 <= s.confidence <= 1.0 and 0.0 <= s.severity <= 1.0


def test_org_id_propagation_and_tenant_isolation():
    a = run_reasoning_v2(_state(ORG_A))
    b = run_reasoning_v2(_state(ORG_B))
    assert all(s.org_id == ORG_A for s in a)
    assert all(s.org_id == ORG_B for s in b)
    assert {s.signal_id for s in a}.isdisjoint({s.signal_id for s in b})


def test_all_engines_registered():
    """The registry is the contract: a signal type with no engine is dead code,
    and an engine missing from ENGINES silently stops producing."""
    names = {e.name for e in ENGINES}
    assert names == {
        "commitment_reasoning", "relationship_reasoning", "project_reasoning",
        "customer_reasoning", "decision_reasoning", "meeting_reasoning",
        # Proactive obligation engines — see engines/waiting.py and
        # engines/obligations.py.
        "waiting_reply_reasoning", "commitment_follow_up", "decision_owed",
        "priority", "opportunity", "leverage", "trend", "context",
    }


def test_obligation_signals_survive_a_stale_window_but_decay_signals_do_not():
    """The deliberate split between the two upper age ceilings.

    `_state()` pushes as_of ~40 days past the corpus, which is beyond the
    30-day DECAY window and inside the 90-day OBLIGATION window. That is the
    whole point of the split:

      * an unmet promise or an unresolved decision does not stop being owed
        because a month went by — a Chief of Staff that forgets promises past
        an arbitrary horizon has the worst failure mode this product can have;
      * silence, by contrast, genuinely stops being "cooling" and becomes a
        lapsed relationship, which a daily brief should not keep re-raising.

    This test previously asserted BOTH kinds fire at 40 days and had been
    failing since MAX_SURFACE_AGE_DAYS was introduced — the failure was real,
    and it was the commitment half that was wrong.
    """
    types = {s.signal_type for s in run_reasoning_v2(_state())}
    # Obligations — still owed at 40 days.
    assert SignalType.COMMITMENT_SLIPPING in types
    assert SignalType.DECISION_UNRESOLVED in types
    # Decay — past the actionable window at 40 days.
    assert SignalType.RELATIONSHIP_COOLING not in types
    # Always-on meta signals.
    assert SignalType.PRIORITY_LOAD in types
    assert SignalType.CONTEXT_SUMMARY in types


def test_decay_signals_fire_inside_their_own_window():
    """Guards the other side: the 30-day ceiling must not be so tight that
    decay signals stop working altogether."""
    import datetime as _dt

    from navis.state.v2.builders import build_projection
    from tests.state.fakes import corpus

    records = corpus(ORG_A)
    as_of = max(r.event_timestamp for r in records) + _dt.timedelta(days=10)
    types = {
        s.signal_type for s in run_reasoning_v2(build_projection(ORG_A, records, as_of=as_of))
    }
    assert SignalType.RELATIONSHIP_COOLING in types


def test_fresh_state_has_no_risk_signals():
    types = {s.signal_type for s in run_reasoning_v2(_state(stale=False))}
    assert SignalType.COMMITMENT_SLIPPING not in types
    assert SignalType.RELATIONSHIP_COOLING not in types


def test_empty_state_yields_no_signals():
    state = build_projection(ORG_A, [])
    assert run_reasoning_v2(state) == []  # offline-safe, no crash, deterministic


def test_trend_requires_prior_state():
    full = _state()
    small = build_projection(ORG_A, corpus(ORG_A)[:3])  # fewer entities
    no_prior = {s.signal_type for s in run_reasoning_v2(full)}
    with_prior = {s.signal_type for s in run_reasoning_v2(full, prior_state=small)}
    assert SignalType.ORG_DRIFT not in no_prior
    assert SignalType.ORG_DRIFT in with_prior  # growth vs prior → drift


import ast

_FORBIDDEN_IMPORT_ROOTS = {"openai", "anthropic", "httpx", "sqlalchemy", "redis"}
_FORBIDDEN_IMPORT_PREFIXES = ("navis.ml", "navis.memory", "navis.signals")
_FORBIDDEN_CALLS = {"now", "uuid4"}  # datetime.now(...), uuid.uuid4(...)


def test_reasoning_v2_has_no_clock_llm_or_io_dependencies():
    """AST-level (not text): no forbidden imports, no now()/uuid4()/random() calls."""
    root = pathlib.Path(__file__).resolve().parent.parent.parent / "navis" / "reasoning" / "v2"
    for py in root.rglob("*.py"):
        tree = ast.parse(py.read_text(), filename=str(py))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    root_mod = a.name.split(".")[0]
                    assert root_mod not in _FORBIDDEN_IMPORT_ROOTS, f"{py.name} imports {a.name}"
                    assert not a.name.startswith(_FORBIDDEN_IMPORT_PREFIXES), f"{py.name} imports {a.name}"
            elif isinstance(node, ast.ImportFrom) and node.module:
                root_mod = node.module.split(".")[0]
                assert root_mod not in _FORBIDDEN_IMPORT_ROOTS, f"{py.name} imports {node.module}"
                assert not node.module.startswith(_FORBIDDEN_IMPORT_PREFIXES), f"{py.name} imports {node.module}"
            elif isinstance(node, ast.Call):
                fn = node.func
                if isinstance(fn, ast.Attribute) and fn.attr in _FORBIDDEN_CALLS:
                    raise AssertionError(f"{py.name} calls forbidden {fn.attr}()")
                if isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name) and fn.value.id == "random":
                    raise AssertionError(f"{py.name} uses random.*")
