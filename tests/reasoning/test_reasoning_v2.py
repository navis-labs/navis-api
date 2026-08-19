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


def _human_evidence(state) -> dict[str, frozenset[str]]:
    """Human-bearing evidence for every relationship in a projection.

    RELATIONSHIP_COOLING requires POSITIVE evidence of a human counterparty
    since reasoning v4, so a window test that supplied none would be asserting
    the relevance gate rather than the age ceiling it is actually about. Every
    live relationship carries classification evidence, so this is the normal
    case, not a contrivance.
    """
    return {r.entity_id: frozenset({"meeting"}) for r in state.relationships()}


def test_decay_signals_fire_inside_their_own_window():
    """Guards the other side: the 30-day ceiling must not be so tight that
    decay signals stop working altogether."""
    import datetime as _dt

    from navis.state.v2.builders import build_projection
    from tests.state.fakes import corpus

    records = corpus(ORG_A)
    as_of = max(r.event_timestamp for r in records) + _dt.timedelta(days=10)
    state = build_projection(ORG_A, records, as_of=as_of)
    types = {
        s.signal_type
        for s in run_reasoning_v2(state, evidence=_human_evidence(state))
    }
    assert SignalType.RELATIONSHIP_COOLING in types


def test_the_decay_window_is_still_enforced_for_relevant_relationships():
    """The gate must not become the ONLY thing keeping cooling quiet: a
    human-bearing relationship past the 30-day ceiling still stays silent."""
    state = _state()  # as_of ~40 days past the corpus
    types = {
        s.signal_type
        for s in run_reasoning_v2(state, evidence=_human_evidence(state))
    }
    assert SignalType.RELATIONSHIP_COOLING not in types
    assert SignalType.COMMITMENT_SLIPPING in types  # obligations still owed


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


# ---- Durable provenance (reasoning/store.py) --------------------------------
#
# Execution, Observation, Learning and Intelligence all persist
# `reasoning_signal_ids`. Those were references to rows that did not exist, and
# they cannot be recomputed: `compute_signal_id` hashes `reasoning_version`, so
# the 2 -> 3 bump permanently changed every id minted before it. `store.py` is
# the table they now resolve to; it lives OUTSIDE v2 precisely so the engines can
# keep the purity the AST test above pins.


def test_signal_ids_are_pinned_and_must_not_drift():
    """The identity tuple, spelled out.

    Every stored `reasoning_signal_id` in execution_requests, outcomes,
    learning_events and intelligence_insights is one of these. Changing the key
    format — or `REASONING_VERSION` — orphans every one of them with no way back,
    which is exactly what already happened once at v2.
    """
    import uuid

    from navis.reasoning.v2.contract import _REASONING_NAMESPACE, compute_signal_id

    org = ORG_A
    expected = uuid.uuid5(
        _REASONING_NAMESPACE,
        f"{org}|{REASONING_VERSION}|commitment_reasoning|1"
        f"|{SignalType.COMMITMENT_SLIPPING.value}|ent-1|dedup-1",
    )
    assert compute_signal_id(
        org, REASONING_VERSION, "commitment_reasoning", "1",
        SignalType.COMMITMENT_SLIPPING, "ent-1", "dedup-1",
    ) == expected
    # Same tuple, different org => different id. Ids carry tenancy themselves,
    # so a replayed id can never mean another org's signal.
    assert compute_signal_id(
        ORG_B, REASONING_VERSION, "commitment_reasoning", "1",
        SignalType.COMMITMENT_SLIPPING, "ent-1", "dedup-1",
    ) != expected


def test_store_round_trips_every_field_reasoning_emitted():
    """`_row` is the whole mapping; a dropped field is provenance silently lost."""
    from navis.reasoning.store import _row

    signal = run_reasoning_v2(_state())[0]
    row = _row(signal)
    assert row["signal_id"] == signal.signal_id
    assert row["organization_id"] == signal.org_id
    assert row["signal_type"] == signal.signal_type.value
    assert row["confidence"] == signal.confidence and row["severity"] == signal.severity
    assert row["engine_name"] == signal.engine_name
    assert row["engine_version"] == signal.engine_version
    assert row["state_watermark"] == signal.state_watermark
    assert row["reasoning_version"] == signal.reasoning_version
    # Logical time, never wall clock — a replayed row is byte-identical.
    assert row["created_at"] == signal.created_at == signal.state_as_of
    # The engine's own evidence, verbatim. Never re-derived here.
    assert row["evidence"] == (signal.evidence or {})


async def test_resolve_signals_always_filters_by_organization():
    """org_id is a required filter, not a convenience.

    These ids travel through Execution/Observation/Learning payloads. A lookup
    keyed on id alone would resolve another tenant's signal the moment an id were
    replayed across orgs.
    """
    from navis.reasoning.store import resolve_signals

    captured = {}

    class _FakeResult:
        def all(self):
            return []

    class _FakeDB:
        async def scalars(self, stmt):
            captured["sql"] = str(stmt.compile(compile_kwargs={"literal_binds": True}))
            return _FakeResult()

    import uuid

    await resolve_signals(_FakeDB(), ORG_A, [uuid.uuid4()])
    # Rendered without hyphens by the postgres UUID type — assert on the value
    # actually bound, not on how a human writes it.
    assert "organization_id" in captured["sql"]
    assert ORG_A.hex in captured["sql"]


async def test_resolve_signals_skips_unresolvable_references():
    """A non-uuid reference cannot address a signal, and an empty set must not
    round-trip to the database at all."""
    from navis.reasoning.store import resolve_signals

    class _ExplodingDB:
        async def scalars(self, stmt):  # pragma: no cover - must never run
            raise AssertionError("queried the database with nothing to resolve")

    assert await resolve_signals(_ExplodingDB(), ORG_A, []) == {}
    assert await resolve_signals(_ExplodingDB(), ORG_A, ["not-a-uuid", None]) == {}


def test_store_lives_outside_v2_so_the_engines_stay_pure():
    """Storage and purity coexist only while they are in different modules."""
    store = pathlib.Path(__file__).resolve().parent.parent.parent / "navis" / "reasoning" / "store.py"
    assert store.exists(), "reasoning provenance has no storage module"
    v2 = pathlib.Path(__file__).resolve().parent.parent.parent / "navis" / "reasoning" / "v2"
    assert store.parent != v2, "store.py inside v2 would break the no-sqlalchemy pin"
    assert "sqlalchemy" in store.read_text(encoding="utf-8")  # it really is the DB seam
