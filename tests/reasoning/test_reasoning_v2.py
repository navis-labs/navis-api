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


# ---- Stable identity vs. temporal observation -------------------------------
#
# A signal is two things and used to be stored as one. Its IDENTITY is stable by
# construction — not one `dedup_key` in the engine set carries time — and must
# stay stable, because execution_requests / learning_events / intelligence_insights
# all hold `reasoning_signal_ids` that cannot be recomputed once REASONING_VERSION
# moves. Its ATTRIBUTES are functions of `state.as_of` and change at every fold.
#
# Storing both on one row under ON CONFLICT DO NOTHING meant first detection won
# forever: production's commitment f33778d8 sits at 7.16 days / 0.7386 while the
# decision citing that exact row reports 12 days / 0.8951.


def test_observation_id_is_the_signal_snapshot_uniqueness_constraint():
    """One conclusion at one snapshot is one row, and a null watermark is ONE
    snapshot rather than infinitely many.

    The null case is the reason this is a hashed primary key and not a UNIQUE
    index: Postgres treats NULLs as distinct, so an organization with no folded
    memory would have accumulated a fresh duplicate observation on every pass.
    """
    import uuid

    from navis.reasoning.store import compute_observation_id

    sig = uuid.uuid5(uuid.NAMESPACE_DNS, "sig-1")
    other = uuid.uuid5(uuid.NAMESPACE_DNS, "sig-2")

    # Same conclusion, same snapshot -> same row.
    assert compute_observation_id(sig, "wm-1") == compute_observation_id(sig, "wm-1")
    # Same conclusion, DIFFERENT snapshot -> a new row. This is the write that
    # ON CONFLICT DO NOTHING used to swallow.
    assert compute_observation_id(sig, "wm-1") != compute_observation_id(sig, "wm-2")
    # Different conclusions never collide at one snapshot.
    assert compute_observation_id(sig, "wm-1") != compute_observation_id(other, "wm-1")
    # A null watermark is stable and distinct from the literal string.
    assert compute_observation_id(sig, None) == compute_observation_id(sig, None)
    assert compute_observation_id(sig, None) != compute_observation_id(sig, "-")


def test_signal_identity_is_unchanged_by_the_observation_split():
    """The whole point of splitting the tables was to leave this alone.

    If `compute_signal_id` ever starts hashing the watermark, every stored
    reference in execution/learning/intelligence orphans at once — 52 of
    production's 122 decision references already resolve to nothing.
    """
    import inspect

    from navis.reasoning.v2 import contract

    src = inspect.getsource(contract.compute_signal_id)
    assert "watermark" not in src, (
        "compute_signal_id must not hash the state watermark — stable identity is "
        "what makes historical reasoning references resolvable"
    )
    params = inspect.signature(contract.compute_signal_id).parameters
    assert list(params) == [
        "org_id", "reasoning_version", "engine_name", "engine_version",
        "signal_type", "entity_id", "dedup_key",
    ]


def test_observation_row_carries_every_snapshot_dependent_field():
    """A field left behind on the identity row is a value frozen at first
    detection — the exact defect. Everything derived from `state.as_of` must
    travel on the observation."""
    from navis.reasoning.store import _observation_row, compute_observation_id

    signal = run_reasoning_v2(_state())[0]
    row = _observation_row(signal)

    assert row["observation_id"] == compute_observation_id(
        signal.signal_id, signal.state_watermark
    )
    assert row["signal_id"] == signal.signal_id
    assert row["organization_id"] == signal.org_id
    assert row["state_watermark"] == signal.state_watermark
    assert row["state_as_of"] == signal.state_as_of
    # The four snapshot-dependent values, verbatim from the engine.
    assert row["confidence"] == signal.confidence
    assert row["severity"] == signal.severity
    assert row["why"] == signal.why
    assert row["evidence"] == (signal.evidence or {})
    # Logical time, never wall clock.
    assert row["created_at"] == signal.created_at == signal.state_as_of


async def test_persist_signals_writes_identity_once_and_an_observation_per_snapshot():
    """Two statements, two DIFFERENT conflict rules — that difference IS the fix.

    Asserts on the SQL actually emitted rather than on a fake's bookkeeping,
    matching how `tests/state/test_service.py` pins the snapshot upsert guard.
    """
    from navis.reasoning.store import persist_signals

    class _Res:
        rowcount = 1

    class _RecDB:
        def __init__(self):
            self.sql = []

        async def execute(self, stmt):
            from sqlalchemy.dialects import postgresql

            self.sql.append(str(stmt.compile(dialect=postgresql.dialect())).upper())
            return _Res()

    db = _RecDB()
    signals = run_reasoning_v2(_state())
    assert signals, "the corpus produced no signals to persist"
    await persist_signals(db, signals)

    assert len(db.sql) == 2, "expected one identity write and one observation write"
    identity, observation = db.sql

    # The stable conclusion: appended once, never overwritten.
    assert "INSERT INTO REASONING_SIGNALS" in identity
    assert "ON CONFLICT (SIGNAL_ID) DO NOTHING" in identity
    assert "DO UPDATE" not in identity, (
        "the identity row must never be overwritten — history would be destroyed"
    )

    # The observation: appended per (conclusion, snapshot).
    assert "INSERT INTO REASONING_SIGNAL_OBSERVATIONS" in observation
    assert "ON CONFLICT (OBSERVATION_ID) DO NOTHING" in observation
    assert "DO UPDATE" not in observation, "observations are append-only"

    # Ordering is a foreign-key requirement, not a preference.
    assert db.sql.index(identity) < db.sql.index(observation)


async def test_persist_signals_does_not_touch_the_database_when_there_is_nothing_to_write():
    from navis.reasoning.store import persist_signals

    class _ExplodingDB:
        async def execute(self, stmt):  # pragma: no cover - must never run
            raise AssertionError("wrote to the database with no signals")

    assert await persist_signals(_ExplodingDB(), []) == 0


async def test_observations_at_fails_closed_rather_than_substituting_a_snapshot():
    """An id with no observation at this watermark is ABSENT — never quietly
    served an older observation, the latest one, or the identity row's own
    first-detection columns. Substituting is how a decision came to report 11.85
    days while its cited row said 7.16."""
    from navis.reasoning.store import observations_at

    captured = {}

    class _Res:
        def all(self):
            return []

    class _FakeDB:
        async def scalars(self, stmt):
            captured["sql"] = str(
                stmt.compile(compile_kwargs={"literal_binds": True})
            )
            return _Res()

    import uuid

    out = await observations_at(
        _FakeDB(), ORG_A, [uuid.uuid4()], state_watermark="wm-1"
    )
    assert out == {}                       # absent, not substituted
    assert "organization_id" in captured["sql"] and ORG_A.hex in captured["sql"]
    assert "state_watermark" in captured["sql"]

    class _ExplodingDB:
        async def scalars(self, stmt):  # pragma: no cover - must never run
            raise AssertionError("queried the database with nothing to resolve")

    assert await observations_at(_ExplodingDB(), ORG_A, [], state_watermark=None) == {}


async def test_obligation_counts_read_observations_not_first_detection():
    """THE `shown > total` DEFECT, pinned at the query.

    Counting `reasoning_signals.state_watermark` asks "which conclusions were BORN
    at this snapshot?". The brief ranks conclusions that were TRUE at it, so every
    obligation carried over from an earlier fold vanished from the count — live,
    `shown = 4, total = 2`, with both missing entries being the top-ranked
    COMMITMENT_SLIPPING decisions. The blind spot grew with obligation age.
    """
    from navis.reasoning.store import signal_type_counts_at

    captured = {}

    class _Res:
        def all(self):
            return []

    class _FakeDB:
        async def execute(self, stmt):
            captured["sql"] = str(
                stmt.compile(compile_kwargs={"literal_binds": True})
            ).upper()
            return _Res()

    await signal_type_counts_at(
        _FakeDB(), ORG_A, reasoning_version=REASONING_VERSION, state_watermark="wm-9"
    )
    sql = captured["sql"]
    # The snapshot filter is on the OBSERVATION, not on the identity row.
    assert "REASONING_SIGNAL_OBSERVATIONS.STATE_WATERMARK" in sql
    assert "REASONING_SIGNALS.STATE_WATERMARK" not in sql
    # One conclusion observed here is one obligation, however many passes ran.
    assert "COUNT(DISTINCT" in sql
    # Still org-scoped, and on the observation's own indexed column.
    assert "REASONING_SIGNAL_OBSERVATIONS.ORGANIZATION_ID" in sql


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
