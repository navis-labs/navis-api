"""A Decision must name the organizational entity it is about.

`entity_type` / `entity_id` exist on the ReasoningSignal, are read by the
ranker to dedupe, and are carried by Intelligence — but used to stop at the
Decision boundary, leaving a decision that named no entity at all. That broke
the only join between a decision and the entity a client already has on screen
(`state/v2/scope.py::EntityRef` reads `.entity_id` off the same State views the
reasoning engines do, so the two share one id space).

These tests pin the passthrough and, just as importantly, pin the things the
passthrough must NOT change: decision identity and brief bytes.
"""
from __future__ import annotations

import uuid

from navis.decisions.base import compute_decision_id
from navis.decisions.what_matters_today import WhatMattersTodayEngine
from navis.memory.envelope import EntityType
from navis.reasoning.v2 import REASONING_VERSION, run_reasoning_v2
from navis.state.v2.builders import build_projection
from tests.state.fakes import ORG_A, corpus, rec


def _signals(org=ORG_A):
    recs = corpus(org) + [
        rec(EntityType.PERSON, "hb@acme.com", {"email": "hb@acme.com", "role": "from"},
            offset=40 * 86400, tag="hb", org_id=org)
    ]
    state = build_projection(org, recs)
    return run_reasoning_v2(state), state


def _decide(signals, state, org=ORG_A):
    return WhatMattersTodayEngine().decide(
        signals, org_id=org, reasoning_version=REASONING_VERSION,
        state_watermark=str(state.watermark.last_memory_id), as_of=state.as_of,
    )


def _signal_by_id(signals, signal_id: str):
    return next(s for s in signals if str(s.signal_id) == signal_id)


def test_decision_carries_its_signals_entity_verbatim():
    signals, state = _signals()
    decisions = _decide(signals, state)
    assert decisions

    for d in decisions:
        origin = _signal_by_id(signals, d.reasoning_signal_ids[0])
        # Verbatim, both fields, including when the signal is org-wide (None).
        assert d.entity_type == origin.entity_type
        assert d.entity_id == origin.entity_id


def test_entity_lineage_survives_serialization():
    signals, state = _signals()
    for d in _decide(signals, state):
        wire = d.to_dict()
        # ALWAYS present — a reader must be able to tell "org-wide" from
        # "this build does not report it".
        assert "entity_type" in wire
        assert "entity_id" in wire
        assert wire["entity_type"] == d.entity_type
        assert wire["entity_id"] == d.entity_id


def test_at_least_one_decision_names_a_real_entity():
    # Guards against the passthrough being vacuously satisfied by every signal
    # in the fixture happening to be org-wide.
    signals, state = _signals()
    decisions = _decide(signals, state)
    assert any(d.entity_id is not None and d.entity_type for d in decisions)


def test_entity_is_not_part_of_decision_identity():
    # Corroboration and provenance may never fork an id: identity stays
    # (org, reasoning_version, watermark, signal_id).
    signals, state = _signals()
    wm = str(state.watermark.last_memory_id)
    for d in _decide(signals, state):
        assert d.decision_id == compute_decision_id(
            ORG_A, REASONING_VERSION, wm, uuid.UUID(d.reasoning_signal_ids[0])
        )


def test_entity_passthrough_is_deterministic():
    signals, state = _signals()
    a = [d.to_dict() for d in _decide(signals, state)]
    b = [d.to_dict() for d in _decide(signals, state)]
    assert a == b


def test_decision_entity_ids_share_the_state_entity_id_space():
    """The join this field exists for.

    A decision's `entity_id` must resolve to an entity in the very State the
    reasoning ran over — the same store `scope_for` selects `/context`'s
    entities from. If these two drifted into separate id spaces the field
    would look like a join and silently never match.
    """
    signals, state = _signals()
    decisions = _decide(signals, state)

    named = [d for d in decisions if d.entity_id]
    assert named, "fixture produced no entity-scoped decision"

    for d in named:
        bucket = state.entities.get(d.entity_type or "", {})
        assert d.entity_id in bucket, (
            f"decision entity {d.entity_type}:{d.entity_id} is not a State entity"
        )
