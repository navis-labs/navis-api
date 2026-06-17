"""State V2 projection invariants: determinism, ordering, parity, no-reasoning, tenancy."""
from __future__ import annotations

import random

from navis.memory.envelope import EntityType
from navis.state.v2.builders import build_projection, continue_projection
from navis.state.v2.models import order_key
from tests.state.fakes import ORG_A, ORG_B, corpus


def test_projection_is_deterministic():
    a = build_projection(ORG_A, corpus())
    b = build_projection(ORG_A, corpus())
    assert a.to_dict() == b.to_dict()


def test_ordering_independent_of_input_order():
    records = corpus()
    shuffled = records[:]
    random.Random(7).shuffle(shuffled)
    a = build_projection(ORG_A, records)
    b = build_projection(ORG_A, shuffled)
    assert a.to_dict() == b.to_dict()  # internal sort makes input order irrelevant


def test_latest_wins_and_accumulation():
    state = build_projection(ORG_A, corpus())
    meeting = state.meetings()[0]
    assert meeting.status == "cancelled"          # latest version wins
    assert meeting.participants == ["alice@acme.com"]  # latest attendees, not union
    thread = state.threads()[0]
    assert sorted(thread.message_ids) == ["m1", "m2"]  # append-only accumulation
    rel = state.relationships()[0]
    assert sorted(rel.directions) == ["inbound", "outbound"]


def test_cross_links_are_built():
    state = build_projection(ORG_A, corpus())
    alice = next(p for p in state.people() if p.email == "alice@acme.com")
    assert alice.relationship_ids  # person -> relationship
    rel = state.relationships()[0]
    assert rel.commitment_ids       # relationship -> commitment (by speaker email)
    commitment = state.commitments()[0]
    assert commitment.owner_id      # commitment -> owner person


def test_incremental_equals_full_rebuild():
    records = sorted(corpus(), key=lambda r: order_key(r.event_timestamp, r.memory_id))
    full = build_projection(ORG_A, records)
    for split in (0, 1, 5, 7, len(records)):
        prev = build_projection(ORG_A, records[:split])
        incr = continue_projection(prev, records[split:])
        assert incr.to_dict() == full.to_dict(), f"parity broke at split={split}"


def test_delete_and_rebuild_is_identical():
    # "Drop State, replay Memory" == independent rebuild from the same records.
    first = build_projection(ORG_A, corpus())
    rebuilt = build_projection(ORG_A, corpus())
    assert rebuilt.to_dict() == first.to_dict()
    assert rebuilt.watermark.to_dict() == first.watermark.to_dict()


def test_multi_tenant_isolation():
    a = build_projection(ORG_A, corpus(ORG_A))
    b = build_projection(ORG_B, corpus(ORG_B))
    assert a.org_id != b.org_id
    a_ids = {e for bucket in a.entities.values() for e in bucket}
    b_ids = {e for bucket in b.entities.values() for e in bucket}
    assert a_ids.isdisjoint(b_ids)  # entity_ids encode org → no cross-tenant overlap


_FORBIDDEN = {
    "risk", "confidence", "recommendation", "recommendations", "priority", "health",
    "opportunity", "opportunities", "leverage", "intelligence", "strategy", "score",
    "scores", "sentiment", "probability", "prediction", "predictions",
}


def _all_keys(obj):
    keys = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            keys.add(str(k).lower())
            keys |= _all_keys(v)
    elif isinstance(obj, list):
        for v in obj:
            keys |= _all_keys(v)
    return keys


def test_no_reasoning_keys_in_projection():
    keys = _all_keys(build_projection(ORG_A, corpus()).to_dict())
    leaked = keys & _FORBIDDEN
    assert not leaked, f"State leaked Reasoning keys: {leaked}"


def test_every_entity_type_projects():
    state = build_projection(ORG_A, corpus())
    counts = state.counts()
    for et in (EntityType.PERSON, EntityType.COMPANY, EntityType.RELATIONSHIP,
               EntityType.COMMITMENT, EntityType.THREAD, EntityType.MEETING, EntityType.DECISION):
        assert counts[et.value] >= 1
