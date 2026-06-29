"""State V2 projection invariants: determinism, ordering, parity, no-reasoning, tenancy."""
from __future__ import annotations

import random

from navis.memory.envelope import EntityType
from navis.state.v2.builders import build_projection, continue_projection
from navis.state.v2.models import order_key
from navis.memory.envelope import EntityType
from navis.state.v2.folds import compute_links
from navis.state.v2.models import _VIEW_FOR
from tests.state.fakes import ORG_A, ORG_B, corpus, rec


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


# --- Null-overwrite protection (latest NON-NULL wins) -----------------------

def _person(natural_key, payload, *, offset, tag):
    return rec(EntityType.PERSON, natural_key, payload, offset=offset, tag=tag)


def test_null_does_not_clobber_known_field():
    # Same person entity: a rich snapshot, then a newer SPARSE record (email/name None).
    # The newer null must NOT erase the established email/name (Slack authorship case).
    recs = [
        _person("u1", {"email": "alice@acme.com", "name": "Alice", "role": "member"}, offset=0, tag="snap"),
        _person("u1", {"email": None, "name": None, "role": "member"}, offset=10, tag="auth"),
    ]
    p = build_projection(ORG_A, recs).people()[0]
    assert p.email == "alice@acme.com" and p.name == "Alice"


def test_latest_non_null_value_still_wins():
    # A newer NON-null value MUST win — the fix suppresses only nulls, not real updates.
    recs = [
        _person("u1", {"email": "old@acme.com", "name": "Alice"}, offset=0, tag="v1"),
        _person("u1", {"email": "new@acme.com", "name": None}, offset=10, tag="v2"),
    ]
    p = build_projection(ORG_A, recs).people()[0]
    assert p.email == "new@acme.com"   # newer non-null wins
    assert p.name == "Alice"           # newer null preserved the known name


def test_incremental_equals_full_with_interleaved_nulls():
    # Parity holds even when null records straddle the snapshot watermark.
    recs = sorted(
        [
            _person("u1", {"email": "a@acme.com", "name": "A"}, offset=0, tag="r0"),
            _person("u1", {"email": None, "name": None}, offset=1, tag="r1"),
            _person("u1", {"email": "b@acme.com", "name": None}, offset=2, tag="r2"),
            _person("u1", {"email": None, "name": "B"}, offset=3, tag="r3"),
        ],
        key=lambda r: order_key(r.event_timestamp, r.memory_id),
    )
    full = build_projection(ORG_A, recs)
    for split in range(len(recs) + 1):
        prev = build_projection(ORG_A, recs[:split])
        incr = continue_projection(prev, recs[split:])
        assert incr.to_dict() == full.to_dict(), f"parity broke at split={split}"
    # And the converged value is the latest non-null of each field.
    p = full.people()[0]
    assert p.email == "b@acme.com" and p.name == "B"


# --- folds.py: deterministic relational links -------------------------------

def _raw(eid, fields):
    return {"entity_id": eid, "first_seen": None, "last_seen": None,
            "fields": fields, "lists": {}, "links": {}, "provenance": {}}


def test_link_resolution_is_iteration_order_independent():
    # Two PERSON entities share an email (email-keyed + slack-keyed person carrying the
    # same snapshot email). Company->people resolution must NOT depend on dict order,
    # which differs between full build (first-appearance) and incremental (id-sorted).
    def company_people(person_order):
        people = {eid: _raw(eid, {"email": "a@acme.com"}) for eid in person_order}
        ents = {"person": people,
                "company": {"C": _raw("C", {"domain": "acme.com"})}}
        compute_links(ents)
        return ents["company"]["C"]["links"].get("people", [])
    assert company_people(["e1", "e2"]) == company_people(["e2", "e1"]) == ["e1"]


def test_channel_links_built():
    recs = [
        rec(EntityType.PERSON, "U1", {"slack_user_id": "U1", "email": None, "name": None}, offset=0, tag="p"),
        rec(EntityType.CHANNEL, "C1", {"channel_id": "C1", "name": "general"}, offset=1, tag="ch"),
        rec(EntityType.RELATIONSHIP, "U1", {"slack_user_id": "U1", "channel_id": "C1",
                                            "person_email": None, "direction": "active_in"}, offset=2, tag="r"),
    ]
    state = build_projection(ORG_A, recs)
    person_eid = next(iter(state.entities["person"]))
    channel_eid = next(iter(state.entities["channel"]))
    assert state.entities["person"][person_eid]["links"].get("channels") == [channel_eid]
    assert state.entities["channel"][channel_eid]["links"].get("people") == [person_eid]


def test_document_project_and_owner_links_built():
    recs = [
        rec(EntityType.PERSON, "owner@acme.com", {"email": "owner@acme.com", "name": "O", "role": "from"}, offset=0, tag="p"),
        rec(EntityType.PROJECT, "folder1", {"folder_id": "folder1", "name": "Proj"}, offset=1, tag="pr"),
        rec(EntityType.DOCUMENT, "doc1", {"document_id": "doc1", "title": "Spec",
                                          "owners": ["owner@acme.com"], "parents": ["folder1"]}, offset=2, tag="d"),
    ]
    state = build_projection(ORG_A, recs)
    person_eid = next(iter(state.entities["person"]))
    project_eid = next(iter(state.entities["project"]))
    doc_eid = next(iter(state.entities["document"]))
    assert state.entities["document"][doc_eid]["links"].get("projects") == [project_eid]
    assert state.entities["project"][project_eid]["links"].get("documents") == [doc_eid]
    assert state.entities["document"][doc_eid]["links"].get("owners") == [person_eid]
    assert state.entities["person"][person_eid]["links"].get("documents") == [doc_eid]


def test_new_links_order_independent_and_no_duplicates():
    recs = [
        rec(EntityType.PERSON, "U1", {"slack_user_id": "U1"}, offset=0, tag="p"),
        rec(EntityType.CHANNEL, "C1", {"channel_id": "C1"}, offset=1, tag="ch"),
        rec(EntityType.RELATIONSHIP, "U1", {"slack_user_id": "U1", "channel_id": "C1"}, offset=2, tag="r1"),
        rec(EntityType.RELATIONSHIP, "U1", {"slack_user_id": "U1", "channel_id": "C1"}, offset=3, tag="r2"),  # dup fact
    ]
    a = build_projection(ORG_A, recs)
    shuffled = recs[::-1]
    b = build_projection(ORG_A, shuffled)
    assert a.to_dict() == b.to_dict()  # order-independent
    channel_eid = next(iter(a.entities["channel"]))
    person_eid = next(iter(a.entities["person"]))
    # duplicate relationship facts collapse to a single, non-duplicated link
    assert a.entities["channel"][channel_eid]["links"]["people"] == [person_eid]


def test_links_recomputed_from_scratch_no_stale():
    # Recomputing on the same entities is idempotent (links never accumulate stale refs).
    state = build_projection(ORG_A, corpus())
    before = state.to_dict()
    compute_links(state.entities)  # run the fold again over the same buckets
    assert state.to_dict() == before


# --- models.py: typed-view completeness + Channel/Document accessors ---------

def test_view_for_covers_every_memory_entity_type():
    # Every entity type Memory can emit MUST have a typed view — guards against a future
    # entity type being added without a State accessor (which would silently drop it).
    assert set(_VIEW_FOR) == set(EntityType)


def test_channel_view_exposes_fields_and_links():
    recs = [
        rec(EntityType.PERSON, "U1", {"slack_user_id": "U1"}, offset=0, tag="p"),
        rec(EntityType.CHANNEL, "C1", {"channel_id": "C1", "name": "general",
                                       "visibility": "public", "workspace_id": "W1"}, offset=1, tag="ch"),
        rec(EntityType.RELATIONSHIP, "U1", {"slack_user_id": "U1", "channel_id": "C1"}, offset=2, tag="r"),
    ]
    state = build_projection(ORG_A, recs)
    ch = state.channels()[0]
    assert ch.name == "general" and ch.visibility == "public" and ch.workspace_id == "W1"
    person_eid = state.people()[0].entity_id
    assert ch.person_ids == [person_eid]
    assert state.people()[0].channel_ids == [ch.entity_id]


def test_document_view_exposes_fields_and_links():
    recs = [
        rec(EntityType.PERSON, "owner@acme.com", {"email": "owner@acme.com", "role": "from"}, offset=0, tag="p"),
        rec(EntityType.PROJECT, "folder1", {"folder_id": "folder1", "name": "Proj"}, offset=1, tag="pr"),
        rec(EntityType.DOCUMENT, "doc1", {"document_id": "doc1", "title": "Spec", "drive_id": "D1",
                                          "owners": ["owner@acme.com"], "parents": ["folder1"]}, offset=2, tag="d"),
    ]
    state = build_projection(ORG_A, recs)
    doc = state.documents()[0]
    assert doc.title == "Spec" and doc.drive_id == "D1"
    assert doc.owner_emails == ["owner@acme.com"]
    assert doc.project_ids == [state.projects()[0].entity_id]
    assert doc.owner_ids == [state.people()[0].entity_id]
    assert state.projects()[0].document_ids == [doc.entity_id]
    assert state.people()[0].document_ids == [doc.entity_id]


def test_channel_document_buckets_round_trip_and_count():
    recs = [
        rec(EntityType.CHANNEL, "C1", {"channel_id": "C1", "name": "g"}, offset=0, tag="ch"),
        rec(EntityType.DOCUMENT, "doc1", {"document_id": "doc1", "title": "S"}, offset=1, tag="d"),
    ]
    state = build_projection(ORG_A, recs)
    assert state.counts()["channel"] == 1 and state.counts()["document"] == 1
    from navis.state.v2.models import OrganizationalState
    assert OrganizationalState.from_dict(state.to_dict()).to_dict() == state.to_dict()


# --- snapshots.py: serialization, schema evolution, corruption recovery ------

def test_snapshot_serialization_is_deterministic_and_lossless():
    import json
    from navis.state.v2.models import OrganizationalState
    state = build_projection(ORG_A, corpus())
    d1, d2 = state.to_dict(), state.to_dict()
    assert d1 == d2                                            # stable
    rt = OrganizationalState.from_dict(d1).to_dict()
    assert rt == d1                                            # lossless round-trip
    assert json.dumps(d1, sort_keys=True) == json.dumps(rt, sort_keys=True)  # JSONB-stable bytes


def test_schema_version_round_trips_and_defaults_for_old_snapshots():
    from navis.state.v2.models import OrganizationalState, STATE_SCHEMA_VERSION
    state = build_projection(ORG_A, corpus())
    assert state.to_dict()["schema_version"] == STATE_SCHEMA_VERSION
    # A pre-existing snapshot missing schema_version (an upgrade) deserializes to the
    # current default instead of crashing.
    d = state.to_dict(); del d["schema_version"]
    assert OrganizationalState.from_dict(d).schema_version == STATE_SCHEMA_VERSION


def test_partial_corrupt_snapshot_is_resilient():
    # A snapshot missing entities/watermark/as_of yields safe empty defaults (no crash) —
    # the read can still be served, and a rebuild can replace it deterministically.
    import uuid as _u
    from navis.state.v2.models import OrganizationalState
    st = OrganizationalState.from_dict({"org_id": str(_u.uuid4())})
    assert st.entities == {} and st.counts()["person"] == 0
    assert st.watermark.last_timestamp is None and st.as_of is None


def test_corrupt_snapshot_missing_identity_fails_closed():
    # org_id is required identity; a snapshot without it must fail loudly, never silently
    # serve a tenant-less projection.
    import pytest
    from navis.state.v2.models import OrganizationalState
    with pytest.raises(KeyError):
        OrganizationalState.from_dict({})


def test_state_snapshot_declares_integrity_constraints():
    # The table forbids persisting a half-set watermark or a non-positive version.
    from sqlalchemy import Table
    from navis.state.v2.snapshots import StateSnapshot
    tbl = StateSnapshot.__table__
    assert isinstance(tbl, Table)
    names = {c.name for c in tbl.constraints}
    assert "ck_state_snapshots_watermark_paired" in names
    assert "ck_state_snapshots_projection_version_positive" in names
    assert "ck_state_snapshots_schema_version_positive" in names
