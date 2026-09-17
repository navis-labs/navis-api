"""Every signal source produces memory; Slack/Docs/Meet coverage + replay determinism +
the GDPR erasure cascade matcher."""
from __future__ import annotations

import dataclasses
import uuid

from navis.memory.envelope import EntityType
from navis.memory.extractors import extractors_for, run_extractors
from navis.memory.ids import entity_id
from navis.memory.service import build_memory_records
from navis.signals.base import SignalEventType, SignalSource
from tests.memory.fakes import (
    docs_view,
    meet_view,
    slack_channel_snapshot_view,
    slack_message_view,
    slack_user_snapshot_view,
)


def _types(drafts):
    return {d.entity_type for d in drafts}


# ---- Success criterion: no signal source terminates without an extractor ----

def test_every_signal_source_has_at_least_one_extractor():
    for src in SignalSource:
        # CRM/Notion have no connector yet (no signals emitted); the 6 live sources must map.
        if src in (SignalSource.GMAIL, SignalSource.CALENDAR, SignalSource.DRIVE,
                   SignalSource.DOCS, SignalSource.MEET, SignalSource.SLACK):
            assert extractors_for(src), f"{src} has no memory extractor"


# ---- Slack ----

def test_slack_message_produces_person_channel_relationship_commitment_decision():
    t = _types(run_extractors(slack_message_view()))
    assert EntityType.PERSON in t
    assert EntityType.CHANNEL in t
    assert EntityType.RELATIONSHIP in t
    assert EntityType.COMMITMENT in t   # "I'll send ... by Friday"
    assert EntityType.DECISION in t     # "We have decided to ship"


def test_slack_user_snapshot_produces_person_with_identity():
    drafts = run_extractors(slack_user_snapshot_view())
    person = next(d for d in drafts if d.entity_type == EntityType.PERSON)
    assert person.payload["slack_user_id"] == "U777"
    assert person.payload["email"] == "alice@acme.com"  # decrypted by the read seam in prod


def test_slack_channel_snapshot_produces_channel():
    drafts = run_extractors(slack_channel_snapshot_view())
    ch = next(d for d in drafts if d.entity_type == EntityType.CHANNEL)
    assert ch.payload["channel_id"] == "C123" and ch.payload["visibility"] == "private"


def test_slack_thread_reply_produces_thread():
    t = _types(run_extractors(slack_message_view(payload={"thread_root_ts": "1699999999.000"})))
    assert EntityType.THREAD in t


# ---- Docs ----

def test_docs_produces_document():
    drafts = run_extractors(docs_view())
    doc = next(d for d in drafts if d.entity_type == EntityType.DOCUMENT)
    assert doc.payload["document_id"] == "doc-1" and doc.payload["title"] == "Strategy Doc"
    assert doc.payload["owners"] == ["alice@acme.com"]


# ---- Docs lifecycle (07E) ----------------------------------------------
#
# Trashing is an APPENDED fact, not a mutation or a delete: Memory is
# append-only, so `status` flips via State's latest-non-null-wins fold.


def _doc_view(event_type, **kw):
    """`docs_view` pins its own event_type, so vary it on the built view rather
    than reshaping a fixture other tests share."""
    return dataclasses.replace(docs_view(), event_type=event_type, **kw)


def _doc_draft(view):
    return next(
        d for d in run_extractors(view) if d.entity_type == EntityType.DOCUMENT
    )


def _doc_status(view):
    return _doc_draft(view).payload["status"]


def test_doc_created_and_updated_are_active():
    assert _doc_status(_doc_view(SignalEventType.DOC_CREATED)) == "active"
    assert _doc_status(_doc_view(SignalEventType.DOC_UPDATED)) == "active"


def test_doc_trashed_is_marked_trashed():
    assert _doc_status(_doc_view(SignalEventType.DOC_TRASHED)) == "trashed"


def test_doc_restore_reactivates_via_a_later_active_fact():
    """Drive re-emits a restored file with trashed=false, which the connector
    classifies as CREATED/UPDATED. No un-delete mechanism is needed — the later
    'active' fact simply wins the fold."""
    trashed = _doc_view(SignalEventType.DOC_TRASHED, external_version="v3")
    restored = _doc_view(SignalEventType.DOC_UPDATED, external_version="v4")

    assert _doc_status(trashed) == "trashed"
    assert _doc_status(restored) == "active"


def test_status_is_never_null_so_a_stale_trashed_cannot_persist():
    """State's fold SKIPS nulls ('unknown here', not 'cleared'). A None status
    on a later edit would leave `trashed` as the surviving value forever."""
    for et in (SignalEventType.DOC_CREATED, SignalEventType.DOC_UPDATED,
               SignalEventType.DOC_TRASHED):
        assert _doc_status(_doc_view(et)) is not None


def test_repeated_trash_event_is_idempotent():
    """Same Drive version → same dedup_key → same memory_id → ON CONFLICT DO
    NOTHING. Re-delivery of one trash event cannot append a second fact."""
    view = _doc_view(SignalEventType.DOC_TRASHED, external_version="v9")
    a = run_extractors(view)
    b = run_extractors(view)

    assert [d.dedup_key for d in a] == [d.dedup_key for d in b] == ["v9"]
    assert [d.payload for d in a] == [d.payload for d in b]


def test_doc_lifecycle_preserves_identity_and_provenance():
    draft = _doc_draft(_doc_view(SignalEventType.DOC_TRASHED))

    assert draft.entity_natural_key == "doc-1"      # identity survives the trash
    assert draft.payload["document_id"] == "doc-1"
    assert draft.payload["drive_id"] == "D1"        # shared-drive lineage
    assert draft.payload["source"] == "docs"


def test_doc_lifecycle_is_organization_scoped():
    """A trashed Doc in org A must not touch org B's record. The natural key is
    the raw file id; org_id is mixed in when it is hashed into entity_id and
    memory_record_id, so the resulting records never collide."""
    org_a, org_b = uuid.uuid4(), uuid.uuid4()
    a = _doc_draft(_doc_view(SignalEventType.DOC_TRASHED, org_id=org_a))
    b = _doc_draft(_doc_view(SignalEventType.DOC_TRASHED, org_id=org_b))

    assert a.entity_natural_key == b.entity_natural_key == "doc-1"
    assert entity_id(org_a, EntityType.DOCUMENT.value, a.entity_natural_key) != entity_id(
        org_b, EntityType.DOCUMENT.value, b.entity_natural_key
    )


# ---- Meet ----

def test_meet_produces_meeting_with_conference_id_and_attendees():
    drafts = run_extractors(meet_view())
    meeting = next(d for d in drafts if d.entity_type == EntityType.MEETING)
    assert meeting.payload["conference_id"] == "conf-xyz"
    assert EntityType.PERSON in _types(drafts)
    # Meet meeting shares the calendar event id space (entity unification)
    assert meeting.entity_natural_key == "evt-1"


# ---- Replay determinism for the new sources ----

def test_new_sources_replay_deterministically():
    for view in (slack_message_view(), docs_view(), meet_view(), slack_channel_snapshot_view()):
        a = build_memory_records(view)
        b = build_memory_records(view)
        assert [r.memory_id for r in a] == [r.memory_id for r in b]
        assert [r.identity_tuple() for r in a] == [r.identity_tuple() for r in b]


# ---- Erasure cascade ----

def test_memory_erasure_matcher_and_registration():
    from navis.memory.governance import _references_subject, cascade_erase_memory
    from navis.signals.governance import _ERASURE_SUBSCRIBERS

    # the cascade is wired into the signal-layer erasure seam
    assert cascade_erase_memory in _ERASURE_SUBSCRIBERS
    # exact-match by email and by slack_user_id; no false positives
    assert _references_subject({"person_email": "a@x.com"}, email="A@X.com", slack_user_id=None)
    assert _references_subject({"attendees": ["b@x.com", "a@x.com"]}, email="a@x.com", slack_user_id=None)
    assert _references_subject({"slack_user_id": "U1"}, email=None, slack_user_id="U1")
    assert not _references_subject({"person_email": "a@x.com.evil"}, email="a@x.com", slack_user_id=None)


# ---- Signal pointer routing <-> Memory subscription: the seam that broke ----

def test_memory_subscribes_to_every_signal_pointer_route():
    """Regression (production fix in memory/service.py): the signals layer routes each
    ingested signal's pointer to a bus event (_BUS_EVENT_FOR_TYPE, else SIGNAL_INGESTED).
    Memory's subscription tuple silently drifted from that table — CALENDAR_EVENT_UPDATED
    and CALENDAR_EVENT_CANCELLED pointers had NO Memory consumer, so a meeting cancelled
    or rescheduled in Calendar never updated organizational truth (State kept it
    "confirmed"; MEETING_OVERLOAD counted it forever; the Meet deriver covers only
    conference events and skips cancellations). Every routable pointer event must have
    Memory's _on_signal_pointer subscribed, so the two files can never drift again."""
    import navis.memory.service as mem_service  # noqa: F401  (registers subscribers)
    from navis.platform.events import EventType, event_bus
    from navis.signals.service import _BUS_EVENT_FOR_TYPE

    routable = set(_BUS_EVENT_FOR_TYPE.values()) | {EventType.SIGNAL_INGESTED}
    for et in routable:
        handlers = event_bus._handlers.get(et, [])
        assert any(h.__name__ == "_on_signal_pointer" for h in handlers), (
            f"signal pointers routed to {et} have no Memory consumer — "
            "signals reaching this event type silently never become memories"
        )
