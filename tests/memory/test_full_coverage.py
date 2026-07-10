"""Every signal source produces memory; Slack/Docs/Meet coverage + replay determinism +
the GDPR erasure cascade matcher."""
from __future__ import annotations

from navis.memory.envelope import EntityType
from navis.memory.extractors import extractors_for, run_extractors
from navis.memory.service import build_memory_records
from navis.signals.base import SignalSource
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
