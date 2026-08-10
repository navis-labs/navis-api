"""COMMITMENT_FOLLOW_UP and DECISION_OWED — the Phase 2A proactive detectors.

Both are obligations: an unmet duty that does not decay. Both are detected
deterministically from facts Memory already records — no model decides whether a
promise was kept or a decision was made, because both errors are expensive:
chasing someone who already delivered destroys trust in the product, and
forgetting a promise is the failure a Chief of Staff exists to prevent.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone


from navis.memory.envelope import EntityType
from navis.reasoning.v2 import run_reasoning_v2
from navis.reasoning.v2.contract import SignalType
from navis.reasoning.v2.engines.obligations import (
    DECISION_OWED_MIN_DAYS,
    FOLLOW_UP_GRACE_DAYS,
)
from navis.state.v2.builders import MemoryRecordView, build_projection

NS = uuid.UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
ORG = uuid.UUID("11111111-1111-1111-1111-111111111111")
NOW = datetime(2026, 8, 9, 12, 0, tzinfo=timezone.utc)


def _iso(days_from_now: float) -> str:
    return (NOW + timedelta(days=days_from_now)).isoformat()


def _rec(entity_type, natural_key, payload, *, at, tag):
    return MemoryRecordView(
        memory_id=uuid.uuid5(NS, f"mem:{tag}"),
        org_id=ORG,
        entity_type=entity_type.value,
        entity_id=uuid.uuid5(NS, f"ent:{entity_type.value}:{natural_key}"),
        event_timestamp=at,
        payload=payload,
        source_signal_event_id=uuid.uuid5(NS, f"sig:{tag}"),
        schema_version=1,
        confidence=0.9,
    )


def _commitment(
    *,
    key="c-1",
    text="I'll send you the product deck by Friday.",
    speaker="chrissy@partner.com",
    speaker_name="Chrissy Lane",
    thread="t-10",
    promised_days_ago=7.0,
    due_days_ago: float | None = 2.0,
    due_text="Friday",
    subject="Product deck",
):
    promised = NOW - timedelta(days=promised_days_ago)
    return _rec(
        EntityType.COMMITMENT,
        key,
        {
            "text": text,
            "speaker_email": speaker,
            "speaker_name": speaker_name,
            "subject": subject,
            "thread_id": thread,
            "due_at": None if due_days_ago is None else _iso(-due_days_ago),
            "due_text": None if due_days_ago is None else due_text,
            "promised_at": promised.isoformat(),
        },
        at=promised,
        tag=f"commit:{key}:{promised_days_ago}",
    )


def _thread(
    *,
    key="t-10",
    at_days_ago=7.0,
    direction="inbound",
    from_email="chrissy@partner.com",
    subject="Product deck",
    fulfilment_days_ago: float | None = None,
    fulfilment_from: str | None = None,
    resolution_days_ago: float | None = None,
    resolution_from: str | None = None,
    tag: str | None = None,
):
    payload = {
        "thread_id": key,
        "subject": subject,
        "direction": direction,
        "from_email": from_email,
        "category": "personal",
        "is_automated": False,
    }
    if fulfilment_days_ago is not None:
        payload["fulfilment_at"] = _iso(-fulfilment_days_ago)
        payload["fulfilment_from"] = fulfilment_from or from_email
    if resolution_days_ago is not None:
        payload["resolution_at"] = _iso(-resolution_days_ago)
        payload["resolution_from"] = resolution_from or from_email
    return _rec(
        EntityType.THREAD,
        key,
        payload,
        at=NOW - timedelta(days=at_days_ago),
        tag=tag or f"thread:{key}:{at_days_ago}:{direction}",
    )


def _decision_request(
    *,
    key="d-1",
    text="Can you approve the new hiring budget?",
    asker="mia@team.com",
    asker_name="Mia Chen",
    thread="t-20",
    asked_days_ago=3.0,
    subject="Hiring budget",
):
    asked = NOW - timedelta(days=asked_days_ago)
    return _rec(
        EntityType.DECISION,
        key,
        {
            "text": text,
            "kind": "request",
            "asked_by": asker,
            "asked_by_name": asker_name,
            "asked_at": asked.isoformat(),
            "subject": subject,
            "thread_id": thread,
            "rfc_message_id": f"<{key}@mail>",
        },
        at=asked,
        tag=f"decreq:{key}:{asked_days_ago}",
    )


def _signals(*records, want: SignalType):
    state = build_projection(ORG, list(records), as_of=NOW)
    return [s for s in run_reasoning_v2(state) if s.signal_type is want]


def _follow_ups(*records):
    return _signals(*records, want=SignalType.COMMITMENT_FOLLOW_UP)


def _owed(*records):
    return _signals(*records, want=SignalType.DECISION_OWED)


# ══════════════════════════════════════════════════════════════════════
# COMMITMENT FOLLOW-UP
# ══════════════════════════════════════════════════════════════════════


def test_overdue_commitment_produces_a_candidate():
    (s,) = _follow_ups(_commitment(), _thread())
    # Person first — "who do I chase" is the actionable part.
    assert s.title.startswith("Follow up with Chrissy Lane")
    assert "product deck" in s.title
    assert s.evidence["person_email"] == "chrissy@partner.com"
    assert s.evidence["person_name"] == "Chrissy Lane"
    assert s.evidence["due_text"] == "Friday"
    assert s.evidence["overdue_days"] == 2.0
    assert s.evidence["thread_id"] == "t-10"
    assert s.evidence["commitment"].startswith("I'll send you the product deck")


def test_the_signal_explains_itself():
    (s,) = _follow_ups(_commitment(), _thread())
    assert "due Friday" in s.why
    assert "No delivery in the thread since" in s.why


def test_commitment_due_tomorrow_is_not_overdue():
    assert _follow_ups(_commitment(due_days_ago=-1.0), _thread()) == []


def test_commitment_due_today_gets_the_rest_of_the_day():
    """"by Friday" is met any time on Friday. A card at 00:01 Saturday reads as
    pedantic; the grace window is what makes it wait."""
    assert _follow_ups(_commitment(due_days_ago=0.0), _thread()) == []
    assert _follow_ups(_commitment(due_days_ago=FOLLOW_UP_GRACE_DAYS - 0.1), _thread()) == []
    assert len(_follow_ups(_commitment(due_days_ago=FOLLOW_UP_GRACE_DAYS + 0.1), _thread())) == 1


def test_an_undated_commitment_never_becomes_a_follow_up():
    """No date stated means no date invented. It stays with the existing
    COMMITMENT_SLIPPING policy instead."""
    assert _follow_ups(_commitment(due_days_ago=None), _thread()) == []


def test_a_dated_commitment_does_not_ALSO_fire_as_slipping():
    """The two engines are mutually exclusive so one promise is never two cards."""
    state = build_projection(ORG, [_commitment(), _thread()], as_of=NOW)
    types = [s.signal_type for s in run_reasoning_v2(state)]
    assert SignalType.COMMITMENT_FOLLOW_UP in types
    assert SignalType.COMMITMENT_SLIPPING not in types


def test_an_undated_commitment_still_fires_as_slipping():
    state = build_projection(
        ORG, [_commitment(due_days_ago=None), _thread()], as_of=NOW
    )
    types = [s.signal_type for s in run_reasoning_v2(state)]
    assert SignalType.COMMITMENT_SLIPPING in types


# ── Completion suppresses it ──────────────────────────────────────────


def test_a_later_delivery_in_the_thread_suppresses_the_follow_up():
    """"Here's the deck" arrives after the promise — nothing to chase."""
    assert _follow_ups(
        _commitment(),
        _thread(fulfilment_days_ago=1.0, fulfilment_from="chrissy@partner.com"),
    ) == []


def test_a_delivery_BEFORE_the_promise_does_not_excuse_it():
    """They sent something last week, then promised something new. The old
    delivery says nothing about the new promise."""
    assert len(
        _follow_ups(
            _commitment(promised_days_ago=5.0, due_days_ago=2.0),
            _thread(fulfilment_days_ago=9.0, fulfilment_from="chrissy@partner.com"),
        )
    ) == 1


def test_a_delivery_by_SOMEONE_ELSE_does_not_discharge_the_promise():
    assert len(
        _follow_ups(
            _commitment(),
            _thread(fulfilment_days_ago=1.0, fulfilment_from="someone@else.com"),
        )
    ) == 1


def test_an_explicitly_completed_commitment_is_skipped():
    rec = _commitment()
    rec.payload["status"] = "completed"
    assert _follow_ups(rec, _thread()) == []


# ── Bounds and hygiene ────────────────────────────────────────────────


def test_beyond_the_obligation_window_it_stops():
    assert _follow_ups(
        _commitment(promised_days_ago=200.0, due_days_ago=195.0), _thread(at_days_ago=200.0)
    ) == []


def test_a_commitment_with_no_speaker_is_skipped():
    """Nobody to follow up with."""
    assert _follow_ups(_commitment(speaker=""), _thread()) == []


def test_longer_overdue_scores_higher():
    (short,) = _follow_ups(_commitment(due_days_ago=2.0), _thread())
    (long_,) = _follow_ups(
        _commitment(key="c-2", due_days_ago=25.0, promised_days_ago=30.0),
        _thread(),
    )
    assert long_.severity > short.severity


def test_multiple_commitments_produce_separate_candidates():
    signals = _follow_ups(
        _commitment(key="c-1", thread="t-10"),
        _thread(key="t-10"),
        _commitment(
            key="c-2", text="We'll deliver the report by Monday.",
            speaker="sam@vendor.com", speaker_name="Sam", thread="t-11",
        ),
        _thread(key="t-11", from_email="sam@vendor.com"),
    )
    assert {s.evidence["person_email"] for s in signals} == {
        "chrissy@partner.com", "sam@vendor.com"
    }


def test_a_duplicate_sync_does_not_duplicate_the_candidate():
    c, t = _commitment(), _thread()
    assert len(_follow_ups(c, c, t, t)) == 1


def test_the_signal_id_is_stable_across_passes():
    """A stable signal id is what makes the decision id — and so the
    ExecutionRequest — idempotent across brief runs."""
    records = [_commitment(), _thread()]
    assert _follow_ups(*records)[0].signal_id == _follow_ups(*records)[0].signal_id


# ══════════════════════════════════════════════════════════════════════
# DECISION OWED
# ══════════════════════════════════════════════════════════════════════


def test_a_genuine_decision_request_produces_a_candidate():
    (s,) = _owed(_decision_request(), _thread(key="t-20", from_email="mia@team.com"))
    assert s.title == "Decide: Hiring budget"
    assert s.evidence["person_email"] == "mia@team.com"
    assert s.evidence["person_name"] == "Mia Chen"
    assert s.evidence["decision"] == "Can you approve the new hiring budget?"
    assert s.evidence["age_days"] == 3.0
    assert s.evidence["thread_id"] == "t-20"


def test_the_decision_signal_explains_itself():
    (s,) = _owed(_decision_request(), _thread(key="t-20", from_email="mia@team.com"))
    assert "Mia Chen asked" in s.why
    assert "3 days ago" in s.why
    assert "no decision has been recorded" in s.why


def test_a_recorded_decision_is_not_one_that_is_owed():
    """`kind == "record"` is the opposite state of the same entity type."""
    rec = _decision_request()
    rec.payload["kind"] = "record"
    assert _owed(rec, _thread(key="t-20")) == []


def test_too_recent_is_not_yet_owed():
    assert _owed(
        _decision_request(asked_days_ago=DECISION_OWED_MIN_DAYS - 0.5),
        _thread(key="t-20"),
    ) == []


# ── Resolution suppresses it ──────────────────────────────────────────


def test_the_founder_replying_in_the_thread_resolves_it():
    """ANY outbound reply after the ask counts. Judging decisiveness would need
    a model, and telling a founder they owe an answer they already gave is the
    trust-destroying failure."""
    assert _owed(
        _decision_request(asked_days_ago=4.0),
        _thread(key="t-20", at_days_ago=1.0, direction="outbound",
                from_email="founder@navis.ai"),
    ) == []


def test_a_founder_reply_BEFORE_the_ask_does_not_resolve_it():
    """They replied, and THEN were asked something new."""
    assert len(
        _owed(
            _decision_request(asked_days_ago=2.0),
            _thread(key="t-20", at_days_ago=9.0, direction="outbound",
                    from_email="founder@navis.ai"),
        )
    ) == 1


def test_another_participant_recording_a_decision_resolves_it():
    """"We've decided to go with option B" from anyone closes the question."""
    assert _owed(
        _decision_request(asked_days_ago=5.0),
        _thread(key="t-20", from_email="mia@team.com",
                resolution_days_ago=1.0, resolution_from="cofounder@navis.ai"),
    ) == []


def test_a_resolution_BEFORE_the_ask_does_not_resolve_it():
    assert len(
        _owed(
            _decision_request(asked_days_ago=2.0),
            _thread(key="t-20", from_email="mia@team.com", resolution_days_ago=9.0),
        )
    ) == 1


# ── Bounds and hygiene ────────────────────────────────────────────────


def test_beyond_the_obligation_window_it_stops():
    assert _owed(
        _decision_request(asked_days_ago=200.0), _thread(key="t-20", at_days_ago=200.0)
    ) == []


def test_a_request_with_no_asker_is_skipped():
    assert _owed(_decision_request(asker=""), _thread(key="t-20")) == []


def test_multiple_unresolved_decisions_produce_separate_candidates():
    signals = _owed(
        _decision_request(key="d-1", thread="t-20"),
        _thread(key="t-20", from_email="mia@team.com"),
        _decision_request(
            key="d-2", text="Should we launch on Monday or Tuesday?",
            asker="raj@team.com", asker_name="Raj", thread="t-21",
            subject="Launch date",
        ),
        _thread(key="t-21", from_email="raj@team.com"),
    )
    assert {s.evidence["person_email"] for s in signals} == {"mia@team.com", "raj@team.com"}


def test_a_duplicate_sync_does_not_duplicate_the_decision_candidate():
    d, t = _decision_request(), _thread(key="t-20", from_email="mia@team.com")
    assert len(_owed(d, d, t, t)) == 1


def test_older_asks_score_higher():
    (short,) = _owed(_decision_request(asked_days_ago=2.0), _thread(key="t-20"))
    (long_,) = _owed(
        _decision_request(key="d-2", asked_days_ago=20.0), _thread(key="t-20")
    )
    assert long_.severity > short.severity


def test_determinism_across_passes():
    records = [_decision_request(), _thread(key="t-20", from_email="mia@team.com")]
    state = build_projection(ORG, records, as_of=NOW)
    a = [s.to_dict() for s in run_reasoning_v2(state)]
    b = [s.to_dict() for s in run_reasoning_v2(state)]
    assert a == b
