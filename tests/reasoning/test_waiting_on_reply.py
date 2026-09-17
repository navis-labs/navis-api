"""WAITING_ON_USER_REPLY — the first proactive signal.

"John emailed you 4 days ago and you never replied" is the canonical Chief of
Staff observation, and the one Navis could not make.

The detection is DELIBERATELY not a judgement. Gmail labels outbound mail
`SENT`; ThreadExtractor records that as `direction`; State folds thread records
newest-last with latest-wins; so `ThreadState.direction == "inbound"` means "the
other party spoke last" as a matter of record, not inference. These tests pin
that chain end to end, because the failure modes are expensive in both
directions — nagging someone who already replied destroys trust in the product,
and missing a waiting customer is the whole reason the feature exists.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from navis.memory.envelope import EntityType
from navis.reasoning.v2 import run_reasoning_v2
from navis.reasoning.v2.contract import SignalType
from navis.reasoning.v2.engines.waiting import WAITING_MIN_DAYS
from navis.state.v2.builders import MemoryRecordView, build_projection

NS = uuid.UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
ORG = uuid.UUID("11111111-1111-1111-1111-111111111111")
NOW = datetime(2026, 8, 9, 9, 0, tzinfo=timezone.utc)


def _msg(
    thread: str,
    *,
    days_ago: float,
    direction: str,
    from_email: str = "john@partner.com",
    from_name: str | None = "John Carter",
    subject: str = "Contract review",
    category: str = "personal",
    is_automated: bool = False,
    tag: str | None = None,
) -> MemoryRecordView:
    """One THREAD memory record — i.e. one observed message in a thread."""
    ts = NOW - timedelta(days=days_ago)
    tag = tag or f"{thread}:{direction}:{days_ago}"
    return MemoryRecordView(
        memory_id=uuid.uuid5(NS, f"mem:{tag}"),
        org_id=ORG,
        entity_type=EntityType.THREAD.value,
        entity_id=uuid.uuid5(NS, f"ent:thread:{thread}"),
        event_timestamp=ts,
        payload={
            "thread_id": thread,
            "subject": subject,
            "message_id": f"m-{tag}",
            "rfc_message_id": f"<{tag}@mail>",
            "direction": direction,
            "from_email": from_email,
            "from_name": from_name,
            "category": category,
            "is_automated": is_automated,
        },
        source_signal_event_id=uuid.uuid5(NS, f"sig:{tag}"),
        schema_version=1,
        confidence=0.9,
    )


def _waiting(*records) -> list:
    state = build_projection(ORG, list(records), as_of=NOW)
    return [
        s for s in run_reasoning_v2(state)
        if s.signal_type is SignalType.WAITING_ON_USER_REPLY
    ]


# ── The core case ──────────────────────────────────────────────────────


def test_inbound_with_no_reply_is_detected():
    """John emails Monday, it is now Thursday, no reply exists."""
    signals = _waiting(_msg("t-1", days_ago=4, direction="inbound"))

    assert len(signals) == 1
    s = signals[0]
    assert "John Carter" in s.title
    assert "waiting for your reply" in s.title
    assert s.evidence["person_email"] == "john@partner.com"
    assert s.evidence["thread_id"] == "t-1"
    assert s.evidence["subject"] == "Contract review"
    assert s.evidence["waiting_days"] == 4.0
    assert s.evidence["last_inbound_at"]
    assert s.evidence["rfc_message_id"] == "<t-1:inbound:4@mail>"


def test_the_signal_explains_itself():
    """A nudge the founder cannot audit is worse than no nudge."""
    (s,) = _waiting(_msg("t-1", days_ago=4, direction="inbound"))
    assert "Contract review" in s.why
    assert "4 days ago" in s.why
    assert "no reply from you" in s.why


# ── Replies stop it ────────────────────────────────────────────────────


def test_inbound_followed_by_the_users_reply_is_NOT_detected():
    signals = _waiting(
        _msg("t-1", days_ago=4, direction="inbound"),
        _msg("t-1", days_ago=3, direction="outbound", from_email="founder@acme.com"),
    )
    assert signals == []


def test_a_reply_after_several_days_clears_it_with_no_teardown_path():
    """The outbound record simply folds last and flips `direction`. There is no
    "close the signal" step that could be forgotten."""
    inbound = _msg("t-1", days_ago=9, direction="inbound")
    assert len(_waiting(inbound)) == 1

    reply = _msg("t-1", days_ago=1, direction="outbound", from_email="founder@acme.com")
    assert _waiting(inbound, reply) == []


def test_they_wrote_again_after_your_reply_and_are_waiting_again():
    """Ping → reply → ping. The newest message is theirs, so they ARE waiting."""
    signals = _waiting(
        _msg("t-1", days_ago=10, direction="inbound", tag="a"),
        _msg("t-1", days_ago=8, direction="outbound", from_email="founder@acme.com", tag="b"),
        _msg("t-1", days_ago=3, direction="inbound", tag="c"),
    )
    assert len(signals) == 1
    assert signals[0].evidence["waiting_days"] == 3.0


def test_multiple_inbound_messages_use_the_LATEST_as_the_waiting_clock():
    """Two nudges from them: they have been waiting since the LAST one, not the
    first. Overstating the wait would make the card wrong in the user's face."""
    (s,) = _waiting(
        _msg("t-1", days_ago=9, direction="inbound", tag="a"),
        _msg("t-1", days_ago=3, direction="inbound", tag="b"),
    )
    assert s.evidence["waiting_days"] == 3.0


# ── Windows ────────────────────────────────────────────────────────────


def test_too_recent_is_not_a_lapse():
    """Same-day silence is a normal working rhythm, not something to nag about."""
    assert _waiting(_msg("t-1", days_ago=0.5, direction="inbound")) == []


@pytest.mark.parametrize("days", [WAITING_MIN_DAYS, WAITING_MIN_DAYS + 0.5, 20.0])
def test_inside_the_actionable_window_fires(days):
    assert len(_waiting(_msg("t-1", days_ago=days, direction="inbound"))) == 1


def test_beyond_a_month_is_history_not_a_nudge():
    """Matches the other risk engines' MAX_SURFACE_AGE_DAYS ceiling: a founder
    can neither salvage nor quickly act on a six-week-old thread."""
    assert _waiting(_msg("t-1", days_ago=45, direction="inbound")) == []


# ── Exclusions ─────────────────────────────────────────────────────────


def test_automated_email_is_excluded():
    assert _waiting(
        _msg("t-1", days_ago=4, direction="inbound", is_automated=True)
    ) == []


@pytest.mark.parametrize(
    "category", ["promotional", "social", "forum", "update", "automated", "spam"]
)
def test_non_conversational_categories_are_excluded(category):
    """Reuses the EXISTING classifier rather than inventing a second notion of
    "is this a real person"."""
    assert _waiting(
        _msg("t-1", days_ago=4, direction="inbound", category=category)
    ) == []


@pytest.mark.parametrize(
    "sender",
    [
        "no-reply@stripe.com",
        "noreply@github.com",
        "notifications@slack.com",
        "mailer-daemon@google.com",
        "billing@vendor.com",
        "newsletter@substack.com",
    ],
)
def test_no_reply_senders_are_excluded(sender):
    """The residue that slips past category + is_automated: a transactional
    sender that looks personal. You cannot reply to these."""
    assert _waiting(
        _msg("t-1", days_ago=4, direction="inbound", from_email=sender)
    ) == []


def test_a_thread_with_no_subject_is_skipped():
    """Nothing renderable, so nothing actionable."""
    assert _waiting(_msg("t-1", days_ago=4, direction="inbound", subject="")) == []


def test_a_thread_with_no_sender_is_skipped():
    assert _waiting(_msg("t-1", days_ago=4, direction="inbound", from_email="")) == []


# ── Multiple threads ───────────────────────────────────────────────────


def test_each_thread_is_evaluated_independently():
    signals = _waiting(
        _msg("t-1", days_ago=4, direction="inbound", from_email="john@partner.com",
             from_name="John", tag="t1"),
        _msg("t-2", days_ago=6, direction="inbound", from_email="mia@customer.com",
             from_name="Mia", tag="t2"),
        # Answered — must not appear.
        _msg("t-3", days_ago=5, direction="inbound", tag="t3a"),
        _msg("t-3", days_ago=2, direction="outbound", from_email="founder@acme.com", tag="t3b"),
    )
    waiting_by_person = {s.evidence["person_email"]: s for s in signals}
    assert set(waiting_by_person) == {"john@partner.com", "mia@customer.com"}
    assert waiting_by_person["mia@customer.com"].evidence["waiting_days"] == 6.0


def test_longer_waits_score_higher():
    """Severity climbs with the wait so the board ranks the worst first."""
    (short,) = _waiting(_msg("t-1", days_ago=3, direction="inbound"))
    (long_,) = _waiting(_msg("t-2", days_ago=20, direction="inbound", tag="l"))
    assert long_.severity > short.severity


def test_confidence_reflects_observation_not_age():
    """The underlying fact is observed, not inferred, so confidence does not
    drift with how long they have waited."""
    (a,) = _waiting(_msg("t-1", days_ago=3, direction="inbound"))
    (b,) = _waiting(_msg("t-2", days_ago=20, direction="inbound", tag="l"))
    assert a.confidence == b.confidence == 0.9


# ── Determinism / idempotency ──────────────────────────────────────────


def test_the_same_state_yields_byte_identical_signals():
    """Replay-safety: the reasoning pass is pure."""
    records = [_msg("t-1", days_ago=4, direction="inbound")]
    state = build_projection(ORG, records, as_of=NOW)
    a = [s.to_dict() for s in run_reasoning_v2(state)]
    b = [s.to_dict() for s in run_reasoning_v2(state)]
    assert a == b


def test_a_duplicate_sync_does_not_duplicate_the_signal():
    """Re-ingesting the same message (full resync, replay) folds onto the same
    thread entity, so exactly one signal comes out."""
    m = _msg("t-1", days_ago=4, direction="inbound")
    signals = _waiting(m, m)
    assert len(signals) == 1


def test_two_messages_one_thread_produce_one_signal_not_two():
    """dedup_key is the thread, so a chatty thread is one nudge, not five."""
    signals = _waiting(
        _msg("t-1", days_ago=6, direction="inbound", tag="a"),
        _msg("t-1", days_ago=4, direction="inbound", tag="b"),
        _msg("t-1", days_ago=3, direction="inbound", tag="c"),
    )
    assert len(signals) == 1
    # One signal per THREAD entity, not per message.
    assert signals[0].entity_type == "thread"
    assert signals[0].entity_id


def test_the_signal_id_is_stable_across_passes():
    """A stable signal id is what makes the downstream decision id — and so the
    ExecutionRequest — idempotent, so re-running the brief cannot pile up
    duplicate candidates for the same waiting person."""
    records = [_msg("t-1", days_ago=4, direction="inbound")]
    first = _waiting(*records)[0]
    second = _waiting(*records)[0]
    assert first.signal_id == second.signal_id


# ── Signal -> Decision -> ActionCandidate (the full proactive path) ────


def test_the_signal_becomes_a_reply_candidate_with_no_sendable_body():
    """End of the pipeline: Signals -> Reasoning -> Decision -> ActionCandidate.

    The candidate must identify WHO and WHY, and must NOT be a blank email.
    """
    from navis.decisions.base import DecisionCategory, category_for
    from navis.decisions.what_matters_today import WhatMattersTodayEngine
    from navis.execution.models import ActionType
    from navis.execution.proposer import propose_from_decisions

    assert category_for(SignalType.WAITING_ON_USER_REPLY) is DecisionCategory.WAITING_REPLY

    signals = _waiting(_msg("t-1", days_ago=4, direction="inbound"))
    decisions = WhatMattersTodayEngine().decide(
        signals,
        org_id=ORG,
        reasoning_version=1,
        state_watermark=None,
        as_of=NOW,
    )
    waiting = [d for d in decisions if d.category is DecisionCategory.WAITING_REPLY]
    assert len(waiting) == 1

    proposed = propose_from_decisions([d.to_dict() for d in waiting], ORG)
    assert len(proposed) == 1
    p = proposed[0]

    assert p.action_type == ActionType.REPLY_TO_EMAIL.value
    assert p.payload["to"] == ["john@partner.com"]
    assert p.payload["reply_thread_id"] == "t-1"
    # The load-bearing assertion for Phase 1: proposed, but NOT sendable.
    assert p.payload["requires_draft"] is True
    assert not str(p.payload.get("body") or "").strip()
    # And it carries the evidence the card needs to justify the nudge.
    assert p.payload["waiting_days"] == 4.0
    assert p.payload["subject"] == "Contract review"
    assert p.payload["person_name"] == "John Carter"


def test_a_waiting_decision_without_a_thread_proposes_nothing():
    """A candidate that cannot identify what to reply TO is noise."""
    from navis.execution.proposer import propose_from_decisions

    decision = {
        "decision_id": str(uuid.uuid4()),
        "category": "waiting_reply",
        "title": "Someone is waiting",
        "evidence": {"person_email": "john@partner.com"},  # no thread_id
        "reasoning_signal_ids": [],
        "reasoning_version": 1,
        "state_watermark": None,
    }
    assert propose_from_decisions([decision], ORG) == []
