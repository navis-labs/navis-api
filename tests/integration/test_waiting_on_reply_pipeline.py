"""INTEGRATION — a raw Gmail message becomes a proactive reply candidate.

Every layer here is the real one. No mocked extractor, no hand-built projection,
no faked signal type:

    Gmail payload
      -> ThreadExtractor            (real — derives direction from the SENT label)
      -> memory record identity     (real — navis.memory.ids)
      -> State V2 projection        (real — build_projection)
      -> Reasoning V2               (real — the full engine registry)
      -> Decision                   (real — WhatMattersTodayEngine)
      -> Execution proposer         (real — propose_from_decisions)
      -> provider validation        (real — ReplyToEmailProvider)

The unit tests around each layer all pass with hand-written `direction` fields.
This one proves the layers actually AGREE — that the key ThreadExtractor writes
is the key State folds, is the field the engine reads, is the evidence the
proposer needs. A contract drift anywhere in that chain shows up here and
nowhere else.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from navis.decisions.base import DecisionCategory
from navis.decisions.what_matters_today import WhatMattersTodayEngine
from navis.execution.models import ActionType
from navis.execution.providers.base import ProviderValidationError
from navis.execution.providers.reply import ReplyToEmailProvider
from navis.execution.proposer import propose_from_decisions
from navis.memory.extractors.email import ThreadExtractor
from navis.memory.ids import entity_id, memory_record_id
from navis.reasoning.v2 import run_reasoning_v2
from navis.reasoning.v2.contract import SignalType
from navis.signals.base import SignalEventType, SignalSource
from navis.memory.extractors.base import SignalView
from navis.state.v2.builders import MemoryRecordView, build_projection

ORG = uuid.UUID("11111111-1111-1111-1111-111111111111")
NOW = datetime(2026, 8, 9, 9, 0, tzinfo=timezone.utc)
FOUNDER = "founder@navis.ai"


def _gmail_signal(
    *,
    message_id: str,
    days_ago: float,
    sender: str,
    sender_name: str,
    to: str,
    subject: str,
    thread_id: str,
    labels: list[str],
    body: str = "Could you take a look and let me know?",
) -> SignalView:
    """A Gmail message exactly as the connector normalises it."""
    return SignalView(
        event_id=uuid.uuid5(uuid.NAMESPACE_URL, f"sig:{message_id}"),
        org_id=ORG,
        user_id=uuid.uuid4(),
        integration_id=uuid.uuid4(),
        source=SignalSource.GMAIL,
        event_type=SignalEventType.EMAIL_RECEIVED,
        external_id=message_id,
        external_version="1",
        event_timestamp=NOW - timedelta(days=days_ago),
        payload={
            "from": f"{sender_name} <{sender}>",
            "to": to,
            "subject": subject,
            "thread_id": thread_id,
            "sender_email": sender,
            "sender_name": sender_name,
            "to_emails": [to],
            "body_text": body,
            "rfc_message_id": f"<{message_id}@mail.gmail.com>",
            "label_ids": labels,
        },
        metadata={"thread_id": thread_id, "label_ids": labels},
    )


def _to_records(signals: list[SignalView]) -> list[MemoryRecordView]:
    """Run the REAL extractor and derive the REAL memory identity."""
    extractor = ThreadExtractor()
    out: list[MemoryRecordView] = []
    for sig in signals:
        for draft in extractor.extract(sig):
            out.append(
                MemoryRecordView(
                    memory_id=memory_record_id(
                        sig.org_id,
                        sig.event_id,
                        extractor.name,
                        draft.entity_type.value,
                        draft.dedup_key,
                    ),
                    org_id=sig.org_id,
                    entity_type=draft.entity_type.value,
                    entity_id=entity_id(
                        sig.org_id, draft.entity_type.value, draft.entity_natural_key
                    ),
                    event_timestamp=sig.event_timestamp,
                    payload=draft.payload,
                    source_signal_event_id=sig.event_id,
                    schema_version=draft.schema_version,
                    confidence=draft.confidence,
                )
            )
    return out


def _run(signals: list[SignalView]):
    """Signals -> ... -> proposed actions. Returns (waiting_signals, proposals)."""
    state = build_projection(ORG, _to_records(signals), as_of=NOW)
    reasoning = run_reasoning_v2(state)
    waiting = [
        s for s in reasoning if s.signal_type is SignalType.WAITING_ON_USER_REPLY
    ]
    decisions = WhatMattersTodayEngine().decide(
        reasoning, org_id=ORG, reasoning_version=1, state_watermark=None, as_of=NOW
    )
    waiting_decisions = [
        d for d in decisions if d.category is DecisionCategory.WAITING_REPLY
    ]
    proposals = propose_from_decisions([d.to_dict() for d in waiting_decisions], ORG)
    return waiting, proposals


_INBOUND = dict(
    message_id="m-in-1",
    sender="john@partner.com",
    sender_name="John Carter",
    to=FOUNDER,
    subject="Contract review",
    thread_id="thread-77",
    labels=["INBOX", "CATEGORY_PERSONAL", "UNREAD"],
)


async def test_unanswered_email_becomes_an_approvable_reply_candidate():
    """The whole Phase 1B story in one test."""
    waiting, proposals = _run([_gmail_signal(days_ago=4, **_INBOUND)])

    # Reasoning noticed.
    assert len(waiting) == 1
    assert "John Carter is waiting for your reply" == waiting[0].title
    assert waiting[0].evidence["waiting_days"] == 4.0

    # It became exactly one candidate, targeting the right thread and person.
    assert len(proposals) == 1
    p = proposals[0]
    assert p.action_type == ActionType.REPLY_TO_EMAIL.value
    assert p.payload["to"] == ["john@partner.com"]
    assert p.payload["reply_thread_id"] == "thread-77"
    assert p.payload["in_reply_to_message_id"] == "<m-in-1@mail.gmail.com>"
    assert p.payload["subject"] == "Contract review"
    assert p.payload["person_name"] == "John Carter"
    assert p.payload["waiting_days"] == 4.0

    # And it is NOT sendable — Navis proposed who to reply to, not what to say.
    assert p.payload["requires_draft"] is True
    with pytest.raises(ProviderValidationError, match="no draft"):
        await ReplyToEmailProvider().validate(p.payload)


async def test_the_founders_reply_removes_the_candidate_entirely():
    """The reply is an ordinary Gmail message carrying the SENT label. Nothing
    "closes" the signal — the newer outbound record simply folds last."""
    signals = [
        _gmail_signal(days_ago=6, **_INBOUND),
        _gmail_signal(
            message_id="m-out-1",
            # Deliberately OUTSIDE the min-wait window (>= WAITING_MIN_DAYS old).
            # With a 1-day-old reply this test passed even when SENT detection
            # was broken — the thread was simply too recent to fire, so it was
            # asserting the window, not the reply. At 3 days the ONLY thing
            # keeping the signal quiet is that the founder answered.
            days_ago=3,
            sender=FOUNDER,
            sender_name="Founder",
            to="john@partner.com",
            subject="Re: Contract review",
            thread_id="thread-77",
            labels=["SENT"],
            body="Looks good — sending comments today.",
        ),
    ]
    waiting, proposals = _run(signals)
    assert waiting == []
    assert proposals == []


async def test_a_newsletter_never_becomes_a_candidate():
    """Excluded by the EXISTING classifier, not a second bespoke rule."""
    waiting, proposals = _run([
        _gmail_signal(
            message_id="m-news-1",
            days_ago=5,
            sender="digest@substack.com",
            sender_name="Some Newsletter",
            to=FOUNDER,
            subject="Your weekly digest",
            thread_id="thread-news",
            labels=["INBOX", "CATEGORY_PROMOTIONS"],
            body="Here are this week's top stories.",
        )
    ])
    assert waiting == []
    assert proposals == []


async def test_several_threads_produce_one_candidate_each():
    waiting, proposals = _run([
        _gmail_signal(days_ago=4, **_INBOUND),
        _gmail_signal(
            message_id="m-in-2",
            days_ago=6,
            sender="mia@customer.com",
            sender_name="Mia Chen",
            to=FOUNDER,
            subject="Renewal question",
            thread_id="thread-88",
            labels=["INBOX", "CATEGORY_PERSONAL"],
        ),
        # Already answered — must not appear.
        _gmail_signal(
            message_id="m-in-3", days_ago=5, sender="sam@vendor.com",
            sender_name="Sam", to=FOUNDER, subject="Invoice query",
            thread_id="thread-99", labels=["INBOX", "CATEGORY_PERSONAL"],
        ),
        _gmail_signal(
            message_id="m-out-3", days_ago=2, sender=FOUNDER, sender_name="Founder",
            to="sam@vendor.com", subject="Re: Invoice query",
            thread_id="thread-99", labels=["SENT"],
        ),
    ])
    by_person = {p.payload["person_email"] for p in proposals}
    assert by_person == {"john@partner.com", "mia@customer.com"}
    assert len(waiting) == 2


async def test_a_duplicate_sync_does_not_duplicate_the_candidate():
    """Re-ingesting the same message (full resync / replay) must converge.

    The memory record id is deterministic, so the duplicate folds onto the same
    thread entity and yields the same signal, decision and idempotency key.
    """
    sig = _gmail_signal(days_ago=4, **_INBOUND)
    once_waiting, once = _run([sig])
    twice_waiting, twice = _run([sig, sig])

    assert len(twice_waiting) == 1
    assert len(twice) == 1
    assert once[0].decision_id == twice[0].decision_id
    assert once[0].payload == twice[0].payload
    assert once_waiting[0].signal_id == twice_waiting[0].signal_id


async def test_re_running_the_brief_reuses_the_same_execution_identity():
    """Two brief runs over unchanged state must not pile up duplicate cards.

    The identity chain is signal_id -> decision_id -> idempotency_key, so a
    stable signal id is what makes the whole proactive queue idempotent.
    """
    from navis.execution.service import ExecutionService
    from navis.execution.models import ExecutionOrigin
    from tests.execution.fakes import FakeExecutionRepo

    sig = _gmail_signal(days_ago=4, **_INBOUND)
    repo = FakeExecutionRepo()
    svc = ExecutionService(repo=repo)

    created_flags = []
    for _ in range(2):
        _, proposals = _run([sig])
        for p in proposals:
            _req, created = await svc.request(
                p.org_id, p.decision_id, p.action_type, p.payload,
                reasoning_signal_ids=p.reasoning_signal_ids,
                reasoning_version=p.reasoning_version,
                state_watermark=p.state_watermark,
                origin=ExecutionOrigin.PROACTIVE,
            )
            created_flags.append(created)

    assert created_flags == [True, False]
    assert len(repo.rows) == 1


async def test_the_candidate_is_proactive_origin_so_chat_never_shows_it():
    from navis.execution.service import ExecutionService
    from navis.execution.models import ExecutionOrigin
    from tests.execution.fakes import FakeExecutionRepo

    _, proposals = _run([_gmail_signal(days_ago=4, **_INBOUND)])
    repo = FakeExecutionRepo()
    req, _ = await ExecutionService(repo=repo).request(
        proposals[0].org_id, proposals[0].decision_id, proposals[0].action_type,
        proposals[0].payload, reasoning_signal_ids=[], reasoning_version=None,
        state_watermark=None, origin=ExecutionOrigin.PROACTIVE,
    )
    assert req.origin == ExecutionOrigin.PROACTIVE.value
