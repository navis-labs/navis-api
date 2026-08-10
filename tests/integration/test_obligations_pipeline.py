"""INTEGRATION — raw Gmail messages become follow-up / decision-owed candidates.

Every layer is the real one:

    Gmail payload
      -> text rules            (real — binding vs hedged, request vs record)
      -> due-date parser       (real — resolved against the MESSAGE date)
      -> extractors            (real — commitment, decision-request, thread)
      -> memory identity       (real — navis.memory.ids)
      -> State V2 projection   (real — including the sticky thread markers)
      -> Reasoning V2          (real — full engine registry)
      -> Decision              (real — WhatMattersTodayEngine)
      -> Execution proposer    (real — propose_from_decisions)
      -> provider validation   (real — ReplyToEmailProvider)

The per-layer unit tests all pass with hand-written fields. This proves the
layers AGREE: that the key an extractor writes is the key State folds, is the
field an engine reads, is the evidence the proposer needs.
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
from navis.memory.extractors import extractors_for
from navis.memory.extractors.base import SignalView
from navis.memory.ids import entity_id, memory_record_id
from navis.reasoning.v2 import run_reasoning_v2
from navis.reasoning.v2.contract import SignalType
from navis.signals.base import SignalEventType, SignalSource
from navis.state.v2.builders import MemoryRecordView, build_projection

ORG = uuid.UUID("11111111-1111-1111-1111-111111111111")
NOW = datetime(2026, 8, 9, 12, 0, tzinfo=timezone.utc)  # Sunday
FOUNDER = "founder@navis.ai"


def _msg(
    *,
    message_id: str,
    days_ago: float,
    sender: str,
    sender_name: str,
    body: str,
    thread_id: str,
    subject: str = "Product deck",
    to: str = FOUNDER,
    labels: list[str] | None = None,
) -> SignalView:
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
            "label_ids": labels if labels is not None else ["INBOX", "CATEGORY_PERSONAL"],
        },
        metadata={"thread_id": thread_id},
    )


def _records(signals: list[SignalView]) -> list[MemoryRecordView]:
    """Run the REAL extractor registry and derive REAL memory identity."""
    out: list[MemoryRecordView] = []
    for sig in signals:
        # Iterate the registry directly rather than `run_extractors`, which
        # discards the extractor NAME — and the name is part of the memory
        # identity tuple, so dropping it would make this test's ids diverge
        # from production's.
        for extractor in extractors_for(sig.source):
            for draft in extractor.extract(sig):
                out.append(
                    MemoryRecordView(
                        memory_id=memory_record_id(
                            sig.org_id, sig.event_id, extractor.name,
                            draft.entity_type.value, draft.dedup_key,
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
    """Signals -> ... -> proposals. Returns (reasoning_signals, proposals)."""
    state = build_projection(ORG, _records(signals), as_of=NOW)
    reasoning = run_reasoning_v2(state)
    decisions = WhatMattersTodayEngine().decide(
        reasoning, org_id=ORG, reasoning_version=1, state_watermark=None, as_of=NOW
    )
    actionable = [
        d for d in decisions
        if d.category in (DecisionCategory.FOLLOW_UP, DecisionCategory.DECISION_OWED)
    ]
    return reasoning, propose_from_decisions([d.to_dict() for d in actionable], ORG)


def _of(signals, kind: SignalType):
    return [s for s in signals if s.signal_type is kind]


# ══════════════════════════════════════════════════════════════════════
# Commitment follow-up
# ══════════════════════════════════════════════════════════════════════

# Promised Monday 2026-08-03 12:00. "by Friday" resolves to 2026-08-07
# 23:59:59 — END of the named day, because a promise "by Friday" is kept any
# time that Friday. NOW is Sunday 2026-08-09 12:00, so it is 1.5 days overdue.
_PROMISE = dict(
    message_id="m-promise",
    days_ago=6,
    sender="chrissy@partner.com",
    sender_name="Chrissy Lane",
    body="Thanks for the call. I'll send you the product deck by Friday.",
    thread_id="thread-deck",
)


async def test_an_overdue_promise_becomes_an_approvable_follow_up():
    reasoning, proposals = _run([_msg(**_PROMISE)])

    (sig,) = _of(reasoning, SignalType.COMMITMENT_FOLLOW_UP)
    assert "Chrissy Lane" in sig.title
    # The deadline is the one SHE stated, resolved against HER message date.
    assert sig.evidence["due_text"] == "Friday"
    assert sig.evidence["due_at"].startswith("2026-08-07")
    assert sig.evidence["overdue_days"] == 1.5

    assert len(proposals) == 1
    p = proposals[0]
    assert p.action_type == ActionType.REPLY_TO_EMAIL.value
    assert p.payload["candidate_kind"] == "commitment_follow_up"
    assert p.payload["to"] == ["chrissy@partner.com"]
    assert p.payload["reply_thread_id"] == "thread-deck"
    assert p.payload["due_text"] == "Friday"
    assert "product deck" in p.payload["commitment"]

    # Proposed, never sendable.
    assert p.payload["requires_draft"] is True
    with pytest.raises(ProviderValidationError, match="no draft"):
        await ReplyToEmailProvider().validate(p.payload)


async def test_a_later_delivery_email_makes_the_follow_up_disappear():
    """"Attached is the deck" from the same person, after the promise."""
    _, proposals = _run([
        _msg(**_PROMISE),
        _msg(
            message_id="m-delivered", days_ago=1,
            sender="chrissy@partner.com", sender_name="Chrissy Lane",
            body="Attached is the product deck — sorry for the delay!",
            thread_id="thread-deck",
        ),
    ])
    assert proposals == []


@pytest.mark.parametrize(
    "body",
    [
        "Maybe I'll send the deck on Friday if I get time.",
        "We should send the deck to the investors this week.",
        "Can you send me the deck by Friday?",
        "Great chatting today, looking forward to next steps.",
    ],
)
async def test_non_commitments_never_produce_a_follow_up(body):
    reasoning, proposals = _run([_msg(**{**_PROMISE, "body": body})])
    assert _of(reasoning, SignalType.COMMITMENT_FOLLOW_UP) == []
    assert proposals == []


async def test_a_promise_with_no_date_produces_no_fabricated_deadline():
    """It flows through the undated path instead — never with an invented date."""
    reasoning, proposals = _run([
        _msg(**{**_PROMISE, "body": "I'll send you the product deck."})
    ])
    assert _of(reasoning, SignalType.COMMITMENT_FOLLOW_UP) == []
    assert proposals == []
    # It is still REMEMBERED as a commitment — just not as a dated one.
    slipping = _of(reasoning, SignalType.COMMITMENT_SLIPPING)
    assert len(slipping) == 1
    assert "due" not in slipping[0].evidence


async def test_a_promise_not_yet_due_is_not_chased():
    reasoning, _ = _run([
        _msg(**{**_PROMISE, "days_ago": 0.5,
                "body": "I'll send the product deck by next Friday."})
    ])
    assert _of(reasoning, SignalType.COMMITMENT_FOLLOW_UP) == []


async def test_two_people_owing_two_things_produce_two_candidates():
    _, proposals = _run([
        _msg(**_PROMISE),
        _msg(
            message_id="m-promise-2", days_ago=6,
            sender="sam@vendor.com", sender_name="Sam Reid",
            body="We'll deliver the compliance report by Friday.",
            thread_id="thread-report", subject="Compliance report",
        ),
    ])
    assert {p.payload["person_email"] for p in proposals} == {
        "chrissy@partner.com", "sam@vendor.com"
    }


# ══════════════════════════════════════════════════════════════════════
# Decision owed
# ══════════════════════════════════════════════════════════════════════

_ASK = dict(
    message_id="m-ask",
    days_ago=4,
    sender="mia@team.com",
    sender_name="Mia Chen",
    body="The launch plan is ready. Can you confirm whether we should proceed with the Product Release?",
    thread_id="thread-release",
    subject="Product Release",
)


async def test_a_decision_request_becomes_an_approvable_candidate():
    reasoning, proposals = _run([_msg(**_ASK)])

    (sig,) = _of(reasoning, SignalType.DECISION_OWED)
    assert sig.title == "Decide: Product Release"
    assert sig.evidence["person_email"] == "mia@team.com"
    assert sig.evidence["age_days"] == 4.0

    assert len(proposals) == 1
    p = proposals[0]
    assert p.payload["candidate_kind"] == "decision_owed"
    assert p.payload["reply_thread_id"] == "thread-release"
    assert "proceed with the Product Release" in p.payload["decision_question"]
    assert p.payload["requires_draft"] is True
    with pytest.raises(ProviderValidationError, match="no draft"):
        await ReplyToEmailProvider().validate(p.payload)


async def test_the_founder_replying_removes_the_decision_candidate():
    _, proposals = _run([
        _msg(**_ASK),
        _msg(
            message_id="m-reply", days_ago=1,
            sender=FOUNDER, sender_name="Founder", to="mia@team.com",
            body="Yes, go ahead with Monday.",
            thread_id="thread-release", subject="Re: Product Release",
            labels=["SENT"],
        ),
    ])
    assert proposals == []


async def test_another_participant_recording_the_decision_removes_it():
    """A cofounder answering in-thread closes the question just as well."""
    _, proposals = _run([
        _msg(**_ASK),
        _msg(
            message_id="m-cofounder", days_ago=1,
            sender="cofounder@navis.ai", sender_name="Co Founder",
            body="We have decided to proceed with the release on Monday.",
            thread_id="thread-release", subject="Re: Product Release",
        ),
    ])
    assert proposals == []


@pytest.mark.parametrize(
    "body",
    [
        "Here's the decision we made about the launch date.",
        "FYI, the launch is Monday next week.",
        "Let's discuss this sometime next week.",
        "The deck looks great, thanks for putting it together.",
    ],
)
async def test_informational_email_never_produces_a_decision_candidate(body):
    reasoning, proposals = _run([_msg(**{**_ASK, "body": body})])
    assert _of(reasoning, SignalType.DECISION_OWED) == []
    assert proposals == []


async def test_two_open_questions_produce_two_candidates():
    _, proposals = _run([
        _msg(**_ASK),
        _msg(
            message_id="m-ask-2", days_ago=3,
            sender="raj@team.com", sender_name="Raj Patel",
            body="Should we launch on Monday or Tuesday? Need your call.",
            thread_id="thread-launch", subject="Launch date",
        ),
    ])
    assert {p.payload["person_email"] for p in proposals} == {
        "mia@team.com", "raj@team.com"
    }


# ══════════════════════════════════════════════════════════════════════
# Lifecycle + safety across both
# ══════════════════════════════════════════════════════════════════════


async def test_a_duplicate_sync_produces_no_duplicate_candidates():
    """Full resync / replay must converge: deterministic memory ids fold onto
    the same entities, so the same decision id and idempotency key come out."""
    signals = [_msg(**_PROMISE), _msg(**_ASK)]
    _, once = _run(signals)
    _, twice = _run(signals + signals)

    assert len(once) == 2
    assert len(twice) == 2
    assert {p.decision_id for p in once} == {p.decision_id for p in twice}
    assert sorted(p.payload["candidate_kind"] for p in twice) == [
        "commitment_follow_up", "decision_owed",
    ]


async def test_re_running_the_brief_reuses_the_same_execution_identity():
    from navis.execution.models import ExecutionOrigin
    from navis.execution.service import ExecutionService
    from tests.execution.fakes import FakeExecutionRepo

    repo = FakeExecutionRepo()
    svc = ExecutionService(repo=repo)
    created = []
    for _ in range(2):
        _, proposals = _run([_msg(**_PROMISE), _msg(**_ASK)])
        for p in proposals:
            _req, was_created = await svc.request(
                p.org_id, p.decision_id, p.action_type, p.payload,
                reasoning_signal_ids=p.reasoning_signal_ids,
                reasoning_version=p.reasoning_version,
                state_watermark=p.state_watermark,
                origin=ExecutionOrigin.PROACTIVE,
            )
            created.append(was_created)

    assert created == [True, True, False, False]
    assert len(repo.rows) == 2


async def test_lineage_is_preserved_end_to_end():
    _, proposals = _run([_msg(**_PROMISE)])
    p = proposals[0]
    assert p.org_id == ORG
    assert p.reasoning_signal_ids, "the originating reasoning signal must ride along"
    assert p.reasoning_version == 1
    assert p.payload["thread_id"] == "thread-deck"
    assert p.payload["decision_id"]


async def test_no_blank_email_and_no_fake_task_can_come_out_of_this_path():
    """The Phase 1 safety rails still hold for the Phase 2A producers."""
    _, proposals = _run([_msg(**_PROMISE), _msg(**_ASK)])
    assert proposals
    for p in proposals:
        assert p.action_type == ActionType.REPLY_TO_EMAIL.value
        assert p.action_type != ActionType.SEND_EMAIL.value
        assert p.action_type != ActionType.CREATE_TASK.value
        assert not str(p.payload.get("body") or "").strip()
        assert p.payload["requires_draft"] is True
        with pytest.raises(ProviderValidationError):
            await ReplyToEmailProvider().validate(p.payload)


async def test_automated_mail_produces_neither_candidate():
    _, proposals = _run([
        _msg(**{**_PROMISE, "message_id": "m-bulk", "sender": "digest@substack.com",
                "sender_name": "Newsletter",
                "labels": ["INBOX", "CATEGORY_PROMOTIONS"]}),
        _msg(**{**_ASK, "message_id": "m-bulk-2", "sender": "noreply@vendor.com",
                "sender_name": "Vendor Bot",
                "labels": ["INBOX", "CATEGORY_UPDATES"]}),
    ])
    assert proposals == []
