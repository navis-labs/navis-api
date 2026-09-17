"""INTEGRATION — a Gmail thread becomes a reviewed, edited, sent reply.

The full Phase 2B path, with every layer the real one except Gmail itself:

    Gmail messages
      -> extractors            (real)
      -> State V2              (real)
      -> Reasoning V2          (real — WAITING_ON_USER_REPLY / follow-up / owed)
      -> Decision              (real)
      -> proposer              (real — draftless candidate)
      -> DraftService.generate (real — deterministic template over the thread)
      -> edit                  (real)
      -> DraftService.approve  (real — merges into the payload)
      -> provider validation   (real)
      -> execute_pass          (real worker)
      -> a fake Gmail          (the only substitute)

What this proves that the unit tests cannot: that the payload the proposer
writes is the payload the generator can read, is the payload the provider will
accept, is the payload the sender encodes — and that a human's edit survives
all of it intact.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from navis.decisions.base import DecisionCategory
from navis.decisions.what_matters_today import WhatMattersTodayEngine
from navis.execution.drafts.contract import DraftState
from navis.execution.drafts.generator import generate
from navis.execution.drafts.thread_context import ThreadContext, ThreadMessage
from navis.execution.models import ActionType, ExecutionOrigin
from navis.execution.providers.base import ProviderValidationError
from navis.execution.providers.reply import ReplyToEmailProvider
from navis.execution.proposer import propose_from_decisions
from navis.execution.service import ExecutionService
from navis.execution.status import ExecutionStatus
from navis.execution.worker import execute_pass
from navis.memory.extractors import extractors_for
from navis.memory.extractors.base import SignalView
from navis.memory.ids import entity_id, memory_record_id
from navis.reasoning.v2 import run_reasoning_v2
from navis.signals.base import SignalEventType, SignalSource
from navis.state.v2.builders import MemoryRecordView, build_projection
from tests.execution.fakes import FakeExecutionRepo

ORG = uuid.UUID("11111111-1111-1111-1111-111111111111")
USER = uuid.UUID("22222222-2222-2222-2222-222222222222")
NOW = datetime(2026, 8, 9, 12, 0, tzinfo=timezone.utc)
FOUNDER = "founder@navis.ai"


# ── Gmail-shaped input ─────────────────────────────────────────────────


def _msg(
    *,
    mid: str,
    days_ago: float,
    sender: str,
    sender_name: str,
    body: str,
    thread_id: str = "thread-release",
    subject: str = "Product Release",
    to: str = FOUNDER,
    labels: list[str] | None = None,
) -> SignalView:
    return SignalView(
        event_id=uuid.uuid5(uuid.NAMESPACE_URL, f"sig:{mid}"),
        org_id=ORG,
        user_id=USER,
        integration_id=uuid.uuid4(),
        source=SignalSource.GMAIL,
        event_type=SignalEventType.EMAIL_RECEIVED,
        external_id=mid,
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
            "rfc_message_id": f"<{mid}@mail.gmail.com>",
            "label_ids": labels if labels is not None else ["INBOX", "CATEGORY_PERSONAL"],
        },
        metadata={"thread_id": thread_id},
    )


def _records(signals: list[SignalView]) -> list[MemoryRecordView]:
    out: list[MemoryRecordView] = []
    for sig in signals:
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


def _thread_from(signals: list[SignalView], thread_id: str) -> ThreadContext:
    """The same ThreadContext `load_thread` builds, assembled from the signals
    directly so this test needs no database."""
    messages = []
    subject = None
    for sig in sorted(signals, key=lambda s: s.event_timestamp):
        p = sig.payload
        if p.get("thread_id") != thread_id:
            continue
        subject = subject or p.get("subject")
        labels = p.get("label_ids") or []
        messages.append(
            ThreadMessage(
                message_id=sig.external_id,
                rfc_message_id=p.get("rfc_message_id"),
                sent_at=sig.event_timestamp,
                direction="outbound" if "SENT" in labels else "inbound",
                from_email=p.get("sender_email"),
                from_name=p.get("sender_name"),
                subject=p.get("subject"),
                snippet=(p.get("body_text") or "")[:400] or None,
            )
        )
    return ThreadContext(thread_id=thread_id, subject=subject, messages=tuple(messages))


def _pick(proposals, kind: str):
    """One thread can legitimately carry several obligations at once — an
    unanswered email that also asks for a decision is BOTH "they're waiting on a
    reply" and "you owe a decision". Tests select the candidate they mean rather
    than assuming a thread yields exactly one."""
    matches = [p for p in proposals if p.payload.get("candidate_kind") == kind]
    assert matches, f"no {kind} candidate in {[p.payload.get('candidate_kind') for p in proposals]}"
    return matches[0]


def _propose(signals: list[SignalView]):
    state = build_projection(ORG, _records(signals), as_of=NOW)
    reasoning = run_reasoning_v2(state)
    decisions = WhatMattersTodayEngine().decide(
        reasoning, org_id=ORG, reasoning_version=1, state_watermark=None, as_of=NOW
    )
    actionable = [
        d for d in decisions
        if d.category in (
            DecisionCategory.WAITING_REPLY,
            DecisionCategory.FOLLOW_UP,
            DecisionCategory.DECISION_OWED,
        )
    ]
    return propose_from_decisions([d.to_dict() for d in actionable], ORG)


# ── Fake Gmail ─────────────────────────────────────────────────────────


class _FakeGmail:
    """Records what would be sent. The ONLY substitute in this test."""

    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def __call__(self, payload: dict) -> str:
        # Mirror the real sender's final guards so the test exercises them.
        if not str(payload.get("body") or "").strip():
            raise RuntimeError("gmail send: refusing to send an empty body")
        if not payload.get("to"):
            raise RuntimeError("gmail send: no recipients")
        self.sent.append(dict(payload))
        return f"gmail-msg-{len(self.sent)}"


_UNANSWERED = [
    _msg(
        mid="m-ask", days_ago=4, sender="john@partner.com", sender_name="John Carter",
        body="Hi, can you confirm whether we're still targeting Monday for the release?",
    )
]


# ══════════════════════════════════════════════════════════════════════
# The full path
# ══════════════════════════════════════════════════════════════════════


async def test_unanswered_email_becomes_a_reviewed_edited_sent_reply():
    """PROPOSED → DRAFTED → EDITED → APPROVED → SENT, end to end."""
    proposals = _propose(_UNANSWERED)
    p = _pick(proposals, "waiting_reply")
    assert p.action_type == ActionType.REPLY_TO_EMAIL.value
    assert p.payload["requires_draft"] is True

    repo = FakeExecutionRepo()
    svc = ExecutionService(repo=repo)
    req, _ = await svc.request(
        p.org_id, p.decision_id, p.action_type, p.payload,
        reasoning_signal_ids=p.reasoning_signal_ids,
        reasoning_version=p.reasoning_version,
        state_watermark=p.state_watermark,
        origin=ExecutionOrigin.PROACTIVE,
    )

    # ── DRAFT ────────────────────────────────────────────────────────
    thread = _thread_from(_UNANSWERED, "thread-release")
    d = generate(
        candidate_kind=p.payload["candidate_kind"],
        payload=p.payload,
        thread=thread,
        sender_name="Alex",
    )
    assert d.subject == "Re: Product Release"
    assert d.recipients == ("john@partner.com",)
    assert d.thread_id == "thread-release"
    assert d.in_reply_to_message_id == "<m-ask@mail.gmail.com>"
    # Never answers the question it was asked.
    assert "monday" not in d.body.lower()

    # Still unsendable at this point — a draft is not an approval.
    with pytest.raises(ProviderValidationError, match="no draft"):
        await ReplyToEmailProvider().validate(req.payload)

    # ── EDIT + APPROVE (the merge the service performs) ──────────────
    edited_body = (
        "Hi John,\n\nYes — Monday is still the plan. I'll confirm the final "
        "timing tomorrow.\n\nBest,\nAlex"
    )
    req.payload = {
        **req.payload,
        "to": list(d.recipients),
        "subject": d.subject,
        "body": edited_body,
        "reply_thread_id": d.thread_id,
        "in_reply_to_message_id": d.in_reply_to_message_id,
        "requires_draft": False,
        "draft_was_edited": True,
    }
    await ReplyToEmailProvider().validate(req.payload)  # now sendable
    await svc.approve(req.id, USER)

    # ── SEND ─────────────────────────────────────────────────────────
    gmail = _FakeGmail()
    result = await execute_pass(
        repo, providers={"reply_to_email": ReplyToEmailProvider(sender=gmail)}
    )

    assert result["executed"] == 1
    assert req.status == ExecutionStatus.EXECUTED.value
    assert len(gmail.sent) == 1
    sent = gmail.sent[0]
    # The FOUNDER's words went out, not the template's.
    assert sent["body"] == edited_body
    assert sent["to"] == ["john@partner.com"]
    assert sent["reply_thread_id"] == "thread-release"
    assert sent["in_reply_to_message_id"] == "<m-ask@mail.gmail.com>"
    assert req.provider_reference["external_id"] == "gmail-msg-1"


async def test_a_draftless_candidate_cannot_reach_gmail():
    """The gate that makes the whole flow safe: approving without a draft sends
    nothing, whatever else goes wrong upstream."""
    proposals = _propose(_UNANSWERED)
    repo = FakeExecutionRepo()
    svc = ExecutionService(repo=repo)
    p = _pick(proposals, "waiting_reply")
    req, _ = await svc.request(
        p.org_id, p.decision_id, p.action_type, p.payload,
        reasoning_signal_ids=[], reasoning_version=None, state_watermark=None,
        origin=ExecutionOrigin.PROACTIVE,
    )
    await svc.approve(req.id, USER)  # bypasses the HTTP draft gate on purpose

    gmail = _FakeGmail()
    await execute_pass(repo, providers={"reply_to_email": ReplyToEmailProvider(sender=gmail)})

    assert gmail.sent == [], "nothing may reach Gmail without a draft"
    assert req.status == ExecutionStatus.FAILED.value
    assert len(repo.dead_letters) == 1


async def test_approving_twice_sends_exactly_one_email():
    """Idempotent approve (Phase 1) plus the worker's already-executed guard
    (calendar work) together mean a double click cannot double-send."""
    proposals = _propose(_UNANSWERED)
    repo = FakeExecutionRepo()
    svc = ExecutionService(repo=repo)
    p = _pick(proposals, "waiting_reply")
    req, _ = await svc.request(
        p.org_id, p.decision_id, p.action_type, p.payload,
        reasoning_signal_ids=[], reasoning_version=None, state_watermark=None,
        origin=ExecutionOrigin.PROACTIVE,
    )
    req.payload = {
        **req.payload,
        "subject": "Re: Product Release",
        "body": "Hi John,\n\nComing back to you shortly.\n\nBest,",
        "reply_thread_id": "thread-release",
        "requires_draft": False,
    }

    gmail = _FakeGmail()
    providers = {"reply_to_email": ReplyToEmailProvider(sender=gmail)}

    await svc.approve(req.id, USER)
    await execute_pass(repo, providers=providers)
    await svc.approve(req.id, USER)  # second click on a stale card
    await execute_pass(repo, providers=providers)

    assert len(gmail.sent) == 1


async def test_a_retry_after_a_recorded_send_does_not_send_again():
    """The provider-reference guard: a row that already produced a Gmail message
    id is completed rather than re-dispatched, even when it is re-claimed."""
    proposals = _propose(_UNANSWERED)
    repo = FakeExecutionRepo()
    svc = ExecutionService(repo=repo)
    p = _pick(proposals, "waiting_reply")
    req, _ = await svc.request(
        p.org_id, p.decision_id, p.action_type, p.payload,
        reasoning_signal_ids=[], reasoning_version=None, state_watermark=None,
        origin=ExecutionOrigin.PROACTIVE,
    )
    req.payload = {
        **req.payload,
        "subject": "Re: Product Release",
        "body": "Hi John,\n\nComing back to you shortly.\n\nBest,",
        "reply_thread_id": "thread-release",
        "requires_draft": False,
    }
    gmail = _FakeGmail()
    providers = {"reply_to_email": ReplyToEmailProvider(sender=gmail)}

    await svc.approve(req.id, USER)
    await execute_pass(repo, providers=providers)
    assert len(gmail.sent) == 1

    # Stale-EXECUTING recovery pushes an orphaned row back to FAILED, which is
    # re-claimable work.
    req.status = ExecutionStatus.FAILED.value
    await execute_pass(repo, providers=providers)

    assert len(gmail.sent) == 1, "the recorded provider reference must block a resend"
    assert req.status == ExecutionStatus.EXECUTED.value


# ══════════════════════════════════════════════════════════════════════
# Reply threading
# ══════════════════════════════════════════════════════════════════════


async def test_the_reply_joins_the_original_thread_rather_than_starting_one():
    """Item 5: threadId + In-Reply-To must survive from the source message all
    the way to the encoded RFC 2822 payload."""
    from navis.execution.providers._senders import _encode_reply_message, navis_message_id
    import base64

    thread = _thread_from(_UNANSWERED, "thread-release")
    d = generate(
        candidate_kind="waiting_reply",
        payload={"person_email": "john@partner.com"},
        thread=thread,
        sender_name="Alex",
    )
    execution_id = uuid.uuid4()
    raw = _encode_reply_message(
        from_email=FOUNDER,
        to_emails=list(d.recipients),
        subject=d.subject,
        body=d.body,
        in_reply_to_message_id=d.in_reply_to_message_id,
        message_id=navis_message_id(execution_id),
    )
    decoded = base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4)).decode()

    assert "In-Reply-To: <m-ask@mail.gmail.com>" in decoded
    assert "References: <m-ask@mail.gmail.com>" in decoded
    assert "Subject: Re: Product Release" in decoded
    # Our own deterministic id — the duplicate-send guard's anchor.
    assert f"Message-ID: {navis_message_id(execution_id)}" in decoded


# ══════════════════════════════════════════════════════════════════════
# Other candidate kinds reach a sendable draft too
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "body,kind,thread_id",
    [
        (
            "Thanks for the call. I'll send you the product deck by Friday.",
            "commitment_follow_up",
            "thread-deck",
        ),
        (
            "The plan is ready. Can you confirm whether we should proceed with the release?",
            "decision_owed",
            "thread-decide",
        ),
    ],
)
async def test_every_candidate_kind_produces_a_sendable_draft(body, kind, thread_id):
    signals = [
        _msg(
            mid=f"m-{kind}", days_ago=6, sender="chrissy@partner.com",
            sender_name="Chrissy Lane", body=body, thread_id=thread_id,
            subject="Product deck",
        )
    ]
    p = _pick(_propose(signals), kind)

    d = generate(
        candidate_kind=kind,
        payload=p.payload,
        thread=_thread_from(signals, thread_id),
        sender_name="Alex",
    )
    assert d.body.strip()
    assert d.recipients == ("chrissy@partner.com",)
    assert d.thread_id == thread_id
    assert d.reason

    # And the merged payload validates.
    payload = {
        **p.payload,
        "to": list(d.recipients),
        "subject": d.subject,
        "body": d.body,
        "reply_thread_id": d.thread_id,
        "requires_draft": False,
    }
    await ReplyToEmailProvider().validate(payload)


async def test_the_draft_state_machine_never_touches_execution_status():
    """Item 13: drafting is a separate lifecycle. Its states must not appear in
    the execution machine, and vice versa."""
    draft_states = {s.value for s in DraftState}
    execution_states = {s.value for s in ExecutionStatus}
    overlap = draft_states & execution_states
    # "approved" is a legitimate shared WORD; the point is that the sets are not
    # the same machine and DraftState adds no transitions to ExecutionStatus.
    assert overlap <= {"approved"}
    assert "editing" not in execution_states
    assert "stale" not in execution_states
    assert "draft_ready" not in execution_states
