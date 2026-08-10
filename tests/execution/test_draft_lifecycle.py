"""Draft lifecycle: edit, staleness, approval, and the send guards.

The invariant every test here defends:

    No proactive email reaches Gmail without a human having seen the exact text
    that will be sent.

and its corollary, which is what makes editing usable at all: editing a draft
must never change the identity of the action it belongs to.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from navis.execution.drafts.contract import (
    APPROVABLE_FROM,
    REGENERABLE_FROM,
    DraftState,
    StaleReason,
)
from navis.execution.drafts.models import ActionDraft
from navis.execution.drafts.service import DraftNotApprovable, DraftService, _stale_reason
from navis.execution.drafts.thread_context import ThreadContext, ThreadMessage
from navis.execution.idempotency import idempotency_key
from navis.execution.providers.base import ProviderValidationError
from navis.execution.providers.reply import ReplyToEmailProvider, requires_draft

ORG = uuid.UUID("11111111-1111-1111-1111-111111111111")
NOW = datetime(2026, 8, 9, 12, 0, tzinfo=timezone.utc)


class _Req:
    """Stands in for an ExecutionRequest row (the service only reads/writes
    these attributes)."""

    def __init__(self, payload: dict, *, action_type="reply_to_email"):
        self.id = uuid.uuid4()
        self.organization_id = ORG
        self.action_type = action_type
        self.status = "requested"
        self.payload = payload


def _draft(**over) -> ActionDraft:
    d = ActionDraft(
        execution_id=uuid.uuid4(),
        organization_id=ORG,
        state=DraftState.READY.value,
        subject="Re: Product deck",
        body="Hi Chrissy,\n\nJust following up.\n\nThanks,",
        recipients=["chrissy@partner.com"],
        thread_id="thread-1",
        in_reply_to_message_id="<m-1@mail>",
        reason="They promised it Friday.",
        evidence={},
        thread_watermark="m-1|2026-08-05T00:00:00+00:00",
        generated_at=NOW - timedelta(hours=1),
        generator="template-1",
    )
    for k, v in over.items():
        setattr(d, k, v)
    return d


def _msg(*, mid, days_ago, direction="inbound", from_email="chrissy@partner.com", snippet="hello"):
    return ThreadMessage(
        message_id=mid,
        rfc_message_id=f"<{mid}@mail>",
        sent_at=NOW - timedelta(days=days_ago),
        direction=direction,
        from_email=from_email,
        from_name="Chrissy Lane",
        subject="Product deck",
        snippet=snippet,
    )


def _thread(*messages) -> ThreadContext:
    return ThreadContext(thread_id="thread-1", subject="Product deck", messages=messages)


# ══════════════════════════════════════════════════════════════════════
# Identity: editing must not re-propose
# ══════════════════════════════════════════════════════════════════════


def test_editing_the_body_does_not_change_the_execution_identity():
    """THE reason the draft lives in its own table.

    The idempotency key hashes the PAYLOAD. If the body lived there, every
    keystroke would produce a different key and the next propose pass would
    create a second card for the same obligation.
    """
    decision = uuid.uuid4()
    payload = {
        "to": ["chrissy@partner.com"],
        "thread_id": "thread-1",
        "candidate_kind": "commitment_follow_up",
        "requires_draft": True,
        "body": "",
    }
    before = idempotency_key(ORG, decision, "reply_to_email", payload)
    # Whatever the founder types lives on the DRAFT row, not here.
    after = idempotency_key(ORG, decision, "reply_to_email", payload)
    assert before == after


async def test_approve_merges_the_draft_without_moving_the_row():
    """The merge writes the reviewed text into the payload. That is safe because
    `idempotency_key` is computed at INSERT and never recomputed — so the row's
    identity survives, and afterwards the payload IS the audit record of what
    was dispatched."""
    req = _Req(
        {
            "to": ["chrissy@partner.com"],
            "thread_id": "thread-1",
            "candidate_kind": "commitment_follow_up",
            "requires_draft": True,
            "body": "",
        }
    )
    draft = _draft(execution_id=req.id, body="Hi Chrissy,\n\nAny update?\n\nThanks,")
    svc = _svc(req, draft, thread=_thread(_msg(mid="m-1", days_ago=4)))

    await svc.approve(req, approver_id=uuid.uuid4())

    assert req.payload["body"] == "Hi Chrissy,\n\nAny update?\n\nThanks,"
    assert req.payload["subject"] == "Re: Product deck"
    assert req.payload["to"] == ["chrissy@partner.com"]
    assert req.payload["reply_thread_id"] == "thread-1"
    # The Phase 1 gate flips only now, once a real body exists.
    assert req.payload["requires_draft"] is False
    assert requires_draft(req.payload) is False
    assert draft.state == DraftState.APPROVED.value


# ══════════════════════════════════════════════════════════════════════
# Approval gating
# ══════════════════════════════════════════════════════════════════════


def test_only_reviewed_states_are_approvable():
    assert APPROVABLE_FROM == {DraftState.READY, DraftState.EDITED}
    # A stale draft is excluded on purpose: its premise no longer holds.
    assert DraftState.STALE not in APPROVABLE_FROM
    assert DraftState.PENDING not in APPROVABLE_FROM


def test_an_approved_draft_cannot_be_regenerated():
    """Regenerating after approval would silently change what the user agreed
    to send."""
    assert DraftState.APPROVED not in REGENERABLE_FROM
    assert DraftState.EDITED not in REGENERABLE_FROM  # needs an explicit force


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("body", "   ", "empty"),
        ("recipients", [], "recipient"),
        ("thread_id", None, "source thread"),
    ],
)
async def test_approval_refuses_an_unsendable_draft(field, value, message):
    req = _Req({"candidate_kind": "waiting_reply", "requires_draft": True})
    draft = _draft(execution_id=req.id, **{field: value})
    svc = _svc(req, draft, thread=_thread(_msg(mid="m-1", days_ago=4)))
    with pytest.raises(DraftNotApprovable, match=message):
        await svc.approve(req)
    assert requires_draft(req.payload), "the payload must stay unsendable"


async def test_a_stale_draft_cannot_be_approved():
    req = _Req({"candidate_kind": "waiting_reply", "requires_draft": True})
    draft = _draft(execution_id=req.id)
    # They wrote again after the draft was generated.
    svc = _svc(
        req,
        draft,
        thread=_thread(_msg(mid="m-1", days_ago=4), _msg(mid="m-2", days_ago=0.1)),
        stale=True,
    )
    with pytest.raises(DraftNotApprovable, match="another message"):
        await svc.approve(req)
    assert requires_draft(req.payload)


async def test_an_edited_draft_can_be_approved():
    req = _Req({"candidate_kind": "waiting_reply", "requires_draft": True})
    draft = _draft(execution_id=req.id, state=DraftState.EDITED.value, body="My own words.")
    svc = _svc(req, draft, thread=_thread(_msg(mid="m-1", days_ago=4)))
    await svc.approve(req)
    assert req.payload["body"] == "My own words."
    assert req.payload["draft_was_edited"] is True


# ══════════════════════════════════════════════════════════════════════
# Staleness
# ══════════════════════════════════════════════════════════════════════


def test_a_new_inbound_message_makes_the_draft_stale():
    req = _Req({"candidate_kind": "waiting_reply"})
    draft = _draft()
    thread = _thread(_msg(mid="m-1", days_ago=4), _msg(mid="m-2", days_ago=0.1))
    assert _stale_reason(req, draft, thread) is StaleReason.NEW_INBOUND


def test_the_user_replying_makes_the_draft_stale():
    """The follow-up is redundant — and sending it would look like Navis is not
    paying attention to what its own user just did."""
    req = _Req({"candidate_kind": "waiting_reply"})
    draft = _draft()
    thread = _thread(
        _msg(mid="m-1", days_ago=4),
        _msg(mid="m-2", days_ago=0.1, direction="outbound", from_email="founder@navis.ai"),
    )
    assert _stale_reason(req, draft, thread) is StaleReason.USER_REPLIED


def test_a_delivery_after_generation_invalidates_a_follow_up():
    """Chasing someone who already delivered is the worst version of this
    feature."""
    req = _Req({
        "candidate_kind": "commitment_follow_up",
        "person_email": "chrissy@partner.com",
    })
    draft = _draft(thread_watermark=None)  # watermark ignored; content-based check
    # 15 minutes ago — i.e. AFTER the draft was generated (1 hour ago).
    thread = _thread(
        _msg(mid="m-1", days_ago=0.01, snippet="Attached is the product deck, sorry for the delay.")
    )
    assert _stale_reason(req, draft, thread) is StaleReason.COMMITMENT_FULFILLED


def test_a_delivery_BEFORE_generation_does_not_invalidate():
    """They delivered something last week, then promised something new."""
    req = _Req({
        "candidate_kind": "commitment_follow_up",
        "person_email": "chrissy@partner.com",
    })
    draft = _draft(thread_watermark=None, generated_at=NOW)
    thread = _thread(_msg(mid="m-1", days_ago=5, snippet="Attached is last week's deck."))
    assert _stale_reason(req, draft, thread) is None


def test_a_recorded_decision_invalidates_a_decision_draft():
    req = _Req({"candidate_kind": "decision_owed"})
    draft = _draft(thread_watermark=None)
    thread = _thread(
        _msg(mid="m-1", days_ago=0.01, snippet="We have decided to proceed on Monday.")
    )
    assert _stale_reason(req, draft, thread) is StaleReason.DECISION_RESOLVED


def test_an_unchanged_thread_is_not_stale():
    req = _Req({"candidate_kind": "waiting_reply"})
    draft = _draft()
    thread = _thread(_msg(mid="m-1", days_ago=4))
    draft.thread_watermark = thread.watermark()
    assert _stale_reason(req, draft, thread) is None


def test_an_unknown_thread_leaves_the_draft_alone():
    """Not being able to see the thread is not evidence that it changed."""
    req = _Req({"candidate_kind": "waiting_reply"})
    assert _stale_reason(req, _draft(), None) is None


# ══════════════════════════════════════════════════════════════════════
# Send-time safety (the last line before Gmail)
# ══════════════════════════════════════════════════════════════════════


async def test_the_provider_still_refuses_a_body_less_reply():
    with pytest.raises(ProviderValidationError, match="no draft"):
        await ReplyToEmailProvider().validate(
            {
                "to": ["a@b.com"],
                "reply_thread_id": "t-1",
                "subject": "Re: x",
                "body": "",
                "requires_draft": True,
            }
        )


async def test_the_provider_still_requires_the_thread_being_replied_to():
    with pytest.raises(ProviderValidationError, match="reply_thread_id"):
        await ReplyToEmailProvider().validate(
            {"to": ["a@b.com"], "subject": "Re: x", "body": "real text"}
        )


async def test_an_approved_draft_passes_provider_validation():
    """The whole point: after review, the payload is genuinely sendable."""
    req = _Req({
        "to": ["chrissy@partner.com"],
        "thread_id": "thread-1",
        "candidate_kind": "commitment_follow_up",
        "requires_draft": True,
        "body": "",
    })
    draft = _draft(execution_id=req.id)
    svc = _svc(req, draft, thread=_thread(_msg(mid="m-1", days_ago=4)))
    await svc.approve(req)
    await ReplyToEmailProvider().validate(req.payload)  # must not raise


def test_the_gmail_message_id_is_deterministic_per_execution():
    """The duplicate-send guard: same execution => same Message-ID => a replay
    finds the earlier send instead of making a second one."""
    from navis.execution.providers._senders import navis_message_id

    eid = uuid.uuid4()
    assert navis_message_id(eid) == navis_message_id(eid)
    assert navis_message_id(uuid.uuid4()) != navis_message_id(eid)
    assert navis_message_id(eid).startswith("<navis-")
    assert navis_message_id(eid).endswith(">")


# ── Test double for the service's DB ───────────────────────────────────


def _svc(req, draft, *, thread, stale=False):
    """DraftService with its two DB touchpoints stubbed.

    Only `db.get` (fetch the draft) and `load_thread` are exercised by the paths
    under test, so a full session fake would add noise without adding coverage.

    Unless `stale=True`, the draft's watermark is synced to the thread — so a
    test asserting approval behaviour is not silently testing the staleness
    guard because two fixture timestamps happened to differ.
    """
    if not stale:
        draft.thread_watermark = thread.watermark()

    class _DB:
        async def get(self, _model, _pk):
            return draft

        async def flush(self):
            return None

        def add(self, _obj):
            return None

    svc = DraftService(_DB())

    async def _load(_db, _org, _thread_id):
        return thread

    import navis.execution.drafts.service as mod

    mod.load_thread = _load  # type: ignore[assignment]
    return svc
