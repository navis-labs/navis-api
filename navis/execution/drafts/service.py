"""DraftService — generate, edit, stale-check, and freeze a proactive draft.

THE ONE INVARIANT THIS FILE EXISTS TO HOLD:

    No proactive email reaches Gmail without a human having seen the exact text
    that will be sent.

Everything else is mechanism. `approve()` is the only path from a document to an
executable payload, it refuses every draft state except READY and EDITED, and it
re-checks staleness at the moment of approval rather than trusting a flag set
minutes earlier.

IDENTITY. The ExecutionRequest is the identity of the action, and it never
moves. Editing updates a document row; it does not re-propose, does not
recompute the idempotency key, and cannot produce a second card. That is why
the draft lives in its own table — see `drafts/models.py`.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from navis.execution.drafts.contract import (
    APPROVABLE_FROM,
    REGENERABLE_FROM,
    DraftState,
    StaleReason,
)
from navis.execution.drafts.generator import (
    COMMITMENT_FOLLOW_UP,
    DECISION_OWED,
    DRAFTABLE_KINDS,
    InsufficientContext,
    generate,
)
from navis.execution.drafts.models import ActionDraft
from navis.execution.drafts.thread_context import ThreadContext, load_thread
from navis.execution.models import ExecutionRequest
from navis.execution.status import ExecutionStatus
from navis.platform.monitoring import get_logger

logger = get_logger("execution.drafts.service")


class DraftNotFound(Exception):
    pass


class DraftNotApprovable(Exception):
    """The draft cannot be frozen in its current state — stale, or already
    approved. Carries a human-readable reason for the 409."""


class DraftService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    # ── Read ──────────────────────────────────────────────────────────

    async def get(self, execution_id: uuid.UUID) -> ActionDraft | None:
        return await self.db.get(ActionDraft, execution_id)

    async def get_or_generate(
        self, req: ExecutionRequest, *, sender_name: str | None = None
    ) -> ActionDraft:
        """The Action Center's read path: a draft, generating one on first look.

        Generating lazily (rather than at propose time) means a candidate the
        founder never opens costs nothing, and a draft is always written against
        the freshest thread we have.
        """
        draft = await self.get(req.id)
        if draft is None:
            return await self.generate(req, sender_name=sender_name)
        # A draft the user is about to look at must be checked against the world
        # NOW, not against whatever was true when it was written.
        await self.refresh_staleness(req, draft)
        return draft

    # ── Generate ──────────────────────────────────────────────────────

    async def generate(
        self, req: ExecutionRequest, *, sender_name: str | None = None, force: bool = False
    ) -> ActionDraft:
        """Write (or rewrite) the draft for one proactive candidate.

        Raises `InsufficientContext` when there is nothing honest to write —
        the caller surfaces that to the user rather than storing a vague draft.
        """
        payload = req.payload or {}
        kind = str(payload.get("candidate_kind") or "")
        if kind not in DRAFTABLE_KINDS:
            raise InsufficientContext(
                f"this action ({req.action_type}) is not a draftable proactive candidate"
            )

        existing = await self.get(req.id)
        if existing is not None and not force:
            if DraftState(existing.state) not in REGENERABLE_FROM:
                # EDITED is excluded deliberately: silently overwriting the
                # founder's own words with a fresh template would be the single
                # most infuriating thing this feature could do.
                raise DraftNotApprovable(
                    "This draft has your edits. Regenerating would discard them."
                )

        thread_id = str(payload.get("thread_id") or payload.get("reply_thread_id") or "")
        thread = await load_thread(self.db, req.organization_id, thread_id)
        generated = generate(
            candidate_kind=kind, payload=payload, thread=thread, sender_name=sender_name
        )

        now = datetime.now(timezone.utc)
        if existing is None:
            existing = ActionDraft(
                execution_id=req.id, organization_id=req.organization_id
            )
            self.db.add(existing)
        existing.state = DraftState.READY.value
        existing.subject = generated.subject
        existing.body = generated.body
        existing.recipients = list(generated.recipients)
        existing.thread_id = generated.thread_id
        existing.in_reply_to_message_id = generated.in_reply_to_message_id
        existing.reason = generated.reason
        existing.evidence = generated.evidence
        existing.generator = generated.generator
        existing.thread_watermark = thread.watermark() if thread else None
        existing.stale_reason = None
        existing.generated_at = now
        existing.edited_at = None
        existing.edited_by = None
        await self.db.flush()
        logger.info("draft generated execution=%s kind=%s", req.id, kind)
        return existing

    # ── Edit ──────────────────────────────────────────────────────────

    async def edit(
        self,
        req: ExecutionRequest,
        *,
        subject: str | None = None,
        body: str | None = None,
        recipients: list[str] | None = None,
        editor_id: uuid.UUID | None = None,
    ) -> ActionDraft:
        """Save the founder's changes. Updates a document; creates nothing.

        The execution row is untouched, so the action keeps its identity, its
        lineage and its place on the board across any number of edits.
        """
        draft = await self.get(req.id)
        if draft is None:
            raise DraftNotFound(str(req.id))
        if DraftState(draft.state) is DraftState.APPROVED:
            raise DraftNotApprovable(
                "This draft was already approved and can no longer be edited."
            )

        if subject is not None:
            draft.subject = subject.strip()[:500]
        if body is not None:
            draft.body = body
        if recipients is not None:
            draft.recipients = [r.strip().lower() for r in recipients if r and r.strip()]

        draft.state = DraftState.EDITED.value
        draft.edited_at = datetime.now(timezone.utc)
        draft.edited_by = editor_id
        # The founder rewriting the text does NOT un-stale it — the
        # conversation still moved. It is cleared only by regeneration or by an
        # explicit re-review, so an edited-but-stale draft still warns.
        await self.db.flush()
        return draft

    # ── Staleness ─────────────────────────────────────────────────────

    async def refresh_staleness(
        self, req: ExecutionRequest, draft: ActionDraft
    ) -> ActionDraft:
        """Re-check the world and flag the draft if its premise moved.

        Four ways a draft dies:
          * they wrote again        — the draft may not answer the new message;
          * the founder replied     — the follow-up is redundant;
          * they delivered          — chasing them would now be wrong;
          * the decision was made   — the question is closed.
        """
        if DraftState(draft.state) is DraftState.APPROVED:
            return draft  # frozen; execution owns it now

        thread = await load_thread(
            self.db, req.organization_id, draft.thread_id or ""
        )
        reason = _stale_reason(req, draft, thread)
        if reason is None:
            # Recovered — e.g. the draft was regenerated against the new state.
            if draft.stale_reason is not None:
                draft.stale_reason = None
                if DraftState(draft.state) is DraftState.STALE:
                    draft.state = DraftState.READY.value
                await self.db.flush()
            return draft

        if draft.stale_reason != reason.value:
            draft.stale_reason = reason.value
            # An EDITED draft keeps its state so the founder's work is visibly
            # preserved; the stale_reason is what drives the warning.
            if DraftState(draft.state) is DraftState.READY:
                draft.state = DraftState.STALE.value
            await self.db.flush()
            logger.info("draft went stale execution=%s reason=%s", req.id, reason.value)
        return draft

    # ── Approve ───────────────────────────────────────────────────────

    async def approve(
        self, req: ExecutionRequest, *, approver_id: uuid.UUID | None = None
    ) -> ActionDraft:
        """Freeze the document and merge it into the execution payload.

        This is the ONLY way a proactive body becomes executable. It:
          * refuses anything but READY / EDITED;
          * re-checks staleness at this instant, because the check that ran when
            the card was rendered is already history;
          * writes subject/body/recipients into `req.payload`, which is safe
            because `idempotency_key` was computed at INSERT and is never
            recomputed — the row's identity survives the merge, and the payload
            becomes the audit record of exactly what was dispatched.
        """
        draft = await self.get(req.id)
        if draft is None:
            raise DraftNotFound(str(req.id))

        await self.refresh_staleness(req, draft)
        state = DraftState(draft.state)
        if draft.stale_reason:
            from navis.execution.drafts.contract import STALE_REASON_TEXT

            raise DraftNotApprovable(
                STALE_REASON_TEXT.get(
                    StaleReason(draft.stale_reason),
                    "The conversation changed after this draft was written.",
                )
                + " Regenerate it or re-review before sending."
            )
        if state not in APPROVABLE_FROM:
            raise DraftNotApprovable(
                f"A draft in state '{state.value}' cannot be approved."
            )
        if not draft.body.strip():
            raise DraftNotApprovable("The draft body is empty; there is nothing to send.")
        if not draft.recipients:
            raise DraftNotApprovable("The draft has no recipient.")
        if not draft.thread_id:
            raise DraftNotApprovable(
                "This reply has no source thread, so it cannot be sent as a reply."
            )

        payload = dict(req.payload or {})
        payload["to"] = list(draft.recipients)
        payload["subject"] = draft.subject
        payload["body"] = draft.body
        payload["reply_thread_id"] = draft.thread_id
        if draft.in_reply_to_message_id:
            payload["in_reply_to_message_id"] = draft.in_reply_to_message_id
        # The gate Phase 1 installed: with a real body present this flips false,
        # and both the approve endpoint and the provider stop refusing.
        payload["requires_draft"] = False
        payload["draft_generator"] = draft.generator
        payload["draft_was_edited"] = state is DraftState.EDITED
        req.payload = payload

        draft.state = DraftState.APPROVED.value
        draft.approved_at = datetime.now(timezone.utc)
        await self.db.flush()
        logger.info(
            "draft approved execution=%s edited=%s", req.id, state is DraftState.EDITED
        )
        return draft


# ── Staleness rules ────────────────────────────────────────────────────


def _stale_reason(
    req: ExecutionRequest, draft: ActionDraft, thread: ThreadContext | None
) -> StaleReason | None:
    """Why this draft is out of date, or None if it still holds."""
    if thread is None:
        return None  # nothing new can be known; leave the draft alone

    current = thread.watermark()
    if current and draft.thread_watermark and current != draft.thread_watermark:
        # The conversation moved. WHICH way decides the message the user sees.
        latest = thread.latest
        if latest is not None and not latest.is_inbound:
            return StaleReason.USER_REPLIED
        return StaleReason.NEW_INBOUND

    # Obligation-specific invalidation, independent of the watermark: the
    # commitment may have been delivered or the decision made in a message we
    # already had when the draft was written.
    payload = req.payload or {}
    kind = str(payload.get("candidate_kind") or "")
    generated_at = draft.generated_at
    if kind == COMMITMENT_FOLLOW_UP:
        speaker = str(payload.get("person_email") or "").lower()
        for m in thread.messages:
            if (
                m.is_inbound
                and (m.from_email or "").lower() == speaker
                and _mentions_delivery(m.snippet)
                and (m.sent_at is None or generated_at is None or m.sent_at > generated_at)
            ):
                return StaleReason.COMMITMENT_FULFILLED
    elif kind == DECISION_OWED:
        for m in thread.messages:
            if _mentions_decision(m.snippet) and (
                m.sent_at is None or generated_at is None or m.sent_at > generated_at
            ):
                return StaleReason.DECISION_RESOLVED
    return None


def _mentions_delivery(text: str | None) -> bool:
    from navis.signals.text_rules import body_shows_fulfilment

    return bool(text) and body_shows_fulfilment(text or "")


def _mentions_decision(text: str | None) -> bool:
    from navis.signals.text_rules import body_records_decision

    return bool(text) and body_records_decision(text or "")


async def pending_draft_states(
    db: AsyncSession, org_id: uuid.UUID, execution_ids: list[uuid.UUID]
) -> dict[uuid.UUID, ActionDraft]:
    """Drafts for a batch of executions — one query for the whole board."""
    if not execution_ids:
        return {}
    rows = (
        await db.scalars(
            select(ActionDraft).where(
                ActionDraft.organization_id == org_id,
                ActionDraft.execution_id.in_(execution_ids),
            )
        )
    ).all()
    return {r.execution_id: r for r in rows}


def is_awaiting_draft(req: ExecutionRequest) -> bool:
    """True when the execution still refuses to run for want of a body."""
    from navis.execution.providers.reply import requires_draft

    return (
        req.status == ExecutionStatus.REQUESTED.value
        and requires_draft(req.payload)
    )
