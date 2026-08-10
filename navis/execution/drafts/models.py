"""ActionDraft — the reviewable document behind a proactive ExecutionRequest.

WHY A SEPARATE TABLE, and not `ExecutionRequest.payload`
--------------------------------------------------------
The execution idempotency key is a hash of (org, decision, action_type,
PAYLOAD). If the draft body lived in the payload, every keystroke of the user's
edit would produce a different key — and therefore, on the next propose pass, a
DIFFERENT ExecutionRequest. The Action Center would grow a new card per edit and
the "one obligation, one action" identity would be gone.

Keeping the document in its own row, keyed BY execution_id, means:

  * the ExecutionRequest identity is fixed at proposal time and never moves;
  * editing is a plain UPDATE of a document, not a re-proposal;
  * the draft can carry review state the execution machine has no business
    knowing about (stale, edited, who edited).

At approval the draft is merged INTO the payload — which is safe precisely
because `idempotency_key` is computed once at insert and never recomputed, so
the row's identity survives the merge. After that the payload is the audit
record of exactly what was dispatched.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from navis.execution.drafts.contract import DraftState
from navis.platform.database.base import Base, Timestamped


class ActionDraft(Timestamped, Base):
    """One reviewable email draft per proactive ExecutionRequest."""

    __tablename__ = "action_drafts"

    # The execution row IS the identity of the action, so it is also the PK of
    # its draft. One action, at most one draft, no join key to keep in sync.
    execution_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("execution_requests.id", ondelete="CASCADE"),
        primary_key=True,
    )
    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    state: Mapped[str] = mapped_column(
        String(16), nullable=False, default=DraftState.PENDING.value, index=True
    )

    # --- The document ---
    subject: Mapped[str] = mapped_column(String(500), nullable=False, default="")
    body: Mapped[str] = mapped_column(Text, nullable=False, default="")
    # list[str]; the provider normalises, but a list is the honest shape.
    recipients: Mapped[list] = mapped_column(JSONB, default=list)

    # --- Reply threading (see providers/_senders.gmail_sender) ---
    thread_id: Mapped[str | None] = mapped_column(String(256), nullable=True)
    in_reply_to_message_id: Mapped[str | None] = mapped_column(String(512), nullable=True)

    # --- Why this exists (shown to the user, never internal reasoning) ---
    reason: Mapped[str] = mapped_column(Text, nullable=False, default="")
    evidence: Mapped[dict] = mapped_column(JSONB, default=dict)

    # --- Staleness ---
    # Snapshot of the thread at generation time. A newer message id or a newer
    # timestamp in the same thread means the conversation moved on.
    thread_watermark: Mapped[str | None] = mapped_column(String(512), nullable=True)
    stale_reason: Mapped[str | None] = mapped_column(String(40), nullable=True)

    generated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    edited_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    edited_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    approved_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Which template produced the body. Lets a future generator version
    # regenerate old drafts, and tells the audit trail that a HUMAN wrote the
    # final text when the state is EDITED.
    generator: Mapped[str] = mapped_column(String(40), nullable=False, default="")

    def to_dict(self) -> dict:
        """Wire shape. `requires_review` is always True — see the module docs in
        `drafts/service.py`: nothing proactive may reach Gmail unreviewed."""
        return {
            "execution_id": str(self.execution_id),
            "state": self.state,
            "subject": self.subject,
            "body": self.body,
            "recipients": list(self.recipients or []),
            "thread_id": self.thread_id,
            "in_reply_to_message_id": self.in_reply_to_message_id,
            "reason": self.reason,
            "evidence": self.evidence or {},
            "stale_reason": self.stale_reason,
            "generated_at": self.generated_at.isoformat() if self.generated_at else None,
            "edited_at": self.edited_at.isoformat() if self.edited_at else None,
            "approved_at": self.approved_at.isoformat() if self.approved_at else None,
            "generator": self.generator,
            "requires_review": True,
        }
