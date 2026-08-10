"""Draft lifecycle — deliberately SEPARATE from ExecutionStatus.

An ExecutionRequest's status machine describes a side effect: requested,
approved, executing, executed. A draft's states describe a document a human is
still working on. Folding "the user is editing the body" into ExecutionStatus
would mean inventing transitions on a machine whose whole job is to make double
side effects impossible.

So the two run in parallel over one identity:

    ExecutionRequest.status   REQUESTED ─────────────────────▶ APPROVED ─▶ EXECUTED
    ActionDraft.state         PENDING ─▶ READY ⇄ EDITED ─────▶ APPROVED
                                          └─▶ STALE ─▶ (regenerate) ─▶ READY

The execution row cannot leave REQUESTED until the draft reaches APPROVED —
enforced in the approve endpoint and, as a backstop, by the provider refusing a
body-less payload.
"""
from __future__ import annotations

import enum


class DraftState(str, enum.Enum):
    """Where the DOCUMENT is, not where the side effect is."""

    # A candidate exists but nothing has been written yet. This is the state
    # Phase 2A shipped: `requires_draft = True` and no draft row at all.
    PENDING = "pending"
    # Navis generated a draft. Awaiting human review.
    READY = "ready"
    # The human changed it. Distinct from READY so the UI can say "your edits"
    # and so regeneration knows not to silently overwrite them.
    EDITED = "edited"
    # The world moved: a new message arrived, the commitment was delivered, the
    # decision was made. Sending now would be replying to a conversation that
    # no longer exists in that form.
    STALE = "stale"
    # Merged into the ExecutionRequest payload. Terminal — the document is
    # frozen and the execution machine owns what happens next.
    APPROVED = "approved"


# States a draft may be approved FROM. STALE is excluded on purpose: a stale
# draft must be regenerated or explicitly re-reviewed first, because the whole
# point of the stale flag is that its premise no longer holds.
APPROVABLE_FROM = frozenset({DraftState.READY, DraftState.EDITED})

# States that can be regenerated in place. APPROVED is terminal — regenerating
# after approval would silently change what the user agreed to send.
REGENERABLE_FROM = frozenset({DraftState.PENDING, DraftState.READY, DraftState.STALE})


class StaleReason(str, enum.Enum):
    """WHY a draft went stale. Surfaced verbatim so the user understands what
    changed rather than being told to re-review for no stated reason."""

    NEW_INBOUND = "new_inbound"          # they wrote again
    USER_REPLIED = "user_replied"        # the founder already answered
    COMMITMENT_FULFILLED = "commitment_fulfilled"
    DECISION_RESOLVED = "decision_resolved"


STALE_REASON_TEXT: dict[StaleReason, str] = {
    StaleReason.NEW_INBOUND: (
        "They sent another message after this draft was written, so it may no "
        "longer answer what they asked."
    ),
    StaleReason.USER_REPLIED: (
        "You already replied in this thread, so this follow-up is probably no "
        "longer needed."
    ),
    StaleReason.COMMITMENT_FULFILLED: (
        "They delivered what they promised after this draft was written — "
        "chasing them now would be wrong."
    ),
    StaleReason.DECISION_RESOLVED: (
        "The decision was recorded in this thread after the draft was written."
    ),
}
