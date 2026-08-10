"""Reviewable email drafts for proactive action candidates.

The layer between "Navis noticed something" (Phase 1/2A) and "Navis sent an
email" (execution). It exists so a human always sees the exact text before it
leaves, and so editing that text never changes the identity of the action.

    ExecutionRequest   the ACTION — identity, lineage, approval, side effect
    ActionDraft        the DOCUMENT — subject, body, recipients, review state

One row each, joined on execution_id, with separate lifecycles on purpose (see
`contract.DraftState`).
"""
from navis.execution.drafts.contract import (
    APPROVABLE_FROM,
    REGENERABLE_FROM,
    STALE_REASON_TEXT,
    DraftState,
    StaleReason,
)
from navis.execution.drafts.generator import (
    DRAFTABLE_KINDS,
    GENERATOR_VERSION,
    GeneratedDraft,
    InsufficientContext,
    generate,
)
from navis.execution.drafts.models import ActionDraft
from navis.execution.drafts.service import (
    DraftNotApprovable,
    DraftNotFound,
    DraftService,
    is_awaiting_draft,
    pending_draft_states,
)
from navis.execution.drafts.thread_context import (
    ThreadContext,
    ThreadMessage,
    load_thread,
)

__all__ = [
    "ActionDraft",
    "APPROVABLE_FROM",
    "DRAFTABLE_KINDS",
    "DraftNotApprovable",
    "DraftNotFound",
    "DraftService",
    "DraftState",
    "GENERATOR_VERSION",
    "GeneratedDraft",
    "InsufficientContext",
    "REGENERABLE_FROM",
    "STALE_REASON_TEXT",
    "StaleReason",
    "ThreadContext",
    "ThreadMessage",
    "generate",
    "is_awaiting_draft",
    "load_thread",
    "pending_draft_states",
]
