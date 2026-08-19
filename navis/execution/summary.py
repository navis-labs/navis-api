"""The public, minimal view of an execution.

Attached to a brief decision so the dashboard can answer "has this already been
sent?" before offering to send it. Previously it could not: `GET /actions`
returns only REQUESTED rows, so a completed reply was invisible to every
frontend surface and the drawer kept offering "Take Action" for work that was
already done.

This is a PROJECTION of the same row `execution/router.py::_serialize` returns —
same table, same `ExecutionStatus` values, same `provider_reference` shape — not
a second status representation. It is deliberately narrower: a decision card
needs to know what happened, not the frozen email body, the reasoning lineage or
the state watermark.

`provider_reference` is passed through verbatim and unjudged. Whether a row
counts as genuinely SENT is decided in one place on the frontend
(lib/executionStatus.ts::describeSend), which requires EXECUTED plus a real
non-dry-run external_id. Pre-deciding it here would create the second source of
truth this module exists to avoid.
"""
from __future__ import annotations

from typing import Any

from navis.execution.actions import awaiting_draft
from navis.execution.models import ExecutionRequest


def execution_summary(r: ExecutionRequest | None) -> dict[str, Any] | None:
    """The decision-facing view of an execution, or None when none exists.

    None is the ordinary case — most decisions are findings to read, and a
    reply obligation has no execution until someone acts on it.
    """
    if r is None:
        return None
    return {
        "execution_id": str(r.id),
        "status": r.status,
        # {} rather than null when the worker has not dispatched: the column is
        # JSONB defaulting to {}, and flattening that to null on the wire would
        # hide the difference between "no receipt" and "receipt with no id".
        "provider_reference": r.provider_reference or {},
        "attempts": r.attempts or 0,
        # Truncated for the same reason as the actions endpoint: this can carry
        # a full provider traceback.
        "last_error": (r.last_error or None) and r.last_error[:500],
        # True while the action is a placeholder Navis proposed but has not
        # written yet — the card shows "needs a draft" rather than Approve.
        #
        # Asks the ActionSpec rather than testing `action_type ==
        # REPLY_TO_EMAIL`. This projection and `router._serialize` are two
        # surfaces reporting the same fact; both now read it from the same
        # authority, so a new draftable action cannot be draftable on one screen
        # and not the other.
        "requires_draft": awaiting_draft(r.action_type, r.payload),
    }
