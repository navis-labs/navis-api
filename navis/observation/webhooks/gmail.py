"""Gmail outcome parser — a reply to a message we sent.

Maps to the sent message's external_id (the gmail_message_id stored in
provider_reference). A real Gmail push carries the thread/message ids; we key on
the referenced message id so the reply attaches to the original action's outcome.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from navis.observation.models import OutcomeType
from navis.observation.webhooks.base import ObservationWebhookEvent


def parse_gmail(body: dict[str, Any]) -> ObservationWebhookEvent | None:
    # `in_reply_to` = the message id we originally sent (== provider_reference.external_id).
    external_id = body.get("in_reply_to") or body.get("message_id")
    if not external_id:
        return None
    ts = body.get("received_at")
    occurred = (
        datetime.fromisoformat(ts) if ts else datetime.now(timezone.utc)
    )
    return ObservationWebhookEvent(
        provider="gmail",
        external_id=str(external_id),
        outcome_type=OutcomeType.REPLY_RECEIVED,
        occurred_at=occurred,
        source_event_id=str(body.get("event_id") or f"gmail:{external_id}:{body.get('reply_id', '')}"),
        raw={"from": body.get("from"), "snippet": body.get("snippet")},
        # WHICH MAILBOX received the reply. `emailAddress` is the key Google's own
        # Gmail push payload uses (see signals/router.py::google_pubsub_webhook),
        # so a real notification already carries it; `mailbox` is accepted as the
        # generic alias. Absent -> the edge refuses rather than scanning tenants.
        external_account=(body.get("emailAddress") or body.get("mailbox") or None),
    )
