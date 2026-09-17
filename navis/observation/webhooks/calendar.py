"""Calendar outcome parser — meeting accepted / completed.

Keys on the calendar event_id (== provider_reference.external_id for a meeting we
scheduled). Status maps to the outcome type.
"""
from __future__ import annotations

from typing import Any

from navis.observation.models import OutcomeType
from navis.observation.webhooks.base import ObservationWebhookEvent, parse_when

_STATUS_OUTCOME = {
    "accepted": OutcomeType.MEETING_ACCEPTED,
    "completed": OutcomeType.MEETING_COMPLETED,
}


def parse_calendar(body: dict[str, Any]) -> ObservationWebhookEvent | None:
    external_id = body.get("event_id")
    status = (body.get("status") or "").lower()
    outcome_type = _STATUS_OUTCOME.get(status)
    if not external_id or outcome_type is None:
        return None
    occurred = parse_when(body.get("occurred_at"))
    return ObservationWebhookEvent(
        provider="calendar",
        external_id=str(external_id),
        outcome_type=outcome_type,
        occurred_at=occurred,
        source_event_id=str(body.get("event_id_token") or f"calendar:{external_id}:{status}"),
        raw={"status": status, "attendee": body.get("attendee")},
        # WHICH CALENDAR the event lives on — the connected Google account, which
        # is what `integrations.external_account` holds for the Google family.
        # NOT `attendee`: an attendee is a counterparty on someone else's meeting
        # and naming one would let an outside address select a tenant.
        external_account=(body.get("calendar_id") or body.get("emailAddress") or None),
    )
