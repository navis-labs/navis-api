"""Webhook framework — normalize a provider notification into one shape.

A parser turns a raw provider body into an ObservationWebhookEvent: which provider,
WHICH CONNECTED ACCOUNT the notification is about, which external object (the
provider_reference external_id stored at execution time), what outcome it
represents, when it happened, and a stable source_event_id for idempotent capture.
The edge resolves the account to its owning organization, and the service then
looks up the original action's outcome WITHIN that organization and appends the
new outcome with inherited lineage.

`external_account` is what makes that chain tenant-safe, and it is why it is
required rather than optional-in-practice. Without it the only thing selecting a
tenant was `(provider, external_id)` scanned across every organization's outcomes,
newest wins — so an external id that two tenants share, or that a caller can
guess, decided whose learning loop received the fact. Naming the account instead
means the organization comes from a persisted Integration row, exactly as it does
on the Gmail push and Slack event paths.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from navis.observation.models import OutcomeType


def parse_when(ts: Any) -> datetime:
    """Webhook timestamp -> AWARE datetime, fault-tolerant. ONE home for every parser.

    A malformed timestamp must DEGRADE to ingest time, never raise: an exception here
    500s the webhook and the provider redelivers forever (poison retry loop). A naive
    timestamp is coerced to UTC — `occurred_at` is a timestamptz column and is compared
    against aware datetimes downstream (a naive value would fail the write or the
    comparison, re-entering the same retry loop)."""
    if isinstance(ts, str) and ts:
        try:
            parsed = datetime.fromisoformat(ts)
        except ValueError:
            return datetime.now(timezone.utc)
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc)


@dataclass(frozen=True, slots=True)
class ObservationWebhookEvent:
    provider: str               # "gmail" | "calendar"
    external_id: str            # matches provider_reference.external_id from Execution
    outcome_type: OutcomeType
    occurred_at: datetime
    source_event_id: str        # stable id for idempotent capture (dedupe)
    raw: dict[str, Any] = field(default_factory=dict)
    # The connected account the notification concerns (the mailbox / calendar
    # address). Resolved against `integrations.external_account` to obtain the
    # owning organization. A parser that cannot determine it leaves it None, and
    # the edge refuses the delivery rather than guessing a tenant.
    external_account: str | None = None
