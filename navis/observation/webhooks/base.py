"""Webhook framework — normalize a provider notification into one shape.

A parser turns a raw provider body into an ObservationWebhookEvent: which provider,
which external object (the provider_reference external_id stored at execution time),
what outcome it represents, when it happened, and a stable source_event_id for
idempotent capture. The service then looks up the original action's outcome by
(provider, external_id) and appends the new outcome with inherited lineage.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from navis.observation.models import OutcomeType


@dataclass(frozen=True, slots=True)
class ObservationWebhookEvent:
    provider: str               # "gmail" | "calendar"
    external_id: str            # matches provider_reference.external_id from Execution
    outcome_type: OutcomeType
    occurred_at: datetime
    source_event_id: str        # stable id for idempotent capture (dedupe)
    raw: dict[str, Any] = field(default_factory=dict)
