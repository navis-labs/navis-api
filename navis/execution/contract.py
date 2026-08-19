"""Execution value objects — provider results + the hooks Observation will consume.

Observation V2 will subscribe to ACTION_EXECUTED and read `OutcomeReference` (carried
in the event payload) to track "did the email get a reply / meeting get booked?".
Execution exposes these so Observation can be built later WITHOUT changing Execution.
"""
from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass, field
from typing import Any


class FailureKind(str, enum.Enum):
    """WHY an execution failed, as a stable value rather than a prose string.

    THE PROBLEM THIS SOLVES. Every failure reached Observation as one
    undifferentiated `ACTION_FAILED` whose only distinguishing feature was a
    free-text `error`. "The organization is no longer entitled to act",
    "nobody has connected Google", "the payload was malformed" and "Gmail was
    down for five attempts" are four completely different facts about the
    organization — three of them need a specific human action and one needs
    nobody — and anything reading outcomes as evidence had to sniff substrings
    out of an error message to tell them apart.

    The worker ALREADY computes this distinction to decide permanent vs
    transient; it simply was not carried forward. Naming it here, beside
    `OutcomeReference`, keeps the dependency pointing the right way: Execution
    knows the cause and publishes it, Observation stores it verbatim and never
    interprets it, and Learning may later map it to meaning — which is Learning's
    job, not this layer's.

    These describe the CAUSE, not the remedy.
    """

    # The organization was not permitted to act at execution time — capability
    # revoked, org suspended, kill switch thrown. Deliberately distinct from a
    # credential problem: possession of a token is not permission to use it.
    AUTHORIZATION_DENIED = "authorization_denied"
    # No usable credential for this connector: not connected, revoked, or
    # connected without the scope the action needs. Needs a human to reconnect.
    CREDENTIAL_UNAVAILABLE = "credential_unavailable"
    # The payload could not be executed as written. Never retryable.
    VALIDATION_FAILED = "validation_failed"
    # The connector declared a terminal business failure — the channel does not
    # exist, the event is gone, the recipient was refused.
    PROVIDER_REJECTED = "provider_rejected"
    # No provider is wired for this action type in this deployment.
    UNROUTABLE = "unroutable"
    # Transient failures that used up the retry budget.
    RETRY_EXHAUSTED = "retry_exhausted"
    # A transient failure with attempts still remaining.
    TRANSIENT = "transient"


@dataclass(frozen=True, slots=True)
class ProviderReference:
    """The external id(s) a provider returns (gmail_message_id, calendar_event_id...)."""

    provider: str
    external_id: str | None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"provider": self.provider, "external_id": self.external_id, **self.extra}


@dataclass(frozen=True, slots=True)
class ProviderResult:
    ok: bool
    reference: ProviderReference | None = None
    error: str | None = None
    retryable: bool = False


@dataclass(frozen=True, slots=True)
class OutcomeReference:
    """Everything Observation needs to watch for an outcome and attribute it back."""

    execution_id: str
    org_id: str
    decision_id: str
    action_type: str
    reasoning_signal_ids: list[str]
    reasoning_version: int
    state_watermark: str | None
    provider_reference: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "org_id": self.org_id,
            "decision_id": self.decision_id,
            "action_type": self.action_type,
            "reasoning_signal_ids": self.reasoning_signal_ids,
            "reasoning_version": self.reasoning_version,
            "state_watermark": self.state_watermark,
            "provider_reference": self.provider_reference,
        }
