"""Execution value objects — provider results + the hooks Observation will consume.

Observation V2 will subscribe to ACTION_EXECUTED and read `OutcomeReference` (carried
in the event payload) to track "did the email get a reply / meeting get booked?".
Execution exposes these so Observation can be built later WITHOUT changing Execution.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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
    reasoning_version: int | None  # mirrors ExecutionRequest.reasoning_version (nullable)
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
