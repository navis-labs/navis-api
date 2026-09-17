"""Wire types for mutating an EXISTING calendar event.

Deliberately mirrors `reasoning.scheduling.contract` — same Clarification shape,
same "agent returns exactly one of clarify | propose, never a side effect" rule —
so the chat router handles all three meeting verbs through one code path and the
frontend renders one card.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CalendarOpsContext(BaseModel):
    """Input to `CalendarOpsAgent.propose`."""

    model_config = ConfigDict(extra="ignore")

    org_id: uuid.UUID
    utterance: str = Field(..., min_length=1, max_length=2000)
    time_zone: str = Field("UTC", max_length=64)
    now: datetime
    # Set by the router from `ChatRoute`. The two verbs share targeting and
    # differ only in what they propose, so one agent handles both.
    operation: Literal["reschedule", "cancel"]
    # An event the user already disambiguated on a prior turn. Echoed back by
    # the frontend exactly like `PendingScheduling`.
    target_event_id: str | None = Field(None, max_length=1024)


class TargetEvent(BaseModel):
    """The existing event a mutation will act on."""

    model_config = ConfigDict(extra="ignore")

    event_id: str
    title: str
    start: datetime
    end: datetime
    time_zone: str = "UTC"
    attendees: list[str] = Field(default_factory=list)


class CalendarOpsClarification(BaseModel):
    """The agent needs more input. Same contract as scheduling's Clarification.

    `field`:
      * `event`  — which meeting? (`options` carries the candidate events)
      * `time`   — reschedule to WHEN?
      * `scope`  — calendar connection / permission problem
      * `none`   — nothing to act on (no matching meeting). Informational; the
                   frontend renders the question as plain prose with no prompt.
    """

    model_config = ConfigDict(extra="ignore")

    kind: Literal["clarification"] = "clarification"
    field: Literal["event", "time", "scope", "none"]
    question: str = Field(..., max_length=500)
    options: list[dict] = Field(default_factory=list)
    target_event_id: str | None = None


class MeetingMutationProposal(BaseModel):
    """A ready-to-approve change to an existing event."""

    model_config = ConfigDict(extra="ignore")

    kind: Literal["proposal"] = "proposal"
    operation: Literal["reschedule", "cancel"]
    target: TargetEvent
    # Populated for `reschedule` only — the slot the event moves TO.
    new_start: datetime | None = None
    new_end: datetime | None = None
    assumptions: list[str] = Field(default_factory=list)

    def to_execution_payload(self) -> dict:
        """The dict that becomes `ExecutionRequest.payload`.

        Must match the sender contracts in
        `execution.providers._senders` (`calendar_reschedule_sender` /
        `calendar_cancel_sender`): both require `event_id`, reschedule also
        requires `start`.
        """
        payload: dict = {
            "event_id": self.target.event_id,
            "title": self.target.title,
            "time_zone": self.target.time_zone,
            "attendees": list(self.target.attendees),
            "send_google_invite": True,
            # Kept so the audit row explains WHAT moved, not just that
            # something did.
            "previous_start": self.target.start.isoformat(),
            "previous_end": self.target.end.isoformat(),
        }
        if self.operation == "reschedule":
            if self.new_start is None or self.new_end is None:  # pragma: no cover
                raise ValueError("reschedule proposal without a new slot")
            payload["start"] = self.new_start.isoformat()
            payload["end"] = self.new_end.isoformat()
            payload["duration_minutes"] = int(
                (self.new_end - self.new_start).total_seconds() // 60
            )
        return payload


CalendarOpsResult = CalendarOpsClarification | MeetingMutationProposal
