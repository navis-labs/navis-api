"""Scheduling wire types — pydantic-validated at every boundary.

Split into three layers:

  * INPUT    — `SchedulingContext`: what the chat endpoint hands to the
               agent (utterance + optional signal/thread refs + timezone +
               `pending` state carried over from a prior clarification).
  * INTERNAL — `MeetingConstraints`, `ResolvedAttendee`, `MeetingSlot`,
               `PendingScheduling`: the stages of resolution the agent
               walks through. Every field is optional so a partial parse
               can tell the router WHICH slot is missing (which drives
               the clarification question).
  * OUTPUT   — `SchedulingResult` = `Clarification` | `ScheduledProposal`.
               The agent returns exactly one — never both, never a side
               effect. `ScheduledProposal` is the shape the chat router
               feeds into `ExecutionService.request(...)`.

Every `Clarification` carries a `PendingScheduling` in `.resolved`
representing everything the agent has resolved SO FAR. The frontend
echoes it back on the next chat turn so bare answers ("tomorrow at
3 PM") merge with the pending attendee/date/time/duration instead of
requiring the user to repeat the whole request.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatTurn(BaseModel):
    """One prior chat turn from the client — used to resolve pronouns.

    Deliberately narrow: role + text + optional signal reference. Rich
    per-turn metadata (citations, action proposals) belongs on the
    NavisChatMessage envelope; the scheduling agent doesn't need it.
    """

    model_config = ConfigDict(extra="ignore")

    role: Literal["user", "assistant"]
    content: str = Field("", max_length=4000)
    active_signal_id: uuid.UUID | None = None


class SchedulingContext(BaseModel):
    """Input to `SchedulingAgent.propose`.

    The chat router assembles this from the ChatRequest — the agent
    itself never reads directly from the HTTP layer, which keeps it
    trivially unit-testable.
    """

    model_config = ConfigDict(extra="ignore")

    org_id: uuid.UUID
    user_id: uuid.UUID
    # The user's natural-language request. Kept small; utterances >2000
    # chars are almost always a paste error, not a scheduling ask.
    utterance: str = Field(..., min_length=1, max_length=2000)
    # Optional pointer to the signal the user is looking at (an email
    # they just opened, a meeting they clicked). When present, pronoun
    # resolution ("him", "her") uses the signal's sender.
    active_signal_id: uuid.UUID | None = None
    # Optional thread grouping (Gmail threadId). Used as a fallback when
    # active_signal_id isn't set — the most-recent signal in the thread
    # supplies the pronoun antecedent.
    thread_id: str | None = Field(None, max_length=256)
    # Prior chat turns, oldest → newest. Only used for lightweight
    # context (last email discussed, prior person named). Capped at 6
    # to keep prompts and DB lookups bounded.
    recent_turns: list[ChatTurn] = Field(default_factory=list, max_length=6)
    # Founder's IANA timezone; needed to convert "Tuesday afternoon"
    # into a concrete UTC window for freebusy. Defaults to UTC when the
    # frontend hasn't supplied one — the resulting proposal makes the
    # assumption explicit in `assumptions[]`.
    time_zone: str = Field("UTC", max_length=64)
    # Anchor "now" so tests can inject a deterministic clock. Router
    # sets this to datetime.now(tz=UTC) in production.
    now: datetime
    # Everything the agent resolved during the LAST turn, echoed back by
    # the frontend on this turn's request. Used to merge follow-up
    # answers ("tomorrow at 3 PM") into the pending scheduling flow so
    # the user doesn't have to repeat the attendee. None on a fresh
    # scheduling request.
    pending: "PendingScheduling | None" = None


class PendingScheduling(BaseModel):
    """Partial scheduling state carried across chat turns.

    Every field is optional. The frontend stores whatever the last
    `Clarification.resolved` carried and echoes it back on the next
    chat call; the agent merges new extractions from the current
    utterance ON TOP (new value wins; null keeps pending).

    Attendee is broken into (email, name, source) rather than a nested
    `ResolvedAttendee` so partial resolutions (name known but email
    not yet) can be carried. Only when BOTH email and source are
    present is the attendee considered "resolved" for merge purposes.
    """

    model_config = ConfigDict(extra="ignore")

    # Attendee already resolved on a prior turn.
    attendee_email: str | None = Field(None, max_length=320)
    attendee_name: str | None = Field(None, max_length=320)
    attendee_source: Literal[
        "active_signal_sender",
        "recent_signal_sender",
        "person_search_unique",
        "direct_email",
    ] | None = None
    attendee_canonical_id: str | None = None

    # Signal / thread lineage — carried so the created event still
    # links back to the email that sparked the scheduling request even
    # after a multi-turn clarification.
    source_signal_event_id: str | None = None
    thread_id: str | None = Field(None, max_length=256)

    # The constraint slots.
    date_constraint: str | None = Field(None, max_length=40)
    time_window: str | None = Field(None, max_length=40)
    duration_minutes: int | None = Field(None, ge=15, le=240)
    title: str | None = Field(None, max_length=200)


class ResolvedAttendee(BaseModel):
    """The person we identified as the meeting counterpart.

    `email` is the load-bearing field — every downstream stage requires
    it. `name` is best-effort (for the proposal card) and `source`
    documents HOW we resolved so the audit trail explains any surprise.
    """

    model_config = ConfigDict(extra="ignore")

    email: str
    name: str | None = None
    canonical_id: str | None = None
    source: Literal[
        "active_signal_sender",
        "recent_signal_sender",
        "person_search_unique",
        "direct_email",
    ]


class MeetingConstraints(BaseModel):
    """LLM-extracted meeting fields.

    Optional-first: the agent uses "missing required field" as the
    trigger for a clarification. `duration_minutes` has a validated
    default because the product allows a reasonable default (30) — the
    assumption is echoed back in the proposal so the founder can see it.
    """

    model_config = ConfigDict(extra="ignore")

    # A short day-scope hint the LLM extracts from the utterance:
    #   "today", "tomorrow", "monday" .. "sunday", "next_week",
    #   or an ISO calendar date "YYYY-MM-DD".
    # Left as a string (not an enum) so the LLM output stays flexible;
    # the availability stage validates it strictly.
    date_constraint: str | None = Field(None, max_length=40)
    # "morning" | "afternoon" | "evening" | "any" | "HH:MM-HH:MM"
    # Same rationale: string sentinel, strictly validated downstream.
    time_window: str | None = Field(None, max_length=40)
    duration_minutes: int | None = Field(None, ge=15, le=240)
    title: str | None = Field(None, max_length=200)
    # LLM-inferred candidate name/handle for the attendee ("Rahul",
    # "Priya S."). NOT trusted for the actual scheduling — used only to
    # drive the person-search step. Empty when the utterance uses a
    # pronoun with no name.
    attendee_hint: str | None = Field(None, max_length=120)
    pronoun: Literal["him", "her", "them", "that_person"] | None = None


class MeetingSlot(BaseModel):
    """One concrete non-conflicting slot on the founder's calendar."""

    model_config = ConfigDict(extra="ignore")

    start: datetime
    end: datetime
    time_zone: str
    duration_minutes: int


class Clarification(BaseModel):
    """The agent needs more information from the user before proposing.

    `field` tells the frontend WHICH slot to focus on (used by the
    ScheduleMeetingProposalCard clarification renderer). `options` is
    populated only for finite pick-lists (multiple person matches, or
    alternative slots when the requested exact time was busy).

    `resolved` carries the FULL pending scheduling state the agent has
    built up across the conversation so the frontend can echo it back
    on the next turn. The next turn's utterance is merged INTO this
    pending state; missing slots that this turn provided fill in, and
    already-known slots (typically the attendee) do NOT have to be
    respoken.
    """

    model_config = ConfigDict(extra="ignore")

    kind: Literal["clarification"] = "clarification"
    # Which slot the user is being asked about. `datetime` covers the
    # "both date AND time missing" case (a single follow-up like
    # "tomorrow at 3 PM" resolves both). `scope` is reserved for
    # calendar-connection / reconnect / Google-error situations — the
    # frontend shows a "Reconnect Google Calendar" affordance ONLY on
    # this field, not on generic date/time asks.
    field: Literal[
        "attendee",
        "date",
        "datetime",
        "time",
        "duration",
        "person_disambiguation",
        "scope",
    ]
    question: str = Field(..., max_length=500)
    options: list[dict] = Field(default_factory=list)
    # Structured pending state — see PendingScheduling. Serializes as a
    # plain dict on the wire; the frontend echoes it back verbatim.
    resolved: "PendingScheduling" = Field(default_factory=lambda: PendingScheduling())


class ScheduledProposal(BaseModel):
    """A ready-to-approve meeting.

    This is the payload the chat router feeds into
    `ExecutionService.request(action_type='schedule_meeting', payload=…)`.
    Every field required by `_senders.calendar_sender` is populated.
    """

    model_config = ConfigDict(extra="ignore")

    kind: Literal["proposal"] = "proposal"
    attendee: ResolvedAttendee
    slot: MeetingSlot
    title: str = Field(..., max_length=200)
    description: str | None = Field(None, max_length=2000)
    source_signal_event_id: str | None = None
    thread_id: str | None = None
    # Human-readable notes explaining defaults the agent applied
    # (e.g. "Assumed 30 min duration.") — surfaced on the frontend
    # proposal card so the founder can spot bad assumptions before
    # approving.
    assumptions: list[str] = Field(default_factory=list)

    def to_execution_payload(self) -> dict:
        """The dict that becomes `ExecutionRequest.payload`.

        Contract MUST match `providers/_senders.calendar_sender`'s
        expected keys: title, start, end, attendees, time_zone,
        description, send_google_invite. Extra keys are fine — the
        sender ignores what it doesn't consume; keeping them here
        makes the audit row self-describing.
        """
        return {
            "title": self.title,
            "start": self.slot.start.isoformat(),
            "end": self.slot.end.isoformat(),
            "time_zone": self.slot.time_zone,
            "duration_minutes": self.slot.duration_minutes,
            "attendees": [self.attendee.email],
            "attendee_name": self.attendee.name,
            "description": self.description,
            "send_google_invite": True,
            # Lineage — carried through so the executed event links back
            # to the signal it was born from.
            "source_signal_event_id": self.source_signal_event_id,
            "thread_id": self.thread_id,
            "assumptions": self.assumptions,
        }


# The agent's only return type — a discriminated union.
SchedulingResult = Clarification | ScheduledProposal


# Resolve the string forward references above (`SchedulingContext.pending`
# points at `PendingScheduling`, `Clarification.resolved` too).
SchedulingContext.model_rebuild()
Clarification.model_rebuild()
