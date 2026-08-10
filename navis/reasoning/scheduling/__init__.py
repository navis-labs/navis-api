"""Scheduling reasoning — Ask Navis → ExecutionRequest(schedule_meeting).

The chat endpoint delegates a scheduling utterance ("schedule with him
Tuesday afternoon") to `SchedulingAgent.propose`. The agent:

  1. Resolves the intended attendee (pronoun → active-signal sender;
     name → identity graph search; email → direct).
  2. Extracts a validated `MeetingConstraints` from natural language
     (LLM slot-fill with strict pydantic validation — the LLM never
     executes anything, only proposes fields we then check).
  3. Consults the founder's real Google Calendar via the existing
     `CalendarClient.freebusy` and picks the first non-conflicting slot
     in the constraint window.
  4. Returns either a `Clarification` (missing / ambiguous input) or a
     `ScheduledProposal` (fully resolved meeting ready for approval).

The agent NEVER creates the Google event. The chat router hands the
proposal to the canonical `ExecutionService.request` pipeline, which
holds it in REQUESTED until the user approves; the Celery worker + the
existing `CalendarProvider` + `_senders.calendar_sender` do the real
Google side effect on approval.

Kept as a package (not a single file) so each stage is unit-testable in
isolation without importing the LLM stack.
"""
from navis.reasoning.scheduling.agent import SchedulingAgent
from navis.reasoning.scheduling.contract import (
    Clarification,
    MeetingConstraints,
    ResolvedAttendee,
    ScheduledProposal,
    SchedulingContext,
    SchedulingResult,
)

__all__ = [
    "Clarification",
    "MeetingConstraints",
    "ResolvedAttendee",
    "ScheduledProposal",
    "SchedulingAgent",
    "SchedulingContext",
    "SchedulingResult",
]
