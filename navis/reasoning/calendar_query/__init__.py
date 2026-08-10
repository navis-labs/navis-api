"""Calendar-query reasoning — "what meetings do I have today?".

Complements the SchedulingAgent (which CREATES meetings) with a
read-only path that lists / previews EXISTING meetings by hitting
Google Calendar directly. This bypasses the RAG grounded-answer
pipeline because calendar events aren't reliably present in memory
(they're synced via Gmail invite notifications, not indexed as
first-class facts) — and even when they are, the LLM composes vague
answers over sparse event metadata.

Route order in the chat endpoint:

  1. is_scheduling_intent(u) → SchedulingAgent
  2. is_calendar_query_intent(u) → CalendarQueryAgent (this module)
  3. fallback → RAG grounded_answer
"""
from navis.reasoning.calendar_query.handler import (
    CalendarQueryAgent,
    CalendarQueryContext,
    CalendarQueryResult,
)
from navis.reasoning.calendar_query.intent import (
    TimeScope,
    is_calendar_query_intent,
    parse_time_scope,
)

__all__ = [
    "CalendarQueryAgent",
    "CalendarQueryContext",
    "CalendarQueryResult",
    "TimeScope",
    "is_calendar_query_intent",
    "parse_time_scope",
]
