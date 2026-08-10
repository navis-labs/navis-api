"""Calendar-query intent detector + time-scope parser.

Positive matches — "what meetings do I have today?", "list my meetings
this week", "prepare me for my next meeting", "any meetings today?".

Negative — anything that's actually a scheduling REQUEST ("fix a
meeting with X", "schedule tomorrow at 3 PM") is caught by
`scheduling.intent.is_scheduling_intent` FIRST and never reaches this
detector. Ambiguous verbs like "schedule" are safe here because the
scheduling detector requires a create-verb + noun pairing whereas this
detector requires a query verb + noun pairing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from enum import Enum
from zoneinfo import ZoneInfo


# ── Intent detection ───────────────────────────────────────────────────

# Query verbs / phrases that indicate the user is ASKING about
# existing calendar entries, not creating one.
_QUERY_VERB_RE = re.compile(
    r"\b("
    r"list|show|tell|prepare|preview|remind|explain|"
    r"what|whats|what's|which|when|how\s+many|"
    r"any|is\s+there|are\s+there|do\s+i\s+have|do\s+we\s+have"
    r")\b",
    re.IGNORECASE,
)

# Calendar-noun set — meetings, calls, syncs, events, appointments,
# calendar itself. Standalone "schedule" as a NOUN also counts
# ("my schedule today").
_CALENDAR_NOUN_RE = re.compile(
    r"\b("
    r"meeting|meetings|call|calls|sync|syncs|catch\s*up|"
    r"1:1|one[- ]?on[- ]?one|standup|kickoff|check[- ]?in|debrief|"
    r"event|events|appointment|appointments|"
    r"calendar|agenda|schedule"
    r")\b",
    re.IGNORECASE,
)


def is_calendar_query_intent(utterance: str) -> bool:
    """True iff the utterance asks about EXISTING calendar entries.

    Requires (query-verb + calendar-noun) both present. Not scheduling
    (create/book/fix) — those match `scheduling.intent` first.
    """
    if not utterance or not utterance.strip():
        return False
    return bool(
        _QUERY_VERB_RE.search(utterance)
        and _CALENDAR_NOUN_RE.search(utterance)
    )


# ── Time-scope parsing ─────────────────────────────────────────────────


class TimeScope(str, Enum):
    """What span the user asked about."""

    TODAY = "today"
    TOMORROW = "tomorrow"
    THIS_WEEK = "this_week"
    NEXT_WEEK = "next_week"
    NEXT_MEETING = "next_meeting"
    ANY = "any"  # default when no explicit scope — treated as "today"


_TODAY_RE = re.compile(r"\btoday\b|\bthis\s+day\b", re.IGNORECASE)
_TOMORROW_RE = re.compile(
    # Common typos: tomorrow, tomorow, tommorrow, tommorow, tommrow,
    # tomrow, tmrw, tmr. The `o` between the m-cluster and the r-cluster
    # is optional too so "tommrow" (no vowel there) still matches.
    r"\btomm?o?rr?ow\b|\btmrw\b|\btmr\b",
    re.IGNORECASE,
)
_THIS_WEEK_RE = re.compile(r"\bthis\s+week\b", re.IGNORECASE)
_NEXT_WEEK_RE = re.compile(r"\bnext\s+week\b", re.IGNORECASE)
_NEXT_MEETING_RE = re.compile(
    r"\b(next|upcoming|coming\s+up)\s+(meeting|call|sync|event|appointment)\b|"
    r"\bmy\s+next\s+meeting\b",
    re.IGNORECASE,
)


def parse_time_scope(utterance: str) -> TimeScope:
    """Deterministic scope classification. Priority: most-specific wins."""
    if _NEXT_MEETING_RE.search(utterance):
        return TimeScope.NEXT_MEETING
    if _TOMORROW_RE.search(utterance):
        return TimeScope.TOMORROW
    if _THIS_WEEK_RE.search(utterance):
        return TimeScope.THIS_WEEK
    if _NEXT_WEEK_RE.search(utterance):
        return TimeScope.NEXT_WEEK
    if _TODAY_RE.search(utterance):
        return TimeScope.TODAY
    return TimeScope.ANY


@dataclass(frozen=True, slots=True)
class ResolvedWindow:
    """Concrete UTC window derived from a TimeScope + user's timezone."""

    scope: TimeScope
    time_min_utc: datetime
    time_max_utc: datetime
    max_events: int
    label: str  # human-readable ("today", "this week", "next 7 days")


def resolve_window(
    scope: TimeScope, *, now: datetime, time_zone: str
) -> ResolvedWindow:
    """Turn a TimeScope + user tz into UTC start/end for calendar API.

    Windows are LOCAL-day-aligned (start of the local day, end of the
    local day + N) so "today" doesn't spill into yesterday because of
    UTC offset.
    """
    try:
        tz = ZoneInfo(time_zone)
    except Exception:  # noqa: BLE001
        tz = ZoneInfo("UTC")
    now_local = now.astimezone(tz)
    local_today = now_local.date()

    if scope is TimeScope.TOMORROW:
        target = local_today + timedelta(days=1)
        start_local = datetime.combine(target, time(0, 0), tzinfo=tz)
        end_local = datetime.combine(target, time(23, 59, 59), tzinfo=tz)
        return ResolvedWindow(
            scope=scope,
            time_min_utc=start_local.astimezone(timezone.utc),
            time_max_utc=end_local.astimezone(timezone.utc),
            max_events=25,
            label="tomorrow",
        )
    if scope is TimeScope.THIS_WEEK:
        # Today → end of next 7 days.
        start_local = datetime.combine(local_today, time(0, 0), tzinfo=tz)
        end_local = start_local + timedelta(days=7)
        return ResolvedWindow(
            scope=scope,
            time_min_utc=start_local.astimezone(timezone.utc),
            time_max_utc=end_local.astimezone(timezone.utc),
            max_events=50,
            label="this week",
        )
    if scope is TimeScope.NEXT_WEEK:
        start_local = datetime.combine(
            local_today + timedelta(days=7), time(0, 0), tzinfo=tz
        )
        end_local = start_local + timedelta(days=7)
        return ResolvedWindow(
            scope=scope,
            time_min_utc=start_local.astimezone(timezone.utc),
            time_max_utc=end_local.astimezone(timezone.utc),
            max_events=50,
            label="next week",
        )
    if scope is TimeScope.NEXT_MEETING:
        # From now → next 7 days, capped to 1 event.
        end_local = now_local + timedelta(days=7)
        return ResolvedWindow(
            scope=scope,
            time_min_utc=now.astimezone(timezone.utc),
            time_max_utc=end_local.astimezone(timezone.utc),
            max_events=1,
            label="upcoming",
        )
    # TODAY / ANY — default to today.
    start_local = datetime.combine(local_today, time(0, 0), tzinfo=tz)
    end_local = datetime.combine(local_today, time(23, 59, 59), tzinfo=tz)
    # If "any", start from now (no point listing meetings that already ended today).
    if scope is TimeScope.ANY:
        start_local = max(start_local, now_local)
    return ResolvedWindow(
        scope=scope,
        time_min_utc=start_local.astimezone(timezone.utc),
        time_max_utc=end_local.astimezone(timezone.utc),
        max_events=25,
        label="today",
    )
