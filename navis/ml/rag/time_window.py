"""Parse relative time expressions in a user question into a UTC (since, until) window.

Vector search alone can't answer "how many meetings this week" — semantic similarity
returns memories mentioning meetings regardless of date. The retriever needs to be told
which window to look in. This module extracts that window from natural language, so
`ContextBuilder` can narrow the candidate set before the LLM ever sees it.

Vocabulary is intentionally small (today / yesterday / tomorrow / this|last|next week /
this|last month / this quarter / last N days). Anything unrecognized returns `None`,
and the retriever falls back to unfiltered semantic search — the same behaviour as
before this parser existed.

Windows are computed in UTC. Multi-timezone org support is a follow-up; day-boundary
drift for users far from UTC is a known limitation until then.
"""
from __future__ import annotations

import re
from calendar import monthrange
from datetime import datetime, timedelta, timezone

_WEEK = re.compile(r"\b(this|last|next|past|current)\s+week\b", re.IGNORECASE)
_MONTH = re.compile(r"\b(this|last|next|past|current)\s+month\b", re.IGNORECASE)
_QUARTER = re.compile(r"\b(this|last|next|past|current)\s+quarter\b", re.IGNORECASE)
_TODAY = re.compile(r"\btoday\b", re.IGNORECASE)
_YESTERDAY = re.compile(r"\byesterday\b", re.IGNORECASE)
# Common shorthand: tomm, tomm., tmrw, tmr, tmw, tomo, tomoro.
_TOMORROW = re.compile(
    r"\b(tomorrow|tomorow|tommorow|tommorrow|tomm(?:orow|orrow)?|tomo(?:ro)?|tmrw|tmr|tmw)\b",
    re.IGNORECASE,
)
_LAST_N_DAYS = re.compile(r"\b(?:last|past)\s+(\d{1,3})\s+days?\b", re.IGNORECASE)
_NEXT_N_DAYS = re.compile(r"\b(?:next|coming|upcoming)\s+(\d{1,3})\s+days?\b", re.IGNORECASE)
# Open-ended "in coming days", "upcoming days", "next few days" — no number given.
# Default to a 7-day forward window so the retriever excludes past events.
_NEXT_OPEN = re.compile(
    r"\b(?:in\s+(?:the\s+)?)?(?:coming|upcoming|next\s+few)\s+days?\b|\bsoon\b",
    re.IGNORECASE,
)
_LAST_OPEN = re.compile(
    r"\b(?:in\s+(?:the\s+)?)?(?:past|last)\s+few\s+days?\b|\brecently\b",
    re.IGNORECASE,
)
_DEFAULT_OPEN_DAYS = 7


def _day_bounds(d: datetime) -> tuple[datetime, datetime]:
    start = d.replace(hour=0, minute=0, second=0, microsecond=0)
    return start, start + timedelta(days=1)


def _week_bounds(d: datetime) -> tuple[datetime, datetime]:
    """Monday 00:00 → next Monday 00:00, UTC."""
    monday = d - timedelta(days=d.weekday())
    monday = monday.replace(hour=0, minute=0, second=0, microsecond=0)
    return monday, monday + timedelta(days=7)


def _month_bounds(d: datetime) -> tuple[datetime, datetime]:
    start = d.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    _, last_day = monthrange(start.year, start.month)
    end = start.replace(day=last_day) + timedelta(days=1)
    return start, end


def _shift_month(d: datetime, delta: int) -> datetime:
    year = d.year + (d.month - 1 + delta) // 12
    month = (d.month - 1 + delta) % 12 + 1
    return d.replace(year=year, month=month, day=1)


def _quarter_bounds(d: datetime) -> tuple[datetime, datetime]:
    q_start_month = ((d.month - 1) // 3) * 3 + 1
    start = d.replace(month=q_start_month, day=1, hour=0, minute=0, second=0, microsecond=0)
    end = _shift_month(start, 3)
    return start, end


def parse_time_window(
    question: str, *, now: datetime | None = None
) -> tuple[datetime, datetime] | None:
    """Return a `(since, until)` UTC window if `question` names a time range.

    `until` is exclusive (half-open interval), matching how SQL range predicates
    are usually expressed. Returns `None` when no recognized expression is found —
    callers should then run unfiltered search.
    """
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)

    m = _LAST_N_DAYS.search(question)
    if m:
        days = int(m.group(1))
        start_today, end_today = _day_bounds(now)
        return start_today - timedelta(days=days), end_today

    m = _NEXT_N_DAYS.search(question)
    if m:
        days = int(m.group(1))
        start_today, _ = _day_bounds(now)
        return start_today, start_today + timedelta(days=days + 1)

    # Open-ended forward window ("coming days", "soon"). Without this,
    # unfiltered semantic search returns past events for future-tense questions.
    if _NEXT_OPEN.search(question):
        start_today, _ = _day_bounds(now)
        return start_today, start_today + timedelta(days=_DEFAULT_OPEN_DAYS + 1)

    if _LAST_OPEN.search(question):
        start_today, end_today = _day_bounds(now)
        return start_today - timedelta(days=_DEFAULT_OPEN_DAYS), end_today

    if _TODAY.search(question):
        return _day_bounds(now)
    if _YESTERDAY.search(question):
        s, e = _day_bounds(now - timedelta(days=1))
        return s, e
    if _TOMORROW.search(question):
        s, e = _day_bounds(now + timedelta(days=1))
        return s, e

    m = _WEEK.search(question)
    if m:
        which = m.group(1).lower()
        base = now
        if which == "last":
            base = now - timedelta(days=7)
        elif which == "next":
            base = now + timedelta(days=7)
        return _week_bounds(base)

    m = _MONTH.search(question)
    if m:
        which = m.group(1).lower()
        base = now
        if which == "last":
            base = _shift_month(now, -1)
        elif which == "next":
            base = _shift_month(now, 1)
        return _month_bounds(base)

    m = _QUARTER.search(question)
    if m:
        which = m.group(1).lower()
        base = now
        if which == "last":
            s, _ = _quarter_bounds(now)
            base = s - timedelta(days=1)
        elif which == "next":
            _, e = _quarter_bounds(now)
            base = e
        return _quarter_bounds(base)

    return None
