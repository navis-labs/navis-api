"""Deterministic due-date extraction from commitment language.

"I'll send the deck by Friday" has a deadline. "I'll send the deck" does not.
The difference decides whether Navis can say *"the deck was due Friday"* or only
*"this has been open a while"*, so it has to be read off the text rather than
guessed at.

TWO RULES, both load-bearing:

  1. NEVER INVENT A DATE. A sentence with no deadline returns None, and the
     commitment flows through the undated path instead. A fabricated deadline
     produces a card that tells the founder something false about a third party
     ("Chrissy missed Friday") — the single most damaging kind of error this
     feature can make.

  2. RESOLVE AGAINST THE MESSAGE, NOT AGAINST NOW. "by Friday" in an email sent
     three weeks ago means THAT Friday. Anchoring to the current date would
     silently push every historical deadline into the future and suppress every
     genuine follow-up.

Pure function of (text, sent_at) — same inputs, same answer, every replay.
"""
from __future__ import annotations

import re
from datetime import date, datetime, time, timedelta, timezone

# Whole days are the unit. A promise "by Friday" is met any time that day, so
# the deadline is the END of the named day — treating it as 00:00 would mark a
# commitment overdue while the person still had the day to deliver.
_END_OF_DAY = time(23, 59, 59)

# Straight AND typographic apostrophe, matching signals.text_rules.
_APOS_CLASS = "['’]"

_WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}
_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}
_MONTH_ALT = "|".join(sorted(_MONTHS, key=len, reverse=True))
_WEEKDAY_ALT = "|".join(_WEEKDAYS)

# Ordered most-specific first — the first match wins, so "by next Friday" is not
# read as the bare "friday" rule.
_ISO_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_DAY_MONTH_RE = re.compile(
    rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({_MONTH_ALT})\b", re.IGNORECASE
)
_MONTH_DAY_RE = re.compile(
    rf"\b({_MONTH_ALT})\s+(\d{{1,2}})(?:st|nd|rd|th)?\b", re.IGNORECASE
)
_EOD_RE = re.compile(r"\b(?:eod|end of (?:the )?day|end of play|cob|by tonight)\b", re.IGNORECASE)
_TODAY_RE = re.compile(r"\btoday\b", re.IGNORECASE)
_TOMORROW_RE = re.compile(r"\btomorrow\b|\btomm?orr?ow\b", re.IGNORECASE)
_END_OF_WEEK_RE = re.compile(r"\bend of (?:the )?week\b|\beow\b", re.IGNORECASE)
_NEXT_WEEK_RE = re.compile(r"\bnext week\b", re.IGNORECASE)
_THIS_WEEK_RE = re.compile(r"\bthis week\b", re.IGNORECASE)
_NEXT_WEEKDAY_RE = re.compile(rf"\bnext\s+({_WEEKDAY_ALT})\b", re.IGNORECASE)
_WEEKDAY_RE = re.compile(rf"\b({_WEEKDAY_ALT})\b", re.IGNORECASE)
_IN_N_DAYS_RE = re.compile(r"\bin\s+(\d{1,2})\s+days?\b", re.IGNORECASE)
_IN_N_WEEKS_RE = re.compile(r"\bin\s+(\d{1,2})\s+weeks?\b", re.IGNORECASE)

# A BARE weekday is ambiguous in a way the other forms are not: "I'll send the
# notes from Monday's call" describes when something happened, not when anything
# is due. So the weekday rule — and only the weekday rule — rejects a weekday
# that is possessive or governed by a backward-looking preposition.
#
# "tomorrow" / "end of day" / "next week" need no such guard: they can only
# refer to the future, so requiring a preposition just lost real deadlines
# ("I'll get back to you tomorrow").
_BACKWARD_WEEKDAY_RE = re.compile(
    rf"\b(?:from|since|after|last|during|on)\s+({_WEEKDAY_ALT})\b|"
    rf"\b({_WEEKDAY_ALT}){_APOS_CLASS}s\b",
    re.IGNORECASE,
)


def _eod(d: date) -> datetime:
    return datetime.combine(d, _END_OF_DAY, tzinfo=timezone.utc)


def _forward_to_weekday(anchor: date, weekday: int, *, at_least_next: bool) -> date:
    """The next occurrence of `weekday` on or after `anchor`.

    `at_least_next` forces a full week forward, which is what "next Friday"
    means when the message itself was sent on a Friday.
    """
    delta = (weekday - anchor.weekday()) % 7
    if at_least_next or delta == 0:
        delta = delta or 7
        if at_least_next and delta < 7:
            delta += 7
    return anchor + timedelta(days=delta)


def extract_due_at(sentence: str, *, sent_at: datetime) -> tuple[datetime, str] | None:
    """(due_at, matched_phrase) for a dated commitment, or None.

    None is the honest and common answer — most promises carry no deadline, and
    those are handled by the undated commitment path rather than by inventing
    one here.
    """
    if not sentence:
        return None
    anchor = sent_at.astimezone(timezone.utc).date()

    # Absolute dates first — they need no cue word and no anchor.
    m = _ISO_RE.search(sentence)
    if m:
        try:
            return _eod(date(int(m.group(1)), int(m.group(2)), int(m.group(3)))), m.group(0)
        except ValueError:
            return None
    for regex, order in ((_DAY_MONTH_RE, "dm"), (_MONTH_DAY_RE, "md")):
        m = regex.search(sentence)
        if m:
            day = int(m.group(1) if order == "dm" else m.group(2))
            month = _MONTHS[(m.group(2) if order == "dm" else m.group(1)).lower()]
            resolved = _resolve_bare_month_day(day, month, anchor)
            return (_eod(resolved), m.group(0)) if resolved else None

    if _EOD_RE.search(sentence):
        return _eod(anchor), "end of day"
    if _TOMORROW_RE.search(sentence):
        return _eod(anchor + timedelta(days=1)), "tomorrow"
    if _TODAY_RE.search(sentence):
        return _eod(anchor), "today"
    m = _IN_N_DAYS_RE.search(sentence)
    if m:
        return _eod(anchor + timedelta(days=int(m.group(1)))), m.group(0)
    m = _IN_N_WEEKS_RE.search(sentence)
    if m:
        return _eod(anchor + timedelta(weeks=int(m.group(1)))), m.group(0)
    m = _NEXT_WEEKDAY_RE.search(sentence)
    if m:
        target = _forward_to_weekday(
            anchor, _WEEKDAYS[m.group(1).lower()], at_least_next=True
        )
        return _eod(target), m.group(0)
    if _END_OF_WEEK_RE.search(sentence):
        # Friday of the message's own week.
        return _eod(_forward_to_weekday(anchor, _WEEKDAYS["friday"], at_least_next=False)), "end of week"
    if _NEXT_WEEK_RE.search(sentence):
        # A whole-week promise is met any time that week; Friday is the last
        # moment it can still be kept, so chasing earlier would be premature.
        return _eod(_forward_to_weekday(anchor + timedelta(days=7), 4, at_least_next=False)), "next week"
    if _THIS_WEEK_RE.search(sentence):
        return _eod(_forward_to_weekday(anchor, 4, at_least_next=False)), "this week"
    m = _WEEKDAY_RE.search(sentence)
    if m and not _BACKWARD_WEEKDAY_RE.search(sentence):
        target = _forward_to_weekday(
            anchor, _WEEKDAYS[m.group(1).lower()], at_least_next=False
        )
        return _eod(target), m.group(0)
    return None


def _resolve_bare_month_day(day: int, month: int, anchor: date) -> date | None:
    """Attach a year to a day/month the writer left unqualified.

    Forward from the message date — a promise is about the future — rolling to
    next year only when that day has already passed in the message's own year.
    Returns None for impossible pairs ("31 February") rather than nudging them
    into a real date.
    """
    for year in (anchor.year, anchor.year + 1):
        try:
            candidate = date(year, month, day)
        except ValueError:
            return None
        if candidate >= anchor:
            return candidate
    return None
