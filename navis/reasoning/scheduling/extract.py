"""LLM slot-fill for a scheduling utterance.

Contract:
  * The LLM is asked to return STRICT JSON matching `MeetingConstraints`.
    No prose, no side effects.
  * We validate the response with pydantic. Anything that doesn't parse
    is treated as "no fields extracted" — the agent then asks a
    clarification. The LLM never gets to make destructive decisions.
  * Provider selection uses the existing `navis.ml.providers.registry`
    (Claude / Azure / Gemini / DeterministicProvider fallback) so tests
    hit the deterministic provider automatically.
  * A regex fast-path handles the trivial "schedule a meeting" case
    without an LLM round-trip when no time/date is mentioned at all.
"""
from __future__ import annotations

import json
import re
from datetime import date, datetime, timezone
from typing import Any

from navis.ml.providers.base import GenRequest, MLProvider
from navis.ml.providers.registry import get_provider as get_ml_provider
from navis.platform.monitoring import get_logger
from navis.reasoning.scheduling.contract import MeetingConstraints

logger = get_logger("reasoning.scheduling.extract")


_SYSTEM = (
    "You extract meeting-scheduling constraints from a founder's chat "
    "message. Return ONLY compact JSON — no prose, no code fences. "
    "Every field is optional; use null when the message doesn't "
    "specify. Do NOT invent details. Do NOT use a default value to "
    "'help' — the caller treats null as 'ask the user'.\n\n"
    "Schema (exact keys):\n"
    "  date_constraint: string|null — one of \"today\", \"tomorrow\", "
    "\"monday\"..\"sunday\", \"next_week\", or an ISO date "
    "\"YYYY-MM-DD\". Use the weekday name (e.g. \"tuesday\") when the "
    "user names a weekday. Null when the user didn't mention a day.\n"
    "  time_window: string|null — how to interpret depends on the "
    "user's phrasing:\n"
    "    * FLEXIBLE preset: \"morning\", \"afternoon\", \"evening\" — "
    "      only when the user actually said those words.\n"
    "    * FLEXIBLE range: \"HH:MM-HH:MM\" in 24h — when the user gave "
    "      a start and end (\"between 2 and 4\" → \"14:00-16:00\").\n"
    "    * EXACT time: \"HH:MM\" in 24h — when the user gave a single "
    "      specific time (\"at 3 PM\" → \"15:00\", \"14:00\" → "
    "      \"14:00\"). This is the common case for \"at X\" phrases.\n"
    "    Null when the user did NOT specify a time. Do NOT use \"any\" "
    "    as a default — null is the correct value for 'no time given'.\n"
    "  duration_minutes: integer|null — 15..240 range. Null when the "
    "user didn't specify a duration.\n"
    "  title: string|null — a short subject (<= 80 chars). Never "
    "invent one — leave null when the user didn't hint at a topic.\n"
    "  attendee_hint: string|null — the person's name or handle "
    "exactly as written (\"Rahul\", \"Priya S.\"). Do NOT extract an "
    "email; the caller handles that separately. Null when the user "
    "only uses a pronoun or no name at all.\n"
    "  pronoun: \"him\"|\"her\"|\"them\"|\"that_person\"|null — set "
    "only when the user refers to the attendee via a pronoun."
)


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

# ── Deterministic (regex) extractors ──────────────────────────────────
# These run BEFORE the LLM so the agent still gets a useful slot-fill
# even when the LLM is the DeterministicProvider (offline dev + tests)
# or when the online provider returns invalid JSON. The LLM refines
# on top; regex hits win for pronoun (more reliable) and are the
# fallback for everything else.

_PRONOUN_RE = re.compile(
    r"\b(him|her|them|they|that person|that guy|that lady)\b", re.IGNORECASE
)
_PRONOUN_MAP = {
    "him": "him",
    "her": "her",
    "them": "them",
    "they": "them",
    "that person": "that_person",
    "that guy": "him",
    "that lady": "her",
}

_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
_WEEKDAY_RE = re.compile(
    r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.IGNORECASE,
)
_TODAY_RE = re.compile(r"\btoday\b", re.IGNORECASE)
_TOMORROW_RE = re.compile(r"\btomorrow\b|\btomm?orr?ow\b", re.IGNORECASE)
_NEXT_WEEK_RE = re.compile(r"\bnext\s+week\b", re.IGNORECASE)
_ISO_DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")

# ── Bare day-month dates ("9 August", "August 9", "9th Aug", "Aug 9th") ──
#
# THE 2023 BUG. Nothing here used to match these, so "9 August" fell through to
# the LLM — which was handed no notion of what today is and answered with a year
# out of its training distribution ("2023-08-09"). `_resolve_date` then rejected
# it as "in the past" (or, worse, an off-by-a-year slot got proposed).
#
# The year is NOT something a user omitting it wants a model to guess: it is
# derivable. We resolve it from the caller's clock and roll forward when the
# day has already passed, which is what "9 August" means in every founder's
# head. Deterministic, testable, and it beats the LLM in the merge below.
_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}
_MONTH_ALTERNATION = "|".join(sorted(_MONTHS, key=len, reverse=True))
# "9 August", "9th Aug", "23 Dec"
_DAY_MONTH_RE = re.compile(
    rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({_MONTH_ALTERNATION})\b",
    re.IGNORECASE,
)
# "August 9", "Aug 9th"
_MONTH_DAY_RE = re.compile(
    rf"\b({_MONTH_ALTERNATION})\s+(\d{{1,2}})(?:st|nd|rd|th)?\b",
    re.IGNORECASE,
)
# "9/8" and "9/8/2026" — day-first (the app's founders are IN/UK-centric).
# SLASH ONLY, deliberately: allowing "." here made "a 1.5 hour sync" parse as
# 1 May, and "at 3.30" as 3 March. A slash between two small numbers has no
# competing meaning in a scheduling utterance.
_NUMERIC_DATE_RE = re.compile(r"\b(\d{1,2})/(\d{1,2})(?:/(\d{4}))?\b")


def _resolve_bare_date(day: int, month: int, *, today: date) -> str | None:
    """Attach a year to a (day, month) pair the user left unqualified.

    Current year, rolled to next year when that date has already passed. Returns
    an ISO string (the `date_constraint` wire format) or None when the pair is
    not a real calendar date (e.g. "31 February").
    """
    for year in (today.year, today.year + 1):
        try:
            candidate = date(year, month, day)
        except ValueError:
            return None  # impossible day/month pair — never rolls into validity
        if candidate >= today:
            return candidate.isoformat()
    return None

_MORNING_RE = re.compile(r"\bmorning\b", re.IGNORECASE)
_AFTERNOON_RE = re.compile(r"\bafternoon\b", re.IGNORECASE)
_EVENING_RE = re.compile(r"\bevening\b", re.IGNORECASE)

# Exact times: "3 PM", "3:30 PM", "15:00", "at 3". "at" prefix is
# strongly correlated with an exact time in scheduling utterances.
_EXACT_TIME_RE = re.compile(
    r"\b(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b|"
    r"\b(?:at\s+)?(\d{1,2}):(\d{2})\b",
    re.IGNORECASE,
)

# Ranges: "between 2 and 4", "2-4 PM", "14:00 to 15:30".
_RANGE_RE = re.compile(
    r"\b(?:between\s+)?(\d{1,2}(?::\d{2})?)\s*(?:-|to|and)\s*(\d{1,2}(?::\d{2})?)\s*(am|pm)?\b",
    re.IGNORECASE,
)

# Duration: "30 minutes", "1 hour", "45 min", "45-minute", "1.5 hours".
# Accepts space OR hyphen between the number and the unit — "45-min call"
# and "45 min call" both extract 45.
_DURATION_RE = re.compile(
    r"\b(\d+(?:\.\d+)?)[\s\-]*(hours?|hrs?|h|minutes?|mins?|m)\b",
    re.IGNORECASE,
)

# "with X" name extractor — matches capitalized run of tokens right
# after "with". Skips words that are clearly not names (today,
# tomorrow, weekdays, morning/afternoon/etc). Conservative so it
# doesn't over-fire on descriptive text.
_WITH_NAME_RE = re.compile(
    r"\bwith\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)\b"
)
_NON_NAME_TOKENS = {
    "Today", "Tomorrow", "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday", "Morning", "Afternoon", "Evening",
    "Next", "This",
}


def _extract_date(utterance: str, *, today: date) -> str | None:
    """Return a normalized date_constraint or None. See MeetingConstraints.

    `today` is the caller's local date — every implicit year in this function is
    derived from it rather than assumed, which is the fix for dates like
    "9 August" resolving into a past year.
    """
    if _TODAY_RE.search(utterance):
        return "today"
    if _TOMORROW_RE.search(utterance):
        return "tomorrow"
    if _NEXT_WEEK_RE.search(utterance):
        return "next_week"
    m = _WEEKDAY_RE.search(utterance)
    if m:
        return m.group(1).lower()
    # Explicit ISO date beats the bare forms — the user spelled out a year.
    m = _ISO_DATE_RE.search(utterance)
    if m:
        return m.group(1)
    m = _DAY_MONTH_RE.search(utterance)
    if m:
        resolved = _resolve_bare_date(
            int(m.group(1)), _MONTHS[m.group(2).lower()], today=today
        )
        if resolved:
            return resolved
    m = _MONTH_DAY_RE.search(utterance)
    if m:
        resolved = _resolve_bare_date(
            int(m.group(2)), _MONTHS[m.group(1).lower()], today=today
        )
        if resolved:
            return resolved
    m = _NUMERIC_DATE_RE.search(utterance)
    if m:
        day, month, year = int(m.group(1)), int(m.group(2)), m.group(3)
        if year:
            try:
                return date(int(year), month, day).isoformat()
            except ValueError:
                return None
        resolved = _resolve_bare_date(day, month, today=today)
        if resolved:
            return resolved
    return None


def _extract_time_window(utterance: str) -> str | None:
    """Return a normalized time_window or None.

    Priority: range > exact > preset. Ranges and exacts are more
    specific than presets, so if the user said "afternoon between 2
    and 4" we return the 2-4 range.
    """
    # Explicit range wins.
    r = _RANGE_RE.search(utterance)
    if r:
        left_raw, right_raw, meridiem = r.group(1), r.group(2), (r.group(3) or "").lower()
        left = _normalise_hhmm(left_raw, meridiem)
        right = _normalise_hhmm(right_raw, meridiem)
        if left and right and left != right:
            return f"{left}-{right}"
    # Exact time.
    t = _EXACT_TIME_RE.search(utterance)
    if t:
        # Group 1-3 = 12-hour form; group 4-5 = 24-hour form.
        if t.group(3):
            hh, mm, meridiem = t.group(1), (t.group(2) or "0"), t.group(3).lower()
            hhmm = _normalise_hhmm(hh + ":" + mm, meridiem)
            if hhmm:
                return hhmm
        if t.group(4):
            hh = int(t.group(4))
            mm = int(t.group(5))
            if 0 <= hh <= 23 and 0 <= mm <= 59:
                return f"{hh:02d}:{mm:02d}"
    # Presets.
    if _MORNING_RE.search(utterance):
        return "morning"
    if _AFTERNOON_RE.search(utterance):
        return "afternoon"
    if _EVENING_RE.search(utterance):
        return "evening"
    return None


def _normalise_hhmm(raw: str, meridiem: str) -> str | None:
    """Turn '3' or '3:30' plus an optional 'am/pm' into 'HH:MM'."""
    hh_str, _, mm_str = raw.partition(":")
    try:
        hh = int(hh_str)
        mm = int(mm_str) if mm_str else 0
    except ValueError:
        return None
    if meridiem:
        if hh < 1 or hh > 12:
            return None
        if hh == 12:
            hh = 0
        if meridiem == "pm":
            hh += 12
    if not (0 <= hh <= 23 and 0 <= mm <= 59):
        return None
    return f"{hh:02d}:{mm:02d}"


def _extract_duration_minutes(utterance: str) -> int | None:
    m = _DURATION_RE.search(utterance)
    if not m:
        return None
    try:
        n = float(m.group(1))
    except ValueError:
        return None
    unit = m.group(2).lower()
    if unit.startswith("h"):
        mins = int(round(n * 60))
    else:
        mins = int(round(n))
    if 15 <= mins <= 240:
        return mins
    return None


def _extract_attendee_hint(utterance: str) -> str | None:
    m = _WITH_NAME_RE.search(utterance)
    if not m:
        return None
    candidate = m.group(1).strip()
    # Reject "with Tomorrow", "with Monday", "with Afternoon" etc.
    first_token = candidate.split()[0]
    if first_token in _NON_NAME_TOKENS:
        return None
    return candidate


def _deterministic_prefill(utterance: str, *, today: date) -> dict[str, Any]:
    """Deterministic slot fill from an utterance.

    Runs BEFORE the LLM. Every slot is best-effort — the LLM refines on
    top, but if the LLM is offline/fails these values are the fallback.
    Pronoun detection here means a bad/missing LLM response still gives
    the agent enough to know it's a pronoun-referential utterance and
    should route through the signal-context resolver instead of asking
    "who?" immediately.
    """
    pronoun = None
    m = _PRONOUN_RE.search(utterance)
    if m:
        pronoun = _PRONOUN_MAP.get(m.group(1).lower())
    return {
        "pronoun": pronoun,
        "date_constraint": _extract_date(utterance, today=today),
        "time_window": _extract_time_window(utterance),
        "duration_minutes": _extract_duration_minutes(utterance),
        "attendee_hint": _extract_attendee_hint(utterance),
    }


def _dated_system_prompt(today: date) -> str:
    """`_SYSTEM` with a concrete "today" bolted on.

    A model with no clock has to invent one, and it invents the year it saw most
    during training — which is how "9 August" became 2023-08-09 in a 2026
    session. The regex layer above now wins for that shape anyway, but the
    anchor still matters for phrasings only the LLM handles ("the Friday after
    next"), and it costs a single line of prompt.
    """
    return (
        f"{_SYSTEM}\n\n"
        f"Today is {today.isoformat()} ({today.strftime('%A')}). Resolve every "
        "relative or partial date against THAT date. When the user gives a day "
        "and month without a year, use the next occurrence — never a past year."
    )


async def extract_constraints(
    utterance: str,
    *,
    provider: MLProvider | None = None,
    now: datetime | None = None,
) -> MeetingConstraints:
    """Run the LLM slot-fill. Returns a validated MeetingConstraints.

    `now` anchors every implicit date. It defaults to the current UTC time so
    existing callers keep working, but the agent passes its own clock through
    so tests are deterministic and the parser agrees with the availability
    stage about what "today" means.

    On ANY failure (network, invalid JSON, schema violation) we return a
    constraints object with just the deterministic prefill applied —
    the agent treats this as "clarify".
    """
    today = (now or datetime.now(timezone.utc)).date()
    prefill = _deterministic_prefill(utterance, today=today)
    llm = provider or get_ml_provider()
    try:
        result = await llm.generate(
            GenRequest(
                system=_dated_system_prompt(today),
                user=utterance,
                # Slot-fill JSON is short; keep the model tight.
                max_tokens=256,
                temperature=0.0,
                prompt_id="scheduling.extract",
                # Bumped: the system prompt now carries a date anchor, so
                # cached/eval'd completions from v1 are not comparable.
                prompt_version=2,
            )
        )
    except Exception:  # broad on purpose — any LLM failure clarifies
        logger.exception("scheduling.extract LLM call failed")
        return MeetingConstraints(**prefill)

    parsed = _parse_json_payload(result.text)
    if not parsed:
        # Deterministic providers (used in tests + as offline fallback)
        # return non-JSON prose. The regex prefill IS the extraction in
        # that case — no clarification is thrown away that shouldn't be.
        return MeetingConstraints(**prefill)

    # Merge: regex prefill wins for every slot it detected (regex is
    # deterministic and cheaper to reason about); LLM fills whatever
    # regex left null. This ordering means an online LLM cannot flip
    # a clearly-stated "tomorrow" back to null on us — nor a clearly-stated
    # "9 August" into some other year.
    merged = {**parsed}
    for key, value in prefill.items():
        if value is not None:
            merged[key] = value
    merged["date_constraint"] = _sanitise_llm_date(
        merged.get("date_constraint"), today=today
    )
    try:
        return MeetingConstraints.model_validate(merged)
    except Exception:
        logger.exception("scheduling.extract validation failed: %r", merged)
        return MeetingConstraints(**prefill)


def _sanitise_llm_date(value: Any, *, today: date) -> Any:
    """Last line of defence against a model-authored year.

    Only touches ISO dates that came back from the LLM. A date in a PAST year
    is never something a founder asked for, so rather than pass it downstream
    (where it becomes a confusing "that day is in the past" clarification, or —
    before the availability guard existed — a slot in the wrong year) we lift it
    onto the same month/day in the next occurring year. Dates the user could
    plausibly mean (today onward) pass through untouched, as do the symbolic
    constraints ("tomorrow", "friday", "next_week").
    """
    if not isinstance(value, str):
        return value
    raw = value.strip()
    try:
        parsed = date.fromisoformat(raw)
    except ValueError:
        return value  # symbolic constraint — validated downstream
    if parsed >= today:
        return raw
    rolled = _resolve_bare_date(parsed.day, parsed.month, today=today)
    if rolled is None:
        return raw
    logger.warning(
        "scheduling.extract: model returned past date %s; rolled to %s", raw, rolled
    )
    return rolled


def _parse_json_payload(text: str) -> dict[str, Any] | None:
    """Extract the first {...} object from the model's response.

    Some providers wrap JSON in Markdown fences or add trailing prose;
    a permissive extract-then-loads keeps us robust without needing
    provider-specific parsing.
    """
    if not text:
        return None
    text = text.strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    match = _JSON_OBJECT_RE.search(text)
    if match is None:
        return None
    try:
        obj = json.loads(match.group(0))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None
