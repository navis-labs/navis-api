"""CalendarOpsAgent — reschedule / cancel an EXISTING meeting.

The missing half of the calendar story. `SchedulingAgent` creates events and
`CalendarQueryAgent` reads them; every request to CHANGE one used to fall through
to RAG-over-memory and come back as "Navis doesn't have enough memory", even
though the event was sitting in Google Calendar.

Same contract as the scheduling agent: `propose()` returns exactly one of
`CalendarOpsClarification` | `MeetingMutationProposal`, and NEVER performs the
mutation. The chat router turns a proposal into an ExecutionRequest; the
execution worker is the only thing that touches Google.

Targeting rules, in order — the agent never guesses which meeting you meant:
  1. An event id carried over from a prior turn's disambiguation.
  2. Exactly one event in the resolved window → that one.
  3. A name / title / attendee hint in the utterance narrowing to exactly one.
  4. More than one → a pick-list clarification.
  5. None → an honest "you have nothing scheduled then".
"""
from __future__ import annotations

import re
import uuid
from datetime import datetime, timedelta
from typing import Any, Callable

from sqlalchemy.ext.asyncio import AsyncSession
from zoneinfo import ZoneInfo

from navis.platform.monitoring import get_logger
from navis.reasoning.calendar_ops.contract import (
    CalendarOpsClarification,
    CalendarOpsContext,
    CalendarOpsResult,
    MeetingMutationProposal,
    TargetEvent,
)
from navis.reasoning.calendar_query.handler import CalendarEvent, _parse_events
from navis.reasoning.calendar_query.intent import parse_time_scope, resolve_window
from navis.reasoning.routing import normalise_utterance
from navis.reasoning.scheduling.availability import (
    DEFAULT_DURATION_MIN,
    CalendarClientProtocol,
    ScopeError,
    _parse_time_token,
    open_calendar_client,
)
from navis.reasoning.scheduling.extract import (
    _extract_date,
    _extract_duration_minutes,
    _extract_time_window,
)
from navis.signals.google.resolve import CalendarNotConnected

logger = get_logger("reasoning.calendar_ops.agent")

# How far ahead to look for the meeting being changed when the utterance names
# no time scope. Two weeks covers "the meeting" in every realistic founder
# usage without dragging in next quarter's board offsite.
_DEFAULT_LOOKAHEAD_DAYS = 14
_MAX_CANDIDATES = 6

# Words that are never a useful title/attendee filter — they are the scaffolding
# of the request itself.
_STOPWORDS = {
    "the", "my", "our", "a", "an", "this", "that", "it", "them", "to", "at",
    "on", "for", "with", "and", "or", "of", "in", "from", "please", "can",
    "you", "i", "we", "me", "reschedule", "schedule", "rebook", "move", "push",
    "shift", "postpone", "bump", "cancel", "delete", "remove", "drop", "call",
    "off", "meeting", "meetings", "meet", "sync", "chat", "invite", "event",
    "appointment", "today", "tomorrow", "week", "next", "am", "pm", "time",
    "into", "till", "until", "by", "make", "change", "put", "set", "new",
}
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'._%+\-]*(?:@[A-Za-z0-9.\-]+)?")


class CalendarOpsAgent:
    """Resolve WHICH meeting, then WHAT change. Injectable calendar for tests."""

    def __init__(
        self,
        db: AsyncSession | None,
        *,
        calendar_client_factory: Callable[
            [AsyncSession, uuid.UUID], "tuple[CalendarClientProtocol, object]"
        ]
        | None = None,
    ) -> None:
        self.db = db
        self._factory = calendar_client_factory or open_calendar_client

    async def propose(self, ctx: CalendarOpsContext) -> CalendarOpsResult:
        utterance = normalise_utterance(ctx.utterance)

        try:
            client, transport = await self._factory(self.db, ctx.org_id)
        except CalendarNotConnected:
            return CalendarOpsClarification(
                field="scope",
                question=(
                    "Google Calendar isn't connected yet. Connect Google from "
                    "the Sources page and I can move or cancel meetings for you."
                ),
            )
        except RuntimeError as exc:
            logger.info("calendar_ops: calendar unusable org=%s: %s", ctx.org_id, exc)
            return CalendarOpsClarification(
                field="scope",
                question=(
                    "Google Calendar is connected, but not with permission to "
                    "change events. Disconnect and reconnect Google Workspace "
                    "to grant it."
                ),
            )

        try:
            events = await self._load_candidates(client, ctx, utterance)
        except Exception as exc:  # noqa: BLE001 — Google is external
            logger.exception("calendar_ops: event lookup failed org=%s", ctx.org_id)
            return CalendarOpsClarification(
                field="scope",
                question=_google_error_message(str(exc)),
            )
        finally:
            if hasattr(transport, "aclose"):
                try:
                    await transport.aclose()
                except Exception:  # noqa: BLE001
                    logger.exception("calendar_ops: transport close failed")

        # ── Which meeting? ────────────────────────────────────────────
        if ctx.target_event_id:
            chosen = next(
                (e for e in events if e.id == ctx.target_event_id), None
            )
            if chosen is None:
                return CalendarOpsClarification(
                    field="none",
                    question=(
                        "That meeting is no longer on your calendar — it may "
                        "have already been moved or cancelled."
                    ),
                )
        else:
            narrowed = _narrow(events, utterance)
            if not narrowed:
                return CalendarOpsClarification(
                    field="none", question=_nothing_found_message(events, utterance)
                )
            if len(narrowed) > 1:
                return CalendarOpsClarification(
                    field="event",
                    question=(
                        f"You have {len(narrowed)} meetings that could match. "
                        "Which one?"
                    ),
                    options=[
                        _event_option(e, ctx.time_zone)
                        for e in narrowed[:_MAX_CANDIDATES]
                    ],
                )
            chosen = narrowed[0]

        target = TargetEvent(
            event_id=chosen.id,
            title=chosen.title,
            start=chosen.start_utc,
            end=chosen.end_utc,
            time_zone=ctx.time_zone,
            attendees=list(chosen.attendees),
        )

        if ctx.operation == "cancel":
            return MeetingMutationProposal(operation="cancel", target=target)

        # ── Reschedule: resolve the NEW slot ─────────────────────────
        return _reschedule_proposal(target, ctx, utterance)

    async def _load_candidates(
        self,
        client: CalendarClientProtocol,
        ctx: CalendarOpsContext,
        utterance: str,
    ) -> list[CalendarEvent]:
        """Events the mutation could plausibly target.

        Honours an explicit scope in the utterance ("cancel tomorrow's standup")
        and otherwise scans a fixed forward window. Past events are excluded:
        moving or cancelling a meeting that already happened is never what was
        meant, and offering it as an option is noise.
        """
        scope = parse_time_scope(utterance)
        window = resolve_window(scope, now=ctx.now, time_zone=ctx.time_zone)
        # `parse_time_scope` returns ANY (=today) when nothing was named. For a
        # mutation that is too narrow — "reschedule the investor call" rarely
        # means one scheduled in the next few hours. Widen to the lookahead.
        time_min = ctx.now
        time_max = window.time_max_utc
        if scope.value in {"any", "today"}:
            time_max = ctx.now + timedelta(days=_DEFAULT_LOOKAHEAD_DAYS)
        else:
            time_min = max(ctx.now, window.time_min_utc)

        fn = getattr(client, "list_events_expanded", None)
        if fn is None:  # pragma: no cover — protocol guarantees this
            raise RuntimeError("calendar client missing list_events_expanded")
        raw: dict[str, Any] = await fn(
            time_min=time_min.isoformat(),
            time_max=time_max.isoformat(),
            max_results=50,
        )
        events = _parse_events(raw)
        return sorted(
            (e for e in events if e.end_utc > ctx.now), key=lambda e: e.start_utc
        )


# ── Targeting ──────────────────────────────────────────────────────────


def _narrow(events: list[CalendarEvent], utterance: str) -> list[CalendarEvent]:
    """Filter candidates by any title/attendee hint in the utterance.

    Returns everything when the utterance carries no usable hint — a single
    upcoming meeting then resolves on its own, and several produce a pick-list.
    """
    if not events:
        return []
    hints = {
        w.lower()
        for w in _WORD_RE.findall(utterance)
        if len(w) > 2 and w.lower() not in _STOPWORDS
    }
    if not hints:
        return events
    matched = [e for e in events if _matches(e, hints)]
    # A hint that matches nothing is more likely a phrasing quirk than a real
    # filter ("move the thing with the investors"). Fall back to the full set so
    # the user gets a pick-list instead of a false "nothing found".
    return matched or events


def _matches(event: CalendarEvent, hints: set[str]) -> bool:
    haystack = " ".join([event.title.lower(), *(a.lower() for a in event.attendees)])
    return any(h in haystack for h in hints)


def _event_option(event: CalendarEvent, viewer_tz: str) -> dict:
    return {
        "event_id": event.id,
        "label": f"{event.title} — {event.render_date(viewer_tz)} · "
                 f"{event.render_time(viewer_tz)}",
        "title": event.title,
        "start": event.start_utc.isoformat(),
        "end": event.end_utc.isoformat(),
        "attendees": list(event.attendees),
    }


def _nothing_found_message(events: list[CalendarEvent], utterance: str) -> str:
    if not events:
        return (
            "You have no upcoming meetings on your calendar in the next "
            f"{_DEFAULT_LOOKAHEAD_DAYS} days, so there's nothing to change."
        )
    return (
        "I couldn't match that to any of your upcoming meetings. Tell me the "
        "meeting title or who it's with and I'll find it."
    )


def _google_error_message(err: str) -> str:
    lowered = err.lower()
    if (
        "insufficientpermissions" in lowered
        or "access_token_scope_insufficient" in lowered
        or "insufficient authentication scopes" in lowered
    ):
        return (
            "Google Calendar rejected the read — disconnect and reconnect "
            "Google Workspace so the calendar scope is granted."
        )
    return "I couldn't reach Google Calendar just now. Try again in a moment."


# ── Reschedule slot resolution ─────────────────────────────────────────


def _reschedule_proposal(
    target: TargetEvent, ctx: CalendarOpsContext, utterance: str
) -> CalendarOpsResult:
    """Turn "to 5-6 PM" / "to Friday 10am" into a concrete new slot.

    Keeps the event's existing DAY when the user only gave a time, and its
    existing DURATION when they only gave a start — moving a meeting should
    change exactly what was asked and nothing else.
    """
    try:
        tz = ZoneInfo(ctx.time_zone)
    except Exception:  # noqa: BLE001
        tz = ZoneInfo("UTC")

    local_now = ctx.now.astimezone(tz)
    time_window = _extract_time_window(utterance)
    date_constraint = _extract_date(utterance, today=local_now.date())
    existing_minutes = max(
        int((target.end - target.start).total_seconds() // 60), 1
    )
    duration = (
        _extract_duration_minutes(utterance)
        or existing_minutes
        or DEFAULT_DURATION_MIN
    )

    if time_window is None and date_constraint is None:
        return CalendarOpsClarification(
            field="time",
            question=(
                f'When should I move "{target.title}" to? '
                "Give me a day and a time."
            ),
            target_event_id=target.event_id,
        )

    # Day: the one the user named, else the day it is already on.
    try:
        target_date = (
            _resolve_op_date(date_constraint, now=ctx.now, tz=tz)
            if date_constraint
            else target.start.astimezone(tz).date()
        )
    except ScopeError as exc:
        return CalendarOpsClarification(
            field="time",
            question=f"That day doesn't work ({exc}). Which day should I move it to?",
            target_event_id=target.event_id,
        )

    if time_window is None:
        # Day given, no time: keep the existing time-of-day on the new date.
        local_start = datetime.combine(
            target_date, target.start.astimezone(tz).timetz()
        )
    else:
        try:
            start_time, explicit_minutes = _parse_new_time(time_window)
        except ScopeError as exc:
            return CalendarOpsClarification(
                field="time",
                question=f"I couldn't read that time ({exc}). What time should I move it to?",
                target_event_id=target.event_id,
            )
        if explicit_minutes is not None:
            duration = explicit_minutes
        local_start = datetime.combine(target_date, start_time, tzinfo=tz)

    local_end = local_start + timedelta(minutes=duration)
    if local_end <= local_now:
        return CalendarOpsClarification(
            field="time",
            question=(
                "That time has already passed. What time should I move "
                f'"{target.title}" to?'
            ),
            target_event_id=target.event_id,
        )

    assumptions: list[str] = []
    if time_window is not None and date_constraint is None:
        assumptions.append(
            f"Kept it on {target.start.astimezone(tz).strftime('%a, %b %d')} — "
            "say a day if you meant a different one."
        )
    if _extract_duration_minutes(utterance) is None and duration == existing_minutes:
        assumptions.append(f"Kept the existing {existing_minutes}-min length.")

    return MeetingMutationProposal(
        operation="reschedule",
        target=target,
        new_start=local_start.astimezone(ctx.now.tzinfo or local_start.tzinfo),
        new_end=local_end.astimezone(ctx.now.tzinfo or local_end.tzinfo),
        assumptions=assumptions,
    )


def _parse_new_time(time_window: str) -> tuple[Any, int | None]:
    """(start_time, explicit_duration_minutes | None) from a time_window token.

    A RANGE ("17:00-18:00", the normalised form of "5-6 PM") pins both ends, so
    it defines the duration too. A single time pins only the start.
    """
    from navis.reasoning.scheduling.availability import _WINDOW_PRESETS

    w = time_window.strip().lower()
    if w in _WINDOW_PRESETS:
        # "evening" etc. — take the preset's start; the length stays whatever
        # the meeting already was.
        return _WINDOW_PRESETS[w][0], None
    if "-" in w and w.count("-") == 1:
        left, _, right = w.partition("-")
        try:
            start = _parse_time_token(left)
            end = _parse_time_token(right)
        except ValueError as exc:
            raise ScopeError(f"unrecognised time range: {time_window!r}") from exc
        minutes = (end.hour * 60 + end.minute) - (start.hour * 60 + start.minute)
        if minutes <= 0:
            raise ScopeError(f"time range must go forward: {time_window!r}")
        return start, minutes
    try:
        return _parse_time_token(w), None
    except ValueError as exc:
        raise ScopeError(f"unrecognised time: {time_window!r}") from exc


def _resolve_op_date(constraint: str, *, now: datetime, tz: ZoneInfo):
    """Delegate to the scheduling date resolver so both verbs agree on what
    "friday" / "tomorrow" / "2026-08-09" mean."""
    from navis.reasoning.scheduling.availability import _resolve_date

    return _resolve_date(constraint, now=now, tz=tz)
