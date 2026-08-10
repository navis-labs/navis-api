"""CalendarQueryAgent — deterministic answers over the user's real calendar.

Bypasses the RAG memory path for calendar queries so the user gets
actual event data instead of a vague "not enough memory" fallback or
an LLM composing over sparse fact snippets.

Never creates events — that's the SchedulingAgent's job. This agent
only READS via `CalendarClient.list_events_expanded` in a scoped
window (today, tomorrow, this week, next meeting).
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.ext.asyncio import AsyncSession

from navis.platform.monitoring import get_logger
from navis.reasoning.calendar_query.intent import (
    TimeScope,
    parse_time_scope,
    resolve_window,
)
from navis.reasoning.scheduling.availability import (
    CalendarClientProtocol,
    open_calendar_client,
)
from navis.signals.google.resolve import CalendarNotConnected

logger = get_logger("reasoning.calendar_query.agent")


class CalendarQueryContext(BaseModel):
    """Input to `CalendarQueryAgent.answer`."""

    model_config = ConfigDict(extra="ignore")

    org_id: uuid.UUID
    utterance: str = Field(..., min_length=1, max_length=2000)
    time_zone: str = Field("UTC", max_length=64)
    now: datetime


@dataclass(frozen=True, slots=True)
class CalendarEvent:
    """One event in a resolved list — normalized for rendering."""

    id: str
    title: str
    start_utc: datetime
    end_utc: datetime
    attendees: tuple[str, ...]
    location: str | None
    hangout_link: str | None
    time_zone: str  # source tz from the event; may differ from user's

    def render_time(self, viewer_tz: str) -> str:
        """Format the event's local time in the VIEWER's timezone."""
        try:
            tz = ZoneInfo(viewer_tz)
        except Exception:  # noqa: BLE001
            tz = ZoneInfo("UTC")
        s = self.start_utc.astimezone(tz)
        e = self.end_utc.astimezone(tz)
        return f"{_fmt_time(s)} – {_fmt_time(e)}"

    def render_date(self, viewer_tz: str) -> str:
        try:
            tz = ZoneInfo(viewer_tz)
        except Exception:  # noqa: BLE001
            tz = ZoneInfo("UTC")
        s = self.start_utc.astimezone(tz)
        return s.strftime("%a, %b %d")


class CalendarQueryResult(BaseModel):
    """Output — prose text plus structured events + resolved scope."""

    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    text: str
    scope: TimeScope
    events: list[dict[str, Any]] = Field(default_factory=list)
    time_zone: str = "UTC"


class CalendarQueryAgent:
    """Read-only calendar answers: today, tomorrow, this week, next.

    Uses the SAME calendar-client factory the scheduling agent uses
    (private DB session, decrypted tokens, safe TokenManager). Errors
    from Google (403 scope, 401 auth, network) are surfaced as
    friendly prose — the frontend renders the text as a normal chat
    answer.
    """

    def __init__(
        self,
        db: AsyncSession | None,
        *,
        calendar_client_factory: Callable[
            [AsyncSession, uuid.UUID],
            "tuple[CalendarClientProtocol, object]",
        ]
        | None = None,
    ) -> None:
        self.db = db
        self._factory = calendar_client_factory or open_calendar_client

    async def answer(self, ctx: CalendarQueryContext) -> CalendarQueryResult:
        scope = parse_time_scope(ctx.utterance)
        window = resolve_window(
            scope, now=ctx.now, time_zone=ctx.time_zone
        )

        try:
            client, transport = await self._factory(self.db, ctx.org_id)
        except CalendarNotConnected as exc:
            logger.info("calendar_query: no calendar integration org=%s: %s", ctx.org_id, exc)
            return CalendarQueryResult(
                text=(
                    "Google Calendar isn't connected yet. Connect it "
                    "from the Sources page to see your meetings here."
                ),
                scope=scope,
                events=[],
                time_zone=ctx.time_zone,
            )
        except RuntimeError as exc:
            # Connected, but the grant is missing the read scope (or the row has
            # no usable tokens). Distinct copy — telling someone to "connect"
            # something they already connected is the fastest way to lose them.
            logger.info("calendar_query: calendar unusable org=%s: %s", ctx.org_id, exc)
            return CalendarQueryResult(
                text=(
                    "Google Calendar is connected, but not with permission to "
                    "read your schedule. Disconnect and reconnect Google "
                    "Workspace to grant it."
                ),
                scope=scope,
                events=[],
                time_zone=ctx.time_zone,
            )

        events: list[CalendarEvent] = []
        try:
            raw = await _call_list_events(
                client,
                time_min=window.time_min_utc.isoformat(),
                time_max=window.time_max_utc.isoformat(),
                max_results=window.max_events,
            )
            events = _parse_events(raw)
        except Exception as exc:  # noqa: BLE001 — Google is external
            logger.exception("calendar_query: list_events failed org=%s", ctx.org_id)
            err_str = str(exc)
            is_scope_error = (
                "insufficientPermissions" in err_str
                or "ACCESS_TOKEN_SCOPE_INSUFFICIENT" in err_str
                or "insufficient authentication scopes" in err_str.lower()
            )
            msg = (
                "Google Calendar rejected the availability read — "
                "disconnect and reconnect Google Workspace so the new "
                "read scope is granted."
                if is_scope_error
                else "I couldn't reach Google Calendar just now. Try again in a moment."
            )
            return CalendarQueryResult(
                text=msg, scope=scope, events=[], time_zone=ctx.time_zone
            )
        finally:
            if hasattr(transport, "aclose"):
                try:
                    await transport.aclose()
                except Exception:
                    logger.exception("calendar_query: transport close failed")

        text = _format_answer(events, window, ctx.time_zone)
        return CalendarQueryResult(
            text=text,
            scope=scope,
            events=[_event_to_dict(e) for e in events],
            time_zone=ctx.time_zone,
        )


# ── Google Calendar → CalendarEvent normalization ──────────────────────


def _parse_events(raw: dict[str, Any]) -> list[CalendarEvent]:
    """Normalize Google's events.list response into our shape."""
    items = raw.get("items") or []
    out: list[CalendarEvent] = []
    for item in items:
        if item.get("status") == "cancelled":
            continue
        start = _parse_iso_from_event_side(item.get("start"))
        end = _parse_iso_from_event_side(item.get("end"))
        if start is None or end is None:
            continue
        tz = (item.get("start") or {}).get("timeZone") or "UTC"
        attendees = tuple(
            (a.get("email") or "").strip().lower()
            for a in (item.get("attendees") or [])
            if isinstance(a, dict) and a.get("email")
        )
        out.append(
            CalendarEvent(
                id=str(item.get("id") or ""),
                title=str(item.get("summary") or "(untitled)"),
                start_utc=start,
                end_utc=end,
                attendees=attendees,
                location=item.get("location"),
                hangout_link=item.get("hangoutLink"),
                time_zone=tz,
            )
        )
    return out


def _parse_iso_from_event_side(side: dict[str, Any] | None) -> datetime | None:
    if not side:
        return None
    raw = side.get("dateTime") or side.get("date")
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        # All-day event ("date"-only) — treat as start-of-day in the
        # event's declared timeZone or UTC as fallback.
        tz_name = (side.get("timeZone") or "UTC")
        try:
            tz = ZoneInfo(tz_name)
        except Exception:  # noqa: BLE001
            tz = ZoneInfo("UTC")
        dt = dt.replace(tzinfo=tz)
    return dt.astimezone(timezone.utc)


# ── Text rendering ─────────────────────────────────────────────────────


def _format_answer(
    events: list[CalendarEvent],
    window,
    viewer_tz: str,
) -> str:
    if not events:
        return _empty_answer(window)
    if window.scope is TimeScope.NEXT_MEETING and events:
        e = events[0]
        return _format_next_meeting(e, viewer_tz)
    return _format_list(events, window, viewer_tz)


def _empty_answer(window) -> str:
    if window.scope is TimeScope.TODAY or window.scope is TimeScope.ANY:
        return "You have no meetings on your calendar today."
    if window.scope is TimeScope.TOMORROW:
        return "You have no meetings on your calendar tomorrow."
    if window.scope is TimeScope.THIS_WEEK:
        return "Your calendar is clear this week."
    if window.scope is TimeScope.NEXT_WEEK:
        return "You have no meetings on your calendar next week."
    if window.scope is TimeScope.NEXT_MEETING:
        return "You have no upcoming meetings in the next 7 days."
    return f"You have no meetings scheduled {window.label}."


def _format_next_meeting(e: CalendarEvent, viewer_tz: str) -> str:
    when = f"{e.render_date(viewer_tz)} · {e.render_time(viewer_tz)}"
    lines = [f"Your next meeting: **{e.title}** — {when}."]
    if e.attendees:
        att = ", ".join(sorted(set(e.attendees)))
        lines.append(f"Attendees: {att}")
    if e.location:
        lines.append(f"Location: {e.location}")
    if e.hangout_link:
        lines.append(f"Meet link: {e.hangout_link}")
    return "\n".join(lines)


def _format_list(
    events: list[CalendarEvent], window, viewer_tz: str
) -> str:
    header = (
        f"You have {len(events)} "
        f"meeting{'s' if len(events) != 1 else ''} {window.label}:"
    )
    lines = [header, ""]
    # Group by day when the window spans multiple days.
    grouped: dict[str, list[CalendarEvent]] = {}
    for e in events:
        day = e.render_date(viewer_tz)
        grouped.setdefault(day, []).append(e)
    for day, day_events in grouped.items():
        if len(grouped) > 1:
            lines.append(f"**{day}**")
        for e in day_events:
            att = ""
            if e.attendees:
                short = list(sorted(set(e.attendees)))[:3]
                att = f" · with {', '.join(short)}"
                if len(set(e.attendees)) > 3:
                    att += f" +{len(set(e.attendees)) - 3} more"
            lines.append(f"· {e.render_time(viewer_tz)} — {e.title}{att}")
        if len(grouped) > 1:
            lines.append("")
    return "\n".join(lines).rstrip()


def _fmt_time(dt: datetime) -> str:
    """OS-portable HH:MM AM/PM without % directives."""
    hour = dt.hour
    minute = dt.minute
    meridiem = "AM" if hour < 12 else "PM"
    hh = hour % 12 or 12
    if minute == 0:
        return f"{hh}:00 {meridiem}"
    return f"{hh}:{minute:02d} {meridiem}"


def _event_to_dict(e: CalendarEvent) -> dict[str, Any]:
    return {
        "id": e.id,
        "title": e.title,
        "start": e.start_utc.isoformat(),
        "end": e.end_utc.isoformat(),
        "attendees": list(e.attendees),
        "location": e.location,
        "hangout_link": e.hangout_link,
        "time_zone": e.time_zone,
    }


async def _call_list_events(
    client: CalendarClientProtocol,
    *,
    time_min: str,
    time_max: str,
    max_results: int,
) -> dict[str, Any]:
    """Shim for the client's list_events_expanded method.

    Uses getattr so tests can pass a lightweight fake without
    implementing the full CalendarClientProtocol.
    """
    fn = getattr(client, "list_events_expanded", None)
    if fn is None:
        raise RuntimeError("calendar client missing list_events_expanded")
    return await fn(time_min=time_min, time_max=time_max, max_results=max_results)
