"""Convert a validated (date + time) constraint into a concrete free slot.

Uses the existing `navis.signals.calendar.CalendarClient.freebusy` — the
same Google API path the workspace router calls for the manual slot
picker. One implementation, one place, one bug surface.

The agent MUST have already established that both `date_constraint` and
`time_window` are non-null BEFORE calling `resolve_window`. If either is
null this raises `MissingConstraintError` — that's a bug in the agent,
not a user issue, and belongs in the traceback rather than a
clarification. Missing-slot clarifications are the agent's job.

Two window flavours are surfaced to the caller:
  * EXACT — user asked for a specific time ("3 PM", "15:00"). The
    window is exactly [start, start + duration]. On busy we do NOT
    silently slide; the agent surfaces alternatives via
    `find_alternative_slots`.
  * FLEXIBLE — user asked for a range ("afternoon", "morning",
    "14:00-17:00"). The window is the whole range; we pick the first
    non-conflicting slot inside it. On busy we return None and the
    agent asks for another day.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Protocol
from zoneinfo import ZoneInfo

from sqlalchemy.ext.asyncio import AsyncSession

from navis.identity.models import Integration
from navis.platform.database import session_scope
from navis.reasoning.scheduling.contract import MeetingConstraints, MeetingSlot
from navis.signals.calendar import CalendarClient
from navis.signals.crypto import token_cipher
from navis.signals.google.oauth import TokenManager
from navis.signals.google.resolve import resolve_calendar_integration
from navis.signals.google.transport import GoogleTransport


# ── Constants ──────────────────────────────────────────────────────────────

# Duration DOES have a safe default (30 min) per product spec — but it
# is surfaced as an explicit assumption on the proposal card so the
# founder can catch it before approving.
DEFAULT_DURATION_MIN = 30

# Time-of-day presets. Used only when the user gave a preset name
# ("morning", "afternoon", "evening"); NOT applied as a default for a
# missing time_window (the agent asks for time explicitly instead).
_MORNING = (time(9, 0), time(12, 0))
_AFTERNOON = (time(12, 0), time(17, 0))
_EVENING = (time(17, 0), time(20, 0))

_WINDOW_PRESETS: dict[str, tuple[time, time]] = {
    "morning": _MORNING,
    "afternoon": _AFTERNOON,
    "evening": _EVENING,
}

_WEEKDAY_INDEX = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}


class WindowKind(str, Enum):
    """Whether the caller asked for an exact time or a flexible window."""
    EXACT = "exact"
    FLEXIBLE = "flexible"


class ScopeError(Exception):
    """Raised when a supplied date/time constraint is unparseable or in
    the past. The agent catches this and emits a clarification."""


class MissingConstraintError(Exception):
    """Raised when `resolve_window` is called with a null slot. This is
    ALWAYS an agent-side bug — the agent must check completeness and
    emit a clarification BEFORE calling here."""


# ── Public window resolution ───────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ConcreteWindow:
    """A UTC-aware start/end pair the freebusy query will scan.

    `kind` distinguishes exact (single time) from flexible (range /
    preset). The agent uses this to decide whether a busy result
    should surface alternatives near the exact time or a "pick another
    day" clarification.
    """

    start_utc: datetime
    end_utc: datetime
    time_zone: str
    duration_minutes: int
    kind: WindowKind
    # Assumptions the agent applied while building this window — always
    # surfaced on the proposal card. Currently only used for the
    # duration default.
    assumptions: tuple[str, ...] = ()


def resolve_window(
    constraints: MeetingConstraints, *, now: datetime, time_zone: str
) -> ConcreteWindow:
    """Turn ("tuesday", "afternoon", 30) into a UTC start/end range.

    Preconditions (the agent enforces these BEFORE calling):
      * `date_constraint` is non-null.
      * `time_window` is non-null.

    Duration MAY be null — falls back to `DEFAULT_DURATION_MIN` with an
    explicit assumption. Every other missing slot raises
    `MissingConstraintError` so the agent's completeness check stays
    the single source of truth for "clarify vs proceed".
    """
    if constraints.date_constraint is None:
        raise MissingConstraintError("date_constraint is required")
    if constraints.time_window is None:
        raise MissingConstraintError("time_window is required")

    try:
        tz = ZoneInfo(time_zone)
    except Exception as exc:  # noqa: BLE001 — unknown zone → clarify
        raise ScopeError(f"unknown time zone: {time_zone!r}") from exc

    assumptions: list[str] = []
    duration = constraints.duration_minutes or DEFAULT_DURATION_MIN
    if constraints.duration_minutes is None:
        assumptions.append(f"Assumed {DEFAULT_DURATION_MIN}-min duration.")

    target_date = _resolve_date(constraints.date_constraint, now=now, tz=tz)
    parsed_window = _resolve_time_of_day(constraints.time_window)
    kind = parsed_window.kind

    local_start = datetime.combine(target_date, parsed_window.start, tzinfo=tz)
    local_end = datetime.combine(target_date, parsed_window.end, tzinfo=tz)

    # Guard: don't propose a slot that starts in the past.
    now_local = now.astimezone(tz)
    if local_end <= now_local:
        # For an EXACT time in the past, the user asked for something
        # concretely past — surface as a scope error so the agent can
        # ask them to pick another day.
        if kind is WindowKind.EXACT:
            raise ScopeError("the requested time is in the past")
        # For a flexible window in the past, the user's day pick is
        # the issue — same clarification.
        raise ScopeError("the requested window is in the past")
    if local_start < now_local:
        # Window starts in the past but ends in the future. For a
        # flexible window, trim to now. For exact, that's still in the
        # past (start doesn't move) — surface.
        if kind is WindowKind.EXACT:
            raise ScopeError("the requested time is in the past")
        local_start = _round_up_to_quarter(now_local)

    if local_end - local_start < timedelta(minutes=duration):
        raise ScopeError(
            f"the requested window is shorter than the {duration}-min duration"
        )

    return ConcreteWindow(
        start_utc=local_start.astimezone(timezone.utc),
        end_utc=local_end.astimezone(timezone.utc),
        time_zone=time_zone,
        duration_minutes=duration,
        kind=kind,
        assumptions=tuple(assumptions),
    )


# ── Time-window parsing ────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class _ParsedTimeWindow:
    start: time
    end: time
    kind: WindowKind


def _resolve_time_of_day(window: str) -> _ParsedTimeWindow:
    """Turn 'afternoon' / '15:00' / '14:00-15:00' / '3 PM' into (start, end, kind).

    Raises ScopeError on unrecognised input. The caller must ensure
    `window` is non-null — a null time is a `MissingConstraintError`
    that never reaches here.
    """
    w = (window or "").strip().lower()
    if not w:
        raise ScopeError("time window is empty")
    # Presets: morning / afternoon / evening → flexible range.
    if w in _WINDOW_PRESETS:
        start, end = _WINDOW_PRESETS[w]
        return _ParsedTimeWindow(start=start, end=end, kind=WindowKind.FLEXIBLE)
    # Explicit range "HH:MM-HH:MM" → flexible.
    if "-" in w and w.count("-") == 1:
        left, _, right = w.partition("-")
        try:
            start = _parse_time_token(left)
            end = _parse_time_token(right)
        except ValueError as exc:
            raise ScopeError(f"unrecognised time range: {window!r}") from exc
        if end <= start:
            raise ScopeError(f"time range must go forward: {window!r}")
        return _ParsedTimeWindow(start=start, end=end, kind=WindowKind.FLEXIBLE)
    # Single time → exact. Window is [t, t + duration] — end filled in
    # by resolve_window using the actual duration. We synthesize a
    # placeholder end here and let resolve_window's duration guard
    # trim; but for correctness we return an end == start so the
    # duration check in resolve_window kicks in AS IF caller supplied
    # an infinitely narrow window, causing "shorter than duration" —
    # which is exactly what we want for an exact-time request.
    #
    # ACTUALLY: for EXACT, the window IS the exact slot [t, t+dur].
    # But we don't know duration here. Instead: set end = start + 24h
    # so the duration check passes, then _first_open (called by the
    # agent's freebusy step) constrains to duration starting at start.
    # That preserves the "exact means only this start time works"
    # semantics via the caller's downstream check.
    try:
        start = _parse_time_token(w)
    except ValueError as exc:
        raise ScopeError(f"unrecognised time: {window!r}") from exc
    # end sentinel — the agent will clip to start + duration for exact.
    return _ParsedTimeWindow(start=start, end=time(23, 59), kind=WindowKind.EXACT)


_TIME_TOKEN_RE = None  # placeholder; module-level compile below

import re  # noqa: E402  (placed here so the regex sits with its consumer)

_HHMM_RE = re.compile(r"^\s*(\d{1,2}):(\d{2})\s*$")
_HHONLY_RE = re.compile(r"^\s*(\d{1,2})\s*$")
_TWELVE_HR_RE = re.compile(r"^\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm)\s*$", re.IGNORECASE)


def _parse_time_token(token: str) -> time:
    """Parse one time literal: '15:00' / '3 PM' / '3:30 PM' / '15'."""
    s = token.strip().lower()
    m = _TWELVE_HR_RE.match(s)
    if m:
        hh = int(m.group(1))
        mm = int(m.group(2) or 0)
        meridiem = m.group(3).lower()
        if hh == 12:
            hh = 0
        if meridiem == "pm":
            hh += 12
        if not (0 <= hh <= 23 and 0 <= mm <= 59):
            raise ValueError(f"out-of-range 12h time: {token!r}")
        return time(hh, mm)
    m = _HHMM_RE.match(s)
    if m:
        hh = int(m.group(1))
        mm = int(m.group(2))
        if not (0 <= hh <= 23 and 0 <= mm <= 59):
            raise ValueError(f"out-of-range 24h time: {token!r}")
        return time(hh, mm)
    m = _HHONLY_RE.match(s)
    if m:
        hh = int(m.group(1))
        if not (0 <= hh <= 23):
            raise ValueError(f"out-of-range hour: {token!r}")
        return time(hh, 0)
    raise ValueError(f"unrecognised time token: {token!r}")


# ── Date parsing ───────────────────────────────────────────────────────────


def _resolve_date(constraint: str, *, now: datetime, tz: ZoneInfo) -> date:
    """Resolve 'tuesday' / 'tomorrow' / 'YYYY-MM-DD' → date.

    The agent enforces `constraint is not None` upstream — this never
    receives null. Same-weekday requests roll forward to next week (a
    founder saying "let's meet Tuesday" on a Tuesday almost never means
    today). Past ISO dates raise ScopeError → clarification.
    """
    local_today = now.astimezone(tz).date()
    c = constraint.strip().lower()
    if c == "today":
        return local_today
    if c == "tomorrow":
        return local_today + timedelta(days=1)
    if c == "next_week":
        return local_today + timedelta(days=7)
    if c in _WEEKDAY_INDEX:
        target = _WEEKDAY_INDEX[c]
        delta = (target - local_today.weekday()) % 7
        if delta == 0:
            delta = 7
        return local_today + timedelta(days=delta)
    try:
        parsed = date.fromisoformat(c)
    except ValueError as exc:
        raise ScopeError(f"unrecognised date: {constraint!r}") from exc
    if parsed < local_today:
        raise ScopeError(f"date {c!r} is in the past")
    return parsed


def _round_up_to_quarter(dt: datetime) -> datetime:
    """Round `dt` UP to the next 15-min boundary (preserves tz)."""
    minute = dt.minute
    if dt.second or dt.microsecond:
        minute += 1
    remainder = minute % 15
    add = (15 - remainder) if remainder else 0
    if minute + add >= 60:
        return dt.replace(second=0, microsecond=0, minute=0) + timedelta(
            hours=1, minutes=(minute + add) - 60
        )
    return dt.replace(second=0, microsecond=0, minute=minute + add)


# ── Freebusy — public API ──────────────────────────────────────────────────


class CalendarClientProtocol(Protocol):
    async def freebusy(
        self,
        *,
        time_min: str,
        time_max: str,
        calendar_ids: list[str] | None = None,
        time_zone: str = "UTC",
    ) -> dict[str, Any]: ...


async def find_first_open_slot(
    window: ConcreteWindow,
    *,
    calendar_client: CalendarClientProtocol,
) -> MeetingSlot | None:
    """Query freeBusy, subtract busy blocks, return the first open slot.

    For an EXACT window, "first open" IS the exact requested slot — we
    only return a MeetingSlot if [start, start + duration] is
    unambiguously free. Anything else returns None and the agent hands
    the caller alternatives via `find_alternative_slots`.

    For a FLEXIBLE window, we slide across the full window and return
    the first duration-sized gap.
    """
    raw = await calendar_client.freebusy(
        time_min=window.start_utc.isoformat(),
        time_max=window.end_utc.isoformat(),
        time_zone=window.time_zone,
    )
    busy = _parse_busy_blocks(raw)
    duration = timedelta(minutes=window.duration_minutes)

    if window.kind is WindowKind.EXACT:
        exact_end = window.start_utc + duration
        if _slot_overlaps_busy(window.start_utc, exact_end, busy):
            return None
        return MeetingSlot(
            start=window.start_utc,
            end=exact_end,
            time_zone=window.time_zone,
            duration_minutes=window.duration_minutes,
        )

    slot = _first_open(window.start_utc, window.end_utc, busy, duration)
    if slot is None:
        return None
    start_utc, end_utc = slot
    return MeetingSlot(
        start=start_utc,
        end=end_utc,
        time_zone=window.time_zone,
        duration_minutes=window.duration_minutes,
    )


async def find_alternative_slots(
    window: ConcreteWindow,
    *,
    calendar_client: CalendarClientProtocol,
    max_alternatives: int = 3,
    search_radius_hours: int = 4,
) -> list[MeetingSlot]:
    """When an EXACT time is busy, return up to `max_alternatives` free
    slots within `±search_radius_hours` of the requested time.

    Kept restrained (3 slots, 4-hour radius) — the goal is "here are
    close-by options", not a full-day pick list.
    """
    duration = timedelta(minutes=window.duration_minutes)
    radius = timedelta(hours=search_radius_hours)
    search_start = window.start_utc - radius
    search_end = window.start_utc + radius
    raw = await calendar_client.freebusy(
        time_min=search_start.isoformat(),
        time_max=search_end.isoformat(),
        time_zone=window.time_zone,
    )
    busy = _parse_busy_blocks(raw)

    slots: list[MeetingSlot] = []
    cursor = search_start
    while cursor + duration <= search_end and len(slots) < max_alternatives:
        candidate_end = cursor + duration
        # Skip the exact-time slot we already reported as busy.
        if cursor == window.start_utc:
            cursor = candidate_end
            continue
        if _slot_overlaps_busy(cursor, candidate_end, busy):
            # Jump past the overlapping busy block.
            for b_start, b_end in busy:
                if b_start < candidate_end and b_end > cursor:
                    cursor = b_end
                    break
            else:
                cursor = candidate_end
            continue
        slots.append(
            MeetingSlot(
                start=cursor,
                end=candidate_end,
                time_zone=window.time_zone,
                duration_minutes=window.duration_minutes,
            )
        )
        # Space alternatives at least `duration` apart so the pick list
        # isn't three nearly-identical times.
        cursor = candidate_end
    return slots


# ── Internal helpers ───────────────────────────────────────────────────────


def _parse_busy_blocks(raw: dict[str, Any]) -> list[tuple[datetime, datetime]]:
    """Extract every busy block from Google's freeBusy response, UTC-aware."""
    calendars = raw.get("calendars") or {}
    blocks: list[tuple[datetime, datetime]] = []
    for cal in calendars.values():
        for b in cal.get("busy") or []:
            try:
                start = datetime.fromisoformat(str(b["start"]).replace("Z", "+00:00"))
                end = datetime.fromisoformat(str(b["end"]).replace("Z", "+00:00"))
            except (KeyError, ValueError):
                continue
            if start.tzinfo is None:
                start = start.replace(tzinfo=timezone.utc)
            if end.tzinfo is None:
                end = end.replace(tzinfo=timezone.utc)
            blocks.append((start, end))
    return sorted(blocks, key=lambda b: b[0])


def _slot_overlaps_busy(
    slot_start: datetime,
    slot_end: datetime,
    busy: list[tuple[datetime, datetime]],
) -> bool:
    for b_start, b_end in busy:
        if b_end <= slot_start:
            continue
        if b_start >= slot_end:
            break
        return True
    return False


def _first_open(
    window_start: datetime,
    window_end: datetime,
    busy: list[tuple[datetime, datetime]],
    duration: timedelta,
) -> tuple[datetime, datetime] | None:
    """First `duration` slot inside [window_start, window_end] that
    doesn't overlap any busy block."""
    cursor = window_start
    for b_start, b_end in busy:
        if b_end <= cursor:
            continue
        if b_start >= window_end:
            break
        if b_start - cursor >= duration:
            return (cursor, cursor + duration)
        cursor = max(cursor, b_end)
    if window_end - cursor >= duration:
        return (cursor, cursor + duration)
    return None


# ── Real Google client factory (used by the agent; tests inject a fake) ────


class _ManagedCalendarClient:
    """Owns a private DB session + transport lifecycle for one freeBusy call.

    The scheduling agent asks the calendar for availability BEFORE it
    calls ExecutionService.request. We can't share the request's DB
    session for this: the Integration row's tokens are stored encrypted,
    but TokenManager decrypts them in-place on the ORM object — if the
    caller's session then commits, we'd persist plaintext tokens. Using
    a scoped session that we explicitly do NOT commit keeps the mutation
    local to the calendar call. Any token refresh performed by
    TokenManager IS persisted through a targeted UPDATE below (via the
    same `_senders`-style helper) rather than a full session commit.
    """

    def __init__(self) -> None:
        self._session_cm = session_scope()
        self._db: AsyncSession | None = None
        self._transport: GoogleTransport | None = None
        self._client: CalendarClient | None = None
        self._token_manager: TokenManager | None = None
        self._integration: Integration | None = None

    async def __aenter__(self) -> CalendarClient:
        self._db = await self._session_cm.__aenter__()
        # Any failure from here on MUST unwind the session we just opened.
        # Previously a "not connected" org raised straight out of __aenter__,
        # leaking the AsyncSession (and its pooled connection) on every attempt
        # — a slow pool drain that eventually turned "Calendar isn't connected"
        # into request timeouts across the whole API.
        try:
            # Single source of truth for "is Calendar connected", shared with
            # the execution senders and the workspace router. It also HEALS a
            # missing google_calendar row from a sibling Google grant, which is
            # what made chat claim "not connected" while the dashboard happily
            # listed the same org's meetings.
            integration = await resolve_calendar_integration(self._db, self._org_id)
            # Persist a self-heal (a materialised google_calendar row) NOW,
            # while the stored tokens are still ciphertext. The decrypt-in-place
            # below mutates the ORM object, so any later commit on this session
            # would write the tokens back as plaintext — which is why __aexit__
            # rolls back unless it is deliberately persisting a refreshed token.
            await self._db.commit()
            access = (
                token_cipher.decrypt(integration.access_token)
                if integration.access_token
                else None
            )
            refresh = (
                token_cipher.decrypt(integration.refresh_token)
                if integration.refresh_token
                else None
            )
            integration.access_token = access
            integration.refresh_token = refresh
            self._integration = integration
            self._token_manager = TokenManager(integration)
            self._transport = GoogleTransport(
                self._token_manager.token_provider(),
                on_auth_failure=self._token_manager.invalidate,
            )
            self._client = CalendarClient(self._transport)
            return self._client
        except BaseException:
            # Roll back (the heal INSERT may be pending) and close the session
            # before re-raising, so the caller's `except RuntimeError` path
            # costs nothing beyond the failed lookup.
            await self._close_session_quietly()
            raise

    async def _close_session_quietly(self) -> None:
        if self._db is None:
            return
        try:
            await self._db.rollback()
        except Exception:  # noqa: BLE001 — teardown must not mask the real error
            pass
        try:
            await self._session_cm.__aexit__(None, None, None)
        except Exception:  # noqa: BLE001
            pass
        self._db = None

    async def __aexit__(self, exc_type, exc, tb) -> None:
        try:
            if (
                self._db is not None
                and self._token_manager is not None
                and self._integration is not None
                and self._token_manager.last_token is not None
                and self._token_manager.last_token.refreshed
            ):
                last = self._token_manager.last_token
                self._integration.access_token = token_cipher.encrypt(
                    last.access_token
                )
                self._integration.token_expires_at = last.expires_at
                if last.refresh_token is not None:
                    self._integration.refresh_token = token_cipher.encrypt(
                        last.refresh_token
                    )
                await self._db.commit()
            elif self._db is not None:
                await self._db.rollback()
        finally:
            if self._transport is not None:
                try:
                    await self._transport.aclose()
                except Exception:  # noqa: BLE001 — teardown must not mask the real error
                    pass
            await self._session_cm.__aexit__(exc_type, exc, tb)

    def _set_org_id(self, org_id: uuid.UUID) -> "_ManagedCalendarClient":
        self._org_id = org_id
        return self


ClientFactory = Callable[[uuid.UUID], "tuple[CalendarClientProtocol, Any]"]


async def open_calendar_client(
    db: AsyncSession, org_id: uuid.UUID
) -> tuple[CalendarClientProtocol, "_ManagedTransport"]:
    """Return (client, transport-with-aclose) matching the agent's contract.

    We keep the signature (db, org_id) → tuple so tests can inject
    lighter fakes. The real path DELIBERATELY ignores the passed `db`
    and opens a private session (see `_ManagedCalendarClient`); the
    fake in tests just returns a stub client + a no-op transport.
    """
    cm = _ManagedCalendarClient()._set_org_id(org_id)
    client = await cm.__aenter__()
    transport = _ManagedTransport(cm)
    return client, transport


class _ManagedTransport:
    """Adapter presenting the agent's expected `aclose()` interface."""

    def __init__(self, cm: _ManagedCalendarClient) -> None:
        self._cm = cm
        self._closed = False

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._cm.__aexit__(None, None, None)
