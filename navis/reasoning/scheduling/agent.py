"""SchedulingAgent — orchestrates resolve / merge / clarify / propose.

Single entry point: `SchedulingAgent.propose(ctx)`. Returns exactly
one of `Clarification` or `ScheduledProposal` — NEVER a side effect,
NEVER a Google API call outside of freeBusy for availability.

Multi-turn flow — a `pending` state on the context is the frontend
echoing back what was resolved on the last turn. The agent merges the
current utterance INTO the pending state so a bare follow-up like
"tomorrow at 3 PM" continues the pending scheduling request without
requiring the user to repeat the attendee.

Behaviour rules (product spec):

  * Missing date OR time → clarification. NEVER silently assumes today
    or business hours.
  * Duration MAY default to 30 min — but only with a visible
    assumption on the proposal card.
  * Exact time busy → alternatives near the requested time; the user
    picks. NEVER silently slides.
  * Flexible window fully busy → clarification, not a proposal.
  * Missing attendee / ambiguous name → clarification.
  * Only when everything is resolved and a free slot is found does the
    chat router hand the proposal to ExecutionService.request.
"""
from __future__ import annotations

import re
import uuid
from datetime import datetime
from typing import Callable

from sqlalchemy.ext.asyncio import AsyncSession
from zoneinfo import ZoneInfo

from navis.platform.monitoring import get_logger
from navis.reasoning.scheduling.availability import (
    CalendarClientProtocol,
    MissingConstraintError,
    ScopeError,
    WindowKind,
    find_alternative_slots,
    find_first_open_slot,
    open_calendar_client,
    resolve_window,
)
from navis.reasoning.scheduling.context import (
    PersonSearchFn,
    SignalGetter,
    resolve_attendee,
)
from navis.reasoning.scheduling.contract import (
    Clarification,
    MeetingConstraints,
    MeetingSlot,
    PendingScheduling,
    ResolvedAttendee,
    ScheduledProposal,
    SchedulingContext,
    SchedulingResult,
)
from navis.reasoning.scheduling.extract import extract_constraints
from navis.signals.google.resolve import CalendarNotConnected

logger = get_logger("reasoning.scheduling.agent")

# Detects an email anywhere in the utterance — used to decide whether
# the CURRENT turn is re-designating a new attendee (in which case any
# pending attendee is discarded).
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")


class SchedulingAgent:
    """Resolve → merge with pending → check completeness → freebusy → propose (or clarify).

    Constructor accepts injection points so unit tests can substitute
    fake LLMs, calendars, signal-getters, and person-search functions
    without touching Google or a real DB.
    """

    def __init__(
        self,
        db: AsyncSession | None,
        *,
        ml_provider=None,
        calendar_client_factory: Callable[
            [AsyncSession, uuid.UUID],
            "tuple[CalendarClientProtocol, object]",
        ]
        | None = None,
        signal_getter: SignalGetter | None = None,
        person_search_fn: PersonSearchFn | None = None,
    ) -> None:
        self.db = db
        self._ml_provider = ml_provider
        self._calendar_client_factory = (
            calendar_client_factory or open_calendar_client
        )
        self._signal_getter = signal_getter
        self._person_search_fn = person_search_fn

    async def propose(self, ctx: SchedulingContext) -> SchedulingResult:
        """Run all stages. Returns Clarification | ScheduledProposal."""
        # ── Stage 1: extract from current utterance ──────────────────
        # Anchor date parsing to the founder's LOCAL now, not UTC. At 01:00 IST
        # the UTC date is still yesterday; resolving "9 August" (or "today")
        # against that would be off by a day for every user east of Greenwich.
        extracted = await extract_constraints(
            ctx.utterance,
            provider=self._ml_provider,
            now=_local_now(ctx.now, ctx.time_zone),
        )

        # ── Stage 2: merge with pending (previous turn) ──────────────
        merged = _merge_constraints(extracted, ctx.pending)

        # ── Stage 3: resolve WHO ────────────────────────────────────
        # If the current turn introduces a new attendee (email, name,
        # or pronoun), re-resolve. Otherwise reuse any resolved
        # attendee from pending.
        current_reintroduces_attendee = bool(
            extracted.attendee_hint
            or extracted.pronoun
            or _EMAIL_RE.search(ctx.utterance)
        )
        if (
            not current_reintroduces_attendee
            and ctx.pending is not None
            and ctx.pending.attendee_email
            and ctx.pending.attendee_source
        ):
            attendee = ResolvedAttendee(
                email=ctx.pending.attendee_email,
                name=ctx.pending.attendee_name,
                canonical_id=ctx.pending.attendee_canonical_id,
                source=ctx.pending.attendee_source,
            )
            source_signal_event_id = ctx.pending.source_signal_event_id
            thread_id = ctx.pending.thread_id or ctx.thread_id
        else:
            attendee_result = await resolve_attendee(
                self.db,
                ctx,
                merged,
                signal_getter=self._signal_getter,
                person_search_fn=self._person_search_fn,
            )
            if attendee_result.is_ambiguous:
                return _person_disambiguation(
                    attendee_result.candidates, merged, ctx, pending_no_attendee=True
                )
            if attendee_result.is_missing:
                return _attendee_clarification(
                    merged, ctx, extracted=extracted
                )
            attendee = attendee_result.attendee  # type: ignore[assignment]
            source_signal_event_id = (
                str(ctx.active_signal_id) if ctx.active_signal_id else None
            )
            thread_id = ctx.thread_id

        # Helper to build the pending state we echo back on any
        # clarification from here on. Every field mirrors the current
        # merged view so the frontend has one consistent snapshot.
        def _pending(
            *,
            date_constraint: str | None = ...,  # type: ignore[assignment]
            time_window: str | None = ...,  # type: ignore[assignment]
        ) -> PendingScheduling:
            # Sentinel `...` means "use merged value"; explicit None
            # means "clear this slot" (used when the user must re-pick
            # a time after a busy result).
            return PendingScheduling(
                attendee_email=attendee.email,
                attendee_name=attendee.name,
                attendee_source=attendee.source,
                attendee_canonical_id=attendee.canonical_id,
                source_signal_event_id=source_signal_event_id,
                thread_id=thread_id,
                date_constraint=(
                    merged.date_constraint if date_constraint is ... else date_constraint
                ),
                time_window=(
                    merged.time_window if time_window is ... else time_window
                ),
                duration_minutes=merged.duration_minutes,
                title=merged.title,
            )

        display = attendee.name or attendee.email

        # ── Stage 4: check date/time completeness ───────────────────
        if merged.date_constraint is None and merged.time_window is None:
            return Clarification(
                # `datetime` — NOT `scope`. `scope` triggers the
                # frontend's reconnect-Calendar affordance which is
                # wrong when we're just asking for a day + time.
                field="datetime",
                question=(
                    f"When would you like to schedule the meeting with {display}?"
                ),
                resolved=_pending(),
            )
        if merged.date_constraint is None:
            return Clarification(
                field="date",
                question=(
                    f"Which day would you like to meet {display}?"
                ),
                resolved=_pending(),
            )
        if merged.time_window is None:
            when = _pretty_date_hint(merged.date_constraint)
            return Clarification(
                field="time",
                question=(
                    f"What time {when} would you like to meet {display}?"
                ),
                resolved=_pending(),
            )

        # ── Stage 5: resolve the concrete window ────────────────────
        try:
            window = resolve_window(
                merged, now=ctx.now, time_zone=ctx.time_zone
            )
        except MissingConstraintError:
            # Programmer error — completeness check above should have
            # caught this. Re-raise so it surfaces in logs.
            raise
        except ScopeError as exc:
            msg = str(exc).lower()
            # Map the specific error to the right slot so the frontend
            # focuses the follow-up correctly.
            if "date" in msg or "past" in msg:
                field, question = (
                    "date",
                    (
                        f"That day looks unusable ({exc}). "
                        f"Which day should we meet {display}?"
                    ),
                )
                return Clarification(
                    field=field,
                    question=question,
                    resolved=_pending(date_constraint=None),
                )
            return Clarification(
                field="time",
                question=(
                    f"I couldn't parse that time ({exc}). "
                    f"What time {_pretty_date_hint(merged.date_constraint)} works?"
                ),
                resolved=_pending(time_window=None),
            )

        # ── Stage 6: freebusy ───────────────────────────────────────
        try:
            client, transport = await self._calendar_client_factory(
                self.db, ctx.org_id
            )
        except CalendarNotConnected:
            return Clarification(
                field="scope",
                question=(
                    "Google Calendar isn't connected yet. Connect Google from "
                    "the Sources page and I'll pick a free slot for you."
                ),
                resolved=_pending(),
            )
        except RuntimeError as exc:
            # CalendarScopeInsufficient (connected, wrong scope) and anything
            # else the factory raises. "Reconnect" is the right verb here —
            # the grant exists, it just doesn't cover what we need.
            logger.info(
                "scheduling.agent calendar unavailable org=%s: %s", ctx.org_id, exc
            )
            return Clarification(
                field="scope",
                question=(
                    "Google Calendar is connected, but not with the permission "
                    "Navis needs to check availability and send invites. "
                    "Disconnect and reconnect Google Workspace to grant it."
                ),
                resolved=_pending(),
            )

        alternatives: list[MeetingSlot] = []
        try:
            slot = await find_first_open_slot(
                window, calendar_client=client
            )
            if slot is None and window.kind is WindowKind.EXACT:
                # Exact time busy — look for near-by alternatives so we
                # can surface them in the clarification.
                alternatives = await find_alternative_slots(
                    window, calendar_client=client
                )
        except Exception as exc:  # noqa: BLE001 — Google is external
            logger.exception("scheduling.agent freebusy failed org=%s", ctx.org_id)
            # Distinguish the "scope insufficient" case (Google 403
            # ACCESS_TOKEN_SCOPE_INSUFFICIENT) from a generic transient
            # error. Scope errors require a real reconnect — the user
            # sees a clear message about it. Everything else is
            # transient and worth a retry.
            err_str = str(exc)
            is_scope_error = (
                "insufficientPermissions" in err_str
                or "ACCESS_TOKEN_SCOPE_INSUFFICIENT" in err_str
                or "insufficient authentication scopes" in err_str.lower()
            )
            question = (
                (
                    "Google Calendar is connected but the current grant "
                    "doesn't include availability access. Disconnect and "
                    "reconnect Google Workspace to grant the new scope."
                )
                if is_scope_error
                else (
                    "I couldn't check your calendar availability just now. "
                    "Try reconnecting Google Calendar if this keeps happening."
                )
            )
            return Clarification(
                field="scope",
                question=question,
                resolved=_pending(),
            )
        finally:
            if hasattr(transport, "aclose"):
                try:
                    await transport.aclose()
                except Exception:  # noqa: BLE001
                    logger.exception("scheduling.agent transport close failed")

        if slot is None:
            # Different UX for exact-busy vs flexible-full:
            if window.kind is WindowKind.EXACT:
                if alternatives:
                    options = [
                        _slot_option(s, ctx.time_zone) for s in alternatives
                    ]
                    when = _pretty_time_local(
                        window.start_utc, ctx.time_zone
                    )
                    labels = ", ".join(o["label"] for o in options)
                    return Clarification(
                        field="time",
                        question=(
                            f"{when} is busy. I found {labels} available. "
                            "Which works?"
                        ),
                        options=options,
                        # Clear time_window so the user's next answer
                        # sets a fresh time; keep the date intact.
                        resolved=_pending(time_window=None),
                    )
                return Clarification(
                    field="time",
                    question=(
                        f"{_pretty_time_local(window.start_utc, ctx.time_zone)} "
                        "is busy and nothing free nearby. Try a different time?"
                    ),
                    resolved=_pending(time_window=None),
                )
            # Flexible window — nothing free in the whole afternoon /
            # morning / range. Ask for another day or a wider window.
            return Clarification(
                field="date",
                question=(
                    f"Your calendar is fully booked "
                    f"{_pretty_date_hint(merged.date_constraint)} in that window. "
                    "Want to try another day?"
                ),
                resolved=_pending(date_constraint=None, time_window=None),
            )

        # ── Stage 7: assemble proposal ──────────────────────────────
        title = _default_title(merged.title, attendee)
        assumptions = list(window.assumptions)
        if merged.title is None:
            assumptions.append(
                f'Titled it "{title}" — change before approving if you want '
                "something else."
            )
        return ScheduledProposal(
            attendee=attendee,
            slot=slot,
            title=title,
            description=None,
            source_signal_event_id=source_signal_event_id,
            thread_id=thread_id,
            assumptions=assumptions,
        )


def _local_now(now: datetime, tz_name: str) -> datetime:
    """`now` in the founder's timezone; falls back to `now` on an unknown zone
    (the same tolerant behaviour `resolve_window` applies)."""
    try:
        return now.astimezone(ZoneInfo(tz_name))
    except Exception:  # noqa: BLE001 — bad tz is a user-data problem, not fatal
        return now


# ── Merge helpers ──────────────────────────────────────────────────────


def _merge_constraints(
    extracted: MeetingConstraints, pending: PendingScheduling | None
) -> MeetingConstraints:
    """New extraction wins per-slot; pending fills nulls.

    Pronoun and attendee_hint are NEVER carried across turns — they're
    utterance-scoped intent tokens, not resolved state. Attendee
    resolution itself is handled separately at the agent level.
    """
    if pending is None:
        return extracted
    return MeetingConstraints(
        date_constraint=extracted.date_constraint or pending.date_constraint,
        time_window=extracted.time_window or pending.time_window,
        duration_minutes=extracted.duration_minutes or pending.duration_minutes,
        title=extracted.title or pending.title,
        attendee_hint=extracted.attendee_hint,
        pronoun=extracted.pronoun,
    )


# ── Clarification builders ─────────────────────────────────────────────


def _person_disambiguation(
    candidates,
    merged: MeetingConstraints,
    ctx: SchedulingContext,
    *,
    pending_no_attendee: bool,
) -> Clarification:
    """Multiple people matched the name hint — surface a pick list.

    Pending state carries the DATE/TIME/DURATION/TITLE the user has
    already given so picking a candidate email in a follow-up
    completes the request. Attendee fields stay null.
    """
    options = [
        {
            "canonical_id": m.canonical_id,
            "display_name": m.primary_display_name,
            "email": m.primary_email,
            "emails": list(m.emails),
        }
        for m in candidates
    ]
    label = merged.attendee_hint or "that person"
    pending = PendingScheduling(
        date_constraint=merged.date_constraint,
        time_window=merged.time_window,
        duration_minutes=merged.duration_minutes,
        title=merged.title,
        thread_id=ctx.thread_id,
        source_signal_event_id=(
            str(ctx.active_signal_id) if ctx.active_signal_id else None
        ),
    )
    return Clarification(
        field="person_disambiguation",
        question=(
            f'I found {len(candidates)} people matching "{label}". '
            "Which one did you mean?"
        ),
        options=options,
        resolved=pending,
    )


def _attendee_clarification(
    merged: MeetingConstraints,
    ctx: SchedulingContext,
    *,
    extracted: MeetingConstraints,
) -> Clarification:
    """No attendee could be resolved — ask the user directly.

    Preserves any date/time/duration the user already provided so
    when they answer the WHO question their previous WHEN input is
    still in force.
    """
    if extracted.pronoun:
        question = (
            "I couldn't tell who you meant by "
            f'"{extracted.pronoun.replace("_", " ")}". Which person '
            "should I invite?"
        )
    elif extracted.attendee_hint:
        question = (
            f'I couldn\'t find anyone named "{extracted.attendee_hint}" '
            "in your workspace yet. Do you have their email address?"
        )
    else:
        question = "Who should I schedule this meeting with?"
    pending = PendingScheduling(
        date_constraint=merged.date_constraint,
        time_window=merged.time_window,
        duration_minutes=merged.duration_minutes,
        title=merged.title,
        thread_id=ctx.thread_id,
        source_signal_event_id=(
            str(ctx.active_signal_id) if ctx.active_signal_id else None
        ),
    )
    return Clarification(field="attendee", question=question, resolved=pending)


# ── Rendering helpers ──────────────────────────────────────────────────


def _default_title(candidate: str | None, attendee: ResolvedAttendee) -> str:
    if candidate and candidate.strip():
        return candidate.strip()[:200]
    display = attendee.name or attendee.email
    return f"Meeting with {display}"[:200]


def _pretty_date_hint(date_constraint: str) -> str:
    """Small helper used in clarification prose (Monday-Friday, etc)."""
    c = date_constraint.strip().lower()
    if c in {"today", "tomorrow"}:
        return c
    if c in {
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
    }:
        return f"on {c.capitalize()}"
    return f"on {date_constraint}"


def _pretty_time_local(dt_utc: datetime, tz_name: str) -> str:
    """Format a UTC datetime as local HH:MM AM/PM without OS-specific
    strftime directives."""
    try:
        local = dt_utc.astimezone(ZoneInfo(tz_name))
    except Exception:  # noqa: BLE001 — bad tz falls back to UTC
        local = dt_utc
    hour = local.hour
    minute = local.minute
    meridiem = "AM" if hour < 12 else "PM"
    display_hour = hour % 12 or 12
    if minute == 0:
        return f"{display_hour}:00 {meridiem}"
    return f"{display_hour}:{minute:02d} {meridiem}"


def _slot_option(slot: MeetingSlot, tz_name: str) -> dict:
    """Turn a MeetingSlot into a Clarification.option dict.

    The `time_window` value is a proper "HH:MM-HH:MM" range so if the
    user picks this option the next turn's follow-up sets a fresh
    flexible window that covers exactly this slot — no re-parsing.
    """
    try:
        tz = ZoneInfo(tz_name)
        local_start = slot.start.astimezone(tz)
        local_end = slot.end.astimezone(tz)
    except Exception:  # noqa: BLE001
        local_start, local_end = slot.start, slot.end
    label = _pretty_time_local(slot.start, tz_name)
    hhmm = f"{local_start.hour:02d}:{local_start.minute:02d}-{local_end.hour:02d}:{local_end.minute:02d}"
    return {
        "label": label,
        "time_window": hhmm,
        "start": slot.start.isoformat(),
        "end": slot.end.isoformat(),
        "duration_minutes": slot.duration_minutes,
    }
