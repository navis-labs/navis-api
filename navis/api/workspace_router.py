"""Workspace router — calendar availability query.

One endpoint here: `POST /workspace/calendar/available-slots`. The founder's
frontend MeetingScheduleDrawer calls it to render 3–5 pickable open slots
from the org's real Google Calendar.

Meeting scheduling itself is NOT here — it flows through the canonical
ExecutionService pipeline (POST /actions with action_type='schedule_meeting'),
so every scheduled meeting is auditable, idempotent per (org, decision,
payload), and can be approval-gated per policy. See
`navis.execution.providers._senders.calendar_sender` for the real Google
Calendar event creation, wired into `CalendarProvider` by the provider
registry.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from navis.identity.authz import Principal, require_permission
from navis.identity.models import Integration, IntegrationProvider
from navis.identity.permissions import Permission
from navis.organization.contract import Capability
from navis.organization.guards import require_capability
from navis.platform.database import get_session
from navis.platform.monitoring import get_logger
from navis.signals.calendar import CalendarClient
from navis.signals.crypto import token_cipher
from navis.signals.google.oauth import TokenManager
from navis.signals.google.transport import GoogleAPIError, GoogleTransport

logger = get_logger("api.workspace_router")

router = APIRouter(prefix="/workspace", tags=["workspace"])


# ── Slot computation constants ──────────────────────────────────────────────

_DEFAULT_DAYS_AHEAD = 5
_DEFAULT_SLOT_MINUTES = 30
_DEFAULT_BUSINESS_START = 9   # 09:00 local
_DEFAULT_BUSINESS_END = 18    # 18:00 local
_MAX_SLOTS_RETURNED = 12       # UI shows at most 5; extra headroom for filtering


class AvailableSlotsRequest(BaseModel):
    days_ahead: int = Field(_DEFAULT_DAYS_AHEAD, ge=1, le=14)
    duration_minutes: int = Field(_DEFAULT_SLOT_MINUTES, ge=15, le=180)
    time_zone: str = Field("UTC", max_length=64)
    # Business-hours window in local (time_zone) hours. Defaults to 09:00–18:00.
    business_start_hour: int = Field(_DEFAULT_BUSINESS_START, ge=0, le=23)
    business_end_hour: int = Field(_DEFAULT_BUSINESS_END, ge=1, le=24)
    # Skip Saturday + Sunday by default. Set false for founders who take
    # weekend meetings (VCs on Sundays, hi Marc Andreessen).
    skip_weekends: bool = True


class Slot(BaseModel):
    start: str  # ISO 8601 with offset
    end: str
    duration_minutes: int


class AvailableSlotsResponse(BaseModel):
    slots: list[Slot]
    time_zone: str
    generated_at: str


# ── Integration helpers ─────────────────────────────────────────────────────


async def _resolve_google_calendar(
    db: AsyncSession, org_id: Any
) -> Integration:
    integration = await db.scalar(
        select(Integration)
        .where(
            Integration.organization_id == org_id,
            Integration.provider == IntegrationProvider.GOOGLE_CALENDAR,
            Integration.is_active.is_(True),
        )
        .order_by(Integration.created_at.desc())
    )
    if integration is None:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "Google Calendar is not connected for this organization",
        )
    if integration.access_token is None or integration.refresh_token is None:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "Google Calendar integration has no stored tokens; reconnect required",
        )
    return integration


# `_decrypt_integration_tokens` lived here and WROTE PLAINTEXT TOKENS TO THE
# DATABASE. It assigned the decrypted values back onto the live `Integration` ORM
# row, which marks the row dirty — and this router's session comes from
# `get_session`, which COMMITS on success (platform/database/session.py). On the
# ordinary path (access token still valid, so `_persist_refreshed_token` returns
# without re-encrypting) that commit persisted the real access AND refresh tokens
# in plaintext.
#
# It then stayed silent, because `token_cipher.decrypt` returns an untagged value
# as-is (the legacy-plaintext path), so everything kept working while
# encryption-at-rest was gone for those columns. The 5-minute `sync_all_sources`
# pass rewrites `access_token` encrypted and so kept repairing it; nothing
# rewrites `refresh_token` unless Google rotates it, which is why the damage
# accumulated on exactly that column.
#
# The helper was also NEVER NEEDED: `TokenManager.get_token` decrypts what it is
# given (signals/google/oauth.py), so handing it the stored CIPHERTEXT is both
# correct and what it already expects — this was decrypting twice. Deleted rather
# than fixed, for the same reason `execution/providers/_senders.py` deleted its
# identical `_decrypt_tokens_inplace`: a helper that hands a live ORM row its own
# plaintext is the shape that reintroduces this bug.


async def _persist_refreshed_token(
    db: AsyncSession, integration: Integration, token_manager: TokenManager
) -> None:
    last = token_manager.last_token
    if last is None or not last.refreshed:
        return
    integration.access_token = token_cipher.encrypt(last.access_token)
    integration.token_expires_at = last.expires_at
    if last.refresh_token is not None:
        integration.refresh_token = token_cipher.encrypt(last.refresh_token)
    await db.commit()


# ── Slot math (pure) ────────────────────────────────────────────────────────


def _compute_open_slots(
    *,
    busy_blocks: list[tuple[datetime, datetime]],
    window_start: datetime,
    window_end: datetime,
    duration: timedelta,
    business_start_hour: int,
    business_end_hour: int,
    skip_weekends: bool,
    max_slots: int,
) -> list[tuple[datetime, datetime]]:
    """Slide a duration-sized cursor across the window, skipping busy blocks."""
    slots: list[tuple[datetime, datetime]] = []
    cursor = window_start
    busy_sorted = sorted(busy_blocks, key=lambda b: b[0])

    while cursor + duration <= window_end and len(slots) < max_slots:
        slot_end = cursor + duration

        if skip_weekends and cursor.weekday() >= 5:
            days_to_monday = 7 - cursor.weekday()
            cursor = (cursor + timedelta(days=days_to_monday)).replace(
                hour=business_start_hour, minute=0, second=0, microsecond=0
            )
            continue

        if cursor.hour < business_start_hour or slot_end.hour > business_end_hour or (
            slot_end.hour == business_end_hour and slot_end.minute > 0
        ):
            cursor = (cursor + timedelta(days=1)).replace(
                hour=business_start_hour, minute=0, second=0, microsecond=0
            )
            continue

        overlaps = False
        for b_start, b_end in busy_sorted:
            if b_end <= cursor:
                continue
            if b_start >= slot_end:
                break
            overlaps = True
            cursor = b_end
            break

        if not overlaps:
            slots.append((cursor, slot_end))
            cursor = slot_end

    return slots


def _parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


# ── Endpoint ────────────────────────────────────────────────────────────────


@router.post("/calendar/available-slots", response_model=AvailableSlotsResponse)
async def available_slots(
    body: AvailableSlotsRequest,
    p: Principal = Depends(require_permission(Permission.EXECUTE_ACTIONS)),
    _: Any = Depends(require_capability(Capability.EXECUTION)),
    db: AsyncSession = Depends(get_session),
) -> AvailableSlotsResponse:
    """Return open time slots on the founder's primary calendar."""
    if body.business_end_hour <= body.business_start_hour:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "business_end_hour must be greater than business_start_hour",
        )

    integration = await _resolve_google_calendar(db, p.org_id)
    # The row is handed over with its tokens still CIPHERTEXT — TokenManager
    # decrypts internally. Nothing here may assign plaintext onto this ORM row:
    # the request's session commits on success. See the note above.
    token_manager = TokenManager(integration)
    transport = GoogleTransport(
        token_manager.token_provider(), on_auth_failure=token_manager.invalidate
    )

    now = datetime.now(timezone.utc)
    window_start = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    window_end = window_start + timedelta(days=body.days_ahead)

    try:
        client = CalendarClient(transport)
        raw = await client.freebusy(
            time_min=window_start.isoformat(),
            time_max=window_end.isoformat(),
            time_zone=body.time_zone,
        )
    except GoogleAPIError as exc:
        if exc.is_auth_error:
            raise HTTPException(
                status.HTTP_401_UNAUTHORIZED,
                "Google Calendar rejected the request; reconnect required",
            )
        logger.exception("freebusy failed org=%s", p.org_id)
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "calendar upstream error")
    finally:
        await _persist_refreshed_token(db, integration, token_manager)
        await transport.aclose()

    calendars = raw.get("calendars") or {}
    busy_blocks: list[tuple[datetime, datetime]] = []
    for cal_data in calendars.values():
        for block in (cal_data.get("busy") or []):
            try:
                busy_blocks.append((_parse_iso(block["start"]), _parse_iso(block["end"])))
            except (KeyError, ValueError):
                continue

    duration = timedelta(minutes=body.duration_minutes)
    open_slots = _compute_open_slots(
        busy_blocks=busy_blocks,
        window_start=window_start,
        window_end=window_end,
        duration=duration,
        business_start_hour=body.business_start_hour,
        business_end_hour=body.business_end_hour,
        skip_weekends=body.skip_weekends,
        max_slots=_MAX_SLOTS_RETURNED,
    )

    return AvailableSlotsResponse(
        slots=[
            Slot(
                start=s.isoformat(),
                end=e.isoformat(),
                duration_minutes=body.duration_minutes,
            )
            for s, e in open_slots
        ],
        time_zone=body.time_zone,
        generated_at=now.isoformat(),
    )
