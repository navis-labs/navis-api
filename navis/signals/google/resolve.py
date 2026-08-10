"""ONE answer to "is Google Calendar connected for this org?".

Before this module there were four independent answers, and they disagreed:

  * `reasoning.scheduling.availability._ManagedCalendarClient` and
    `execution.providers._senders` looked for an ACTIVE row with
    `provider = google_calendar` and hard-failed when it was missing;
  * `signals.read_router` / the dashboard read stored SignalEventRecords, which
    survive independently of the Integration row;
  * `signals.router._backfill_google_family` — the only thing that CREATES the
    per-provider Calendar row for orgs that connected Google before the OAuth
    callback started writing one — ran ONLY inside the manual
    `POST /integrations/google/sync` endpoint.

The visible symptom was the reported one: chat says "Google Calendar isn't
connected", and the very next breath lists your meetings. Both were telling the
truth about different things — a missing `google_calendar` row, and cached
calendar signals that were ingested through the Gmail-family grant.

`resolve_calendar_integration` collapses that into one rule: an org is connected
to Calendar iff SOME active Google-family row for it carries a Calendar scope.
When the scope is present but the dedicated row is missing, we materialise the
row from the grant that already exists (idempotent, ON CONFLICT DO NOTHING) and
return it. Self-healing, so the inconsistency cannot come back.
"""
from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from navis.identity.models import Integration, IntegrationProvider
from navis.platform.monitoring import get_logger
from navis.signals.google.oauth import (
    calendar_read_granted,
    calendar_write_granted,
    gmail_send_granted,
)

logger = get_logger("signals.google.resolve")

# Google-family providers whose OAuth grant can legitimately carry Calendar
# scopes. A single Google consent covers every scope in the request, so the
# tokens on any of these rows work for Calendar when the scope was granted.
_GOOGLE_FAMILY = (
    IntegrationProvider.GOOGLE_CALENDAR,
    IntegrationProvider.GMAIL,
    IntegrationProvider.DRIVE,
)


class CalendarNotConnected(RuntimeError):
    """No active Google grant for this org carries a Calendar scope.

    Distinct from `CalendarScopeInsufficient` so callers can say "connect
    Google" rather than the (wrong, and confusing) "reconnect Google".
    """


class CalendarScopeInsufficient(RuntimeError):
    """Google IS connected, but the grant lacks the scope this call needs."""


async def resolve_calendar_integration(
    db: AsyncSession, org_id: uuid.UUID, *, require_write: bool = False
) -> Integration:
    """Return the org's Calendar-capable Integration, healing the row if needed.

    Raises `CalendarNotConnected` when no active Google grant exists at all, and
    `CalendarScopeInsufficient` when one exists but is missing the Calendar
    read scope (or the write scope, when `require_write`).

    Never commits. The caller owns the transaction — the backfill INSERT is
    flushed so the row is visible to this session, and is committed (or rolled
    back) with whatever else the caller is doing.
    """
    rows = (
        await db.scalars(
            select(Integration)
            .where(
                Integration.organization_id == org_id,
                Integration.provider.in_(_GOOGLE_FAMILY),
                Integration.is_active.is_(True),
            )
            .order_by(Integration.created_at.desc())
        )
    ).all()
    if not rows:
        raise CalendarNotConnected(
            "no active Google integration for this organization"
        )

    dedicated = next(
        (r for r in rows if r.provider == IntegrationProvider.GOOGLE_CALENDAR), None
    )
    # Prefer the dedicated row; otherwise the newest sibling grant that actually
    # carries a Calendar scope (a Gmail-only grant must NOT be treated as
    # Calendar access — that would 403 at Google with a confusing message).
    source = dedicated or next(
        (r for r in rows if calendar_read_granted(r.scopes) or calendar_write_granted(r.scopes)),
        None,
    )
    if source is None:
        raise CalendarScopeInsufficient(
            "Google is connected but no Calendar scope was granted"
        )

    if dedicated is None:
        # Self-heal: materialise the per-provider row from the sibling grant so
        # every OTHER caller (sync orchestrator, workspace router, execution
        # senders) sees a consistent world from here on.
        dedicated = await _backfill_calendar_row(db, org_id, source)
        logger.info(
            "healed missing google_calendar integration org=%s from provider=%s",
            org_id, source.provider,
        )

    if not calendar_read_granted(dedicated.scopes) and not calendar_write_granted(
        dedicated.scopes
    ):
        raise CalendarScopeInsufficient(
            "Google Calendar is connected without an availability (read) scope"
        )
    if require_write and not calendar_write_granted(dedicated.scopes):
        raise CalendarScopeInsufficient(
            "Google Calendar is connected read-only; the calendar.events "
            "write scope is required to create or change events"
        )
    if not dedicated.access_token or not dedicated.refresh_token:
        raise CalendarScopeInsufficient(
            "Google Calendar integration has no stored tokens; reconnect required"
        )
    return dedicated


class GmailNotConnected(RuntimeError):
    """No active Google grant for this org at all."""


class GmailSendScopeInsufficient(RuntimeError):
    """Google IS connected, but no active grant carries a send capability."""


def _send_rank(row: Integration) -> tuple:
    """Total, deterministic order over send-capable grants (best first).

    1. A `gmail` row outranks a sibling Google-family row. One consent covers
       every scope, so a `drive` row's token CAN send — but the Gmail row is the
       sender identity of record, and preferring it keeps the From: address the
       one the user connected for mail.
    2. Then most-recently-connected, which is the closest thing to intent we
       have: reconnecting an account is how a user says "use this one".
    3. Then `id`, so the order is TOTAL. Without it, two grants created in the
       same transaction (identical `created_at`) would tie and the winner would
       depend on the planner — the same class of nondeterminism SIGNAL-REPO-001
       fixes for replay ordering.
    """
    return (
        0 if row.provider == IntegrationProvider.GMAIL else 1,
        -(row.created_at.timestamp() if row.created_at else 0.0),
        str(row.id),
    )


async def resolve_gmail_send_integration(
    db: AsyncSession, org_id: uuid.UUID
) -> Integration:
    """Return the org's SEND-capable Gmail Integration.

    The bug this replaces: both `_senders._resolve_integration` and the router's
    approve-gate took `ORDER BY created_at DESC LIMIT 1` and then checked the
    scope on whatever came back. On a workspace with two connected Google
    accounts that reliably picks the WRONG one — the observed case had a
    read-only grant connected two hours after a send-capable one, so every
    reviewed, approved draft failed a scope pre-flight while a perfectly good
    sender sat one row down.

    Capability is filtered FIRST and recency only breaks ties among grants that
    can actually do the job. A newer grant without `gmail.send` can never
    outrank an older grant that has it.

    Unlike `resolve_calendar_integration` this never INSERTs a healing row: a
    send path should not mutate integration state as a side effect, and no other
    reader depends on a dedicated row existing here.

    Raises `GmailNotConnected` when no active Google grant exists, and
    `GmailSendScopeInsufficient` when one exists but none can send. Both subclass
    RuntimeError, so worker failure handling is unchanged.
    """
    rows = (
        await db.scalars(
            select(Integration).where(
                Integration.organization_id == org_id,
                Integration.provider.in_(_GOOGLE_FAMILY),
                Integration.is_active.is_(True),
            )
        )
    ).all()
    if not rows:
        raise GmailNotConnected("no active Google integration for this organization")

    # Capability first, recency second — the whole point of this function.
    capable = [r for r in rows if gmail_send_granted(r.scopes)]
    if not capable:
        raise GmailSendScopeInsufficient(
            "Gmail is connected without permission to send. Reconnect Google to "
            "grant the gmail.send scope."
        )

    usable = [r for r in capable if r.access_token and r.refresh_token]
    if not usable:
        raise GmailSendScopeInsufficient(
            "the send-capable Gmail integration has no stored tokens; "
            "reconnect required"
        )
    return sorted(usable, key=_send_rank)[0]


async def _backfill_calendar_row(
    db: AsyncSession, org_id: uuid.UUID, source: Integration
) -> Integration:
    """Clone `source`'s grant into a google_calendar row. Idempotent."""
    stmt = (
        pg_insert(Integration)
        .values(
            organization_id=org_id,
            user_id=source.user_id,
            provider=IntegrationProvider.GOOGLE_CALENDAR.value,
            external_account=source.external_account,
            access_token=source.access_token,
            refresh_token=source.refresh_token,
            token_expires_at=source.token_expires_at,
            scopes=source.scopes,
            is_active=True,
            sync_state={},
        )
        .on_conflict_do_nothing(constraint="uq_org_provider_account")
    )
    await db.execute(stmt)
    await db.flush()
    row = await db.scalar(
        select(Integration)
        .where(
            Integration.organization_id == org_id,
            Integration.provider == IntegrationProvider.GOOGLE_CALENDAR,
            Integration.is_active.is_(True),
        )
        .order_by(Integration.created_at.desc())
    )
    if row is None:  # pragma: no cover — the insert just ran
        raise CalendarNotConnected("failed to materialise google_calendar row")
    return row
