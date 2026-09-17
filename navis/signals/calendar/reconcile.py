"""Read-after-write for calendar mutations Navis performs itself.

THE GAP THIS CLOSES
-------------------
Execution talks to Google directly. Nothing told Navis's own read model about
it. The ONLY producer of calendar signals was the sync connector, driven by
Celery beat / Pub/Sub — so between "we cancelled the meeting on Google" and
"the next sync happens to run", every Navis-side reader still served the
pre-mutation world:

    signal_events                     still holds the CREATED/UPDATED row
    memory_records                    still holds `status: confirmed` facts
    /signals/calendar/upcoming        still lists the meeting (dashboard card)
    api.workspace_fallback            still cites it in chat
    ContextBuilder -> RAG             still cites it in chat

`ACTION_EXECUTED` had exactly one subscriber (Observation's outcome capture),
which writes no signals. So the assistant answering "you still have that
meeting" was not a caching bug — the read model had genuinely never been told.

WHAT THIS DOES
--------------
One targeted GET against Google for the single event we just touched, turned
into a signal event and ingested through the NORMAL ingestion path
(`SignalIngestionService`), so it flows to Memory / State / the outbox exactly
as a synced event would. Present -> current version; 404/410/`status:cancelled`
-> a CALENDAR_EVENT_CANCELLED row.

WHERE IT RUNS
-------------
The execution worker calls this AFTER the provider succeeds and BEFORE the
request transitions to EXECUTED. That ordering is the contract, and it is what
makes read-after-write checkable rather than hopeful:

    status == EXECUTED  =>  Navis's read model already reflects the change.

The frontend polls until EXECUTED, so by the time the card says "Meeting
cancelled" the next chat turn is already consistent.

Failure is deliberately NON-fatal: a reconcile that raises must not flip a
genuinely-successful cancellation to FAILED (that would strand a deleted
meeting in a retry loop). We log loudly, record a metric, and leave the row to
converge on the next scheduled sync — degraded, but never wrong about whether
the side effect happened.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from navis.identity.models import Integration
from navis.platform.monitoring import get_logger
from navis.platform.monitoring.metrics import metrics
from navis.signals.base import (
    SignalContext,
    SignalEvent,
    SignalEventType,
    SignalSource,
)
from navis.signals.calendar.client import CalendarClient
from navis.signals.calendar.connector import CalendarConnector
from navis.signals.google.oauth import TokenManager
from navis.signals.google.resolve import resolve_calendar_integration
from navis.signals.google.transport import GoogleAPIError, GoogleTransport
from navis.signals.service import SignalIngestionService

logger = get_logger("signals.calendar.reconcile")

# Action types whose success changes what is on the calendar. `create_task` and
# the Gmail/Slack verbs are deliberately absent — they touch no event.
CALENDAR_MUTATION_ACTION_TYPES = frozenset(
    {"schedule_meeting", "reschedule_meeting", "cancel_meeting"}
)


def event_id_for(action_type: str, payload: dict[str, Any], provider_reference: str | None) -> str | None:
    """Which Google event did this execution touch?

    For a mutation the id is in the payload (it names an existing event). For a
    creation the id only exists once Google has replied, so it comes from the
    provider reference the sender returned.
    """
    if action_type not in CALENDAR_MUTATION_ACTION_TYPES:
        return None
    from_payload = payload.get("event_id") if isinstance(payload, dict) else None
    if isinstance(from_payload, str) and from_payload.strip():
        return from_payload.strip()
    if isinstance(provider_reference, str) and provider_reference.strip():
        return provider_reference.strip()
    return None


@dataclass(frozen=True, slots=True)
class _CalendarGrant:
    """A DETACHED snapshot of the org's Calendar credential.

    Same reasoning as `execution.providers._senders._Grant`, and this path had
    the same two defects. It held the live `Integration` ORM row, assigned the
    DECRYPTED tokens onto it, and then kept that session open across the Google
    call and through `SignalIngestionService.ingest` — whose flush, and the
    `session_scope` exit commit behind it, wrote the real access and refresh
    tokens into `integrations` IN PLAINTEXT. This runs after EVERY calendar
    mutation Navis performs, so it was the highest-frequency instance of it.

    Tokens are carried AS STORED (ciphertext); `TokenManager` decrypts.
    """

    integration_id: uuid.UUID
    user_id: uuid.UUID | None
    access_token: str | None
    refresh_token: str | None
    token_expires_at: Any


async def reconcile_calendar_event(org_id: uuid.UUID, event_id: str) -> bool:
    """Pull one event's CURRENT state from Google into signal_events.

    THREE PHASES, and the boundaries are load-bearing:

        session 1   resolve the grant, persist any self-heal, DETACH
        no session  ask Google for the event
        session 2   ingest the resulting signal

    The `db` parameter this used to accept is gone. Nothing passed it, and
    honouring it would have required holding the caller's transaction across the
    Google call — exactly what the execution worker's Phase B forbids and what
    this function was doing.

    Returns True when a new signal row was written, False when nothing changed
    (the version was already ingested) or the calendar was unreachable. Never
    raises — see the module docstring for why.
    """
    if not event_id:
        return False
    grant = await _load_grant(org_id, event_id)
    if grant is None:
        return False
    raw = await _fetch_event(grant, org_id, event_id)
    if raw is None:
        return False
    return await _ingest(org_id, grant, event_id, raw)


async def _load_grant(org_id: uuid.UUID, event_id: str) -> _CalendarGrant | None:
    """Phase 1 — short session, closed on return."""
    from navis.platform.database import session_scope

    try:
        async with session_scope() as db:
            integration = await resolve_calendar_integration(db, org_id)
            # Persist a self-heal while the stored tokens are still ciphertext.
            # Nothing here mutates the ORM row, so this can only persist the heal.
            await db.commit()
            return _CalendarGrant(
                integration_id=integration.id,
                user_id=integration.user_id,
                access_token=integration.access_token,
                refresh_token=integration.refresh_token,
                token_expires_at=integration.token_expires_at,
            )
    except Exception:  # noqa: BLE001 — no usable grant; sync will converge later
        logger.warning(
            "calendar reconcile skipped (no usable calendar grant) org=%s event=%s",
            org_id, event_id,
        )
        metrics.incr("calendar.reconcile.skipped", labels={"reason": "no_integration"})
        return None


async def _fetch_event(
    grant: _CalendarGrant, org_id: uuid.UUID, event_id: str
) -> dict[str, Any] | None:
    """Phase 2 — the external call, with NO database session open."""
    token_manager = TokenManager(grant)
    transport = GoogleTransport(
        token_manager.token_provider(), on_auth_failure=token_manager.invalidate
    )
    try:
        return await CalendarClient(transport).get_event(event_id)
    except GoogleAPIError as exc:
        if exc.is_not_found or exc.is_gone:
            # The event is gone from Google — which for a cancellation is the
            # SUCCESS case, and the single most important thing to record.
            return {"id": event_id, "status": "cancelled"}
        logger.exception("calendar reconcile failed org=%s event=%s", org_id, event_id)
        metrics.incr("calendar.reconcile.failed", labels={"reason": "google_error"})
        return None
    except Exception:  # noqa: BLE001 — network / transport
        logger.exception("calendar reconcile errored org=%s event=%s", org_id, event_id)
        metrics.incr("calendar.reconcile.failed", labels={"reason": "transport"})
        return None
    finally:
        try:
            await transport.aclose()
        except Exception:  # noqa: BLE001
            pass


async def _ingest(
    org_id: uuid.UUID, grant: _CalendarGrant, event_id: str, raw: dict[str, Any]
) -> bool:
    """Phase 3 — short session for the write-back."""
    from navis.platform.database import session_scope

    signal = _to_signal(org_id, grant, raw)
    if signal is None:
        return False
    try:
        async with session_scope() as db:
            written = await SignalIngestionService(db).ingest(signal)
    except Exception:  # noqa: BLE001 — never fail the execution over bookkeeping
        logger.exception(
            "calendar reconcile ingest failed org=%s event=%s", org_id, event_id
        )
        metrics.incr("calendar.reconcile.failed", labels={"reason": "ingest"})
        return False

    metrics.incr(
        "calendar.reconcile.ok",
        labels={"result": "written" if written else "duplicate"},
    )
    logger.info(
        "calendar reconciled org=%s event=%s status=%s written=%s",
        org_id, event_id, raw.get("status"), written,
    )
    return written


def _to_signal(
    org_id: uuid.UUID, grant: _CalendarGrant, raw: dict[str, Any]
) -> SignalEvent | None:
    """Reuse the CONNECTOR's mapping so a reconciled event is byte-identical to
    a synced one — same event-type rules, same payload minimisation, same
    version key. Two code paths producing subtly different signal rows for the
    same meeting would be its own bug."""
    ctx = SignalContext(
        org_id=org_id,
        user_id=grant.user_id,
        integration_id=grant.integration_id,
    )
    try:
        return CalendarConnector._to_event(
            ctx, raw, calendar_id="primary", capture_mode="reconcile"
        )
    except Exception:  # noqa: BLE001 — a malformed Google payload is not fatal
        logger.exception("calendar reconcile could not map event %s", raw.get("id"))
        return None


def cancellation_signal(
    org_id: uuid.UUID, integration: Integration, event_id: str
) -> SignalEvent:
    """A CALENDAR_EVENT_CANCELLED signal for `event_id`, without calling Google.

    Used by tests and by any caller that already KNOWS the event is gone. The
    version key is time-based because a deleted event has no etag — two
    cancellations of the same event would otherwise dedupe into one row and the
    second would silently vanish.
    """
    now = datetime.now(timezone.utc)
    return SignalEvent.create(
        SignalContext(
            org_id=org_id, user_id=integration.user_id, integration_id=integration.id
        ),
        source=SignalSource.CALENDAR,
        event_type=SignalEventType.CALENDAR_EVENT_CANCELLED,
        external_id=event_id,
        external_version=f"cancelled:{now.isoformat()}",
        timestamp=now,
        payload={"status": "cancelled", "summary": None, "attendees": []},
        metadata={"calendar_id": "primary", "capture_mode": "reconcile"},
    )


