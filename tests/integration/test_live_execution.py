"""LIVE external execution verification against REAL Gmail / Calendar / Slack.

STATUS WHEN THIS FILE WAS WRITTEN: **NEVER EXECUTED**. No Google or Slack user
credentials were available in the environment, so every test here skipped and
NOTHING in this file has verified anything against a real external system. Do
not cite it as evidence of external verification until it has actually run and
the output is attached to a report.

WHAT IT DOES WHEN CREDENTIALS EXIST
-----------------------------------
Drives the FULL production path — `ExecutionService.request` -> approve ->
`execute_pass(providers=production_providers())` -> the real sender -> the real
Google/Slack API — against a disposable organization in a disposable database,
then verifies the external object really exists by reading it back, and cleans
up after itself.

Nothing is stubbed. That is the entire point: the unit and HTTP-stub suites
already prove the code's internal logic, and what they cannot prove is that
Google returns 409 on a replayed client-supplied event id, that Gmail indexes
`rfc822msgid:` fast enough for the duplicate pre-check, or that Slack's error
codes are what the classifier expects.

HOW TO RUN
----------
    # 1. a disposable Postgres (NEVER production Neon)
    export NAVIS_TEST_DATABASE_URL="postgresql+asyncpg://navis@127.0.0.1:54329/navis_test"

    # 2. a disposable Google Workspace account's OAuth grant, with
    #    gmail.send + calendar.events + calendar.readonly
    export GOOGLE_TEST_REFRESH_TOKEN=1//0...
    export GOOGLE_TEST_ACCOUNT=sandbox@your-test-domain.com

    # 3. a disposable Slack workspace's bot token with chat:write,
    #    invited to the channel
    export SLACK_TEST_BOT_TOKEN=xoxb-...
    export SLACK_TEST_CHANNEL=C0123456789

    # 4. the explicit arming switch — these tests SEND REAL EMAIL and BOOK REAL
    #    MEETINGS. Nothing runs without it.
    export NAVIS_LIVE_EXECUTION=1

    pytest tests/integration/test_live_execution.py -v

SAFETY
------
  * `NAVIS_LIVE_EXECUTION` must be set explicitly. Credentials alone do not arm.
  * The database URL is REFUSED if it looks like managed/production infrastructure
    (see `_assert_disposable_database`). These tests write and delete rows.
  * Gmail sends only to the connected account itself, so a mistake mails nobody
    but the sandbox. Messages are trashed afterwards.
  * Calendar events are created on the sandbox's own primary calendar with
    `send_google_invite=False`, so no invitation email reaches a human, and are
    deleted afterwards.
  * Slack posts to the channel named by `SLACK_TEST_CHANNEL` and deletes the
    message afterwards.
"""
from __future__ import annotations

import asyncio
import os
import uuid

import pytest

# ── Arming ─────────────────────────────────────────────────────────────

LIVE = os.environ.get("NAVIS_LIVE_EXECUTION") == "1"
DB_URL = os.environ.get("NAVIS_TEST_DATABASE_URL")
GOOGLE_REFRESH = os.environ.get("GOOGLE_TEST_REFRESH_TOKEN")
GOOGLE_ACCOUNT = os.environ.get("GOOGLE_TEST_ACCOUNT")
SLACK_TOKEN = os.environ.get("SLACK_TEST_BOT_TOKEN")
SLACK_CHANNEL = os.environ.get("SLACK_TEST_CHANNEL")

pytestmark = pytest.mark.skipif(
    not (LIVE and DB_URL),
    reason=(
        "live external execution is not armed; set NAVIS_LIVE_EXECUTION=1 and "
        "NAVIS_TEST_DATABASE_URL (see this module's docstring)"
    ),
)

needs_google = pytest.mark.skipif(
    not (GOOGLE_REFRESH and GOOGLE_ACCOUNT),
    reason="set GOOGLE_TEST_REFRESH_TOKEN + GOOGLE_TEST_ACCOUNT",
)
needs_slack = pytest.mark.skipif(
    not (SLACK_TOKEN and SLACK_CHANNEL),
    reason="set SLACK_TEST_BOT_TOKEN + SLACK_TEST_CHANNEL",
)

_MANAGED_HOST_MARKERS = (
    "neon.tech", "rds.amazonaws.com", "supabase.co", "azure.com",
    "cloudsql", "planetscale", "render.com", "railway.app",
)


def _assert_disposable_database(url: str) -> None:
    """Refuse to run destructive live tests against managed infrastructure.

    These tests create organizations, store credentials and delete rows. The
    production database is reachable from this machine (it is in `.env`), so the
    guard is a hard refusal rather than a convention — a typo in an exported
    variable must not be able to point them at real customer data.
    """
    lowered = url.lower()
    for marker in _MANAGED_HOST_MARKERS:
        if marker in lowered:
            raise RuntimeError(
                f"NAVIS_TEST_DATABASE_URL points at what looks like managed/production "
                f"infrastructure ({marker}). Live execution tests write and delete rows "
                f"and must only ever run against a disposable database."
            )


# ── Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def anyio_backend():
    return "asyncio"


@pytest.fixture()
async def engine():
    from sqlalchemy.ext.asyncio import create_async_engine

    import navis.registry  # noqa: F401 — registers every model on Base
    from navis.platform.database import Base

    assert DB_URL is not None
    _assert_disposable_database(DB_URL)
    eng = create_async_engine(DB_URL)
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    await eng.dispose()


@pytest.fixture()
async def live_org(engine):
    """A disposable organization holding the REAL sandbox credential.

    Provisioned exactly as production provisions one (`OrganizationService.
    bootstrap`), because the worker re-checks entitlements immediately before the
    provider call — an unprovisioned org would be correctly refused and the test
    would be asserting the denial path while claiming to test a send.
    """
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from navis.identity.models import (
        Integration,
        IntegrationProvider,
        Organization,
        User,
    )
    from navis.organization.contract import Plan
    from navis.organization.service import OrganizationService
    from navis.signals.crypto import token_cipher
    from navis.signals.google.oauth import GOOGLE_SCOPES
    from navis.signals.slack.oauth import SLACK_SCOPES

    Session = async_sessionmaker(engine, expire_on_commit=False)
    async with Session() as s:
        org = Organization(name="Live Sandbox", slug=f"live-{uuid.uuid4().hex[:8]}")
        user = User(email=GOOGLE_ACCOUNT or f"sandbox-{uuid.uuid4().hex[:6]}@example.com")
        s.add_all([org, user])
        await s.flush()
        # ENTERPRISE carries EXECUTION + GMAIL + CALENDAR + SLACK.
        await OrganizationService(s).bootstrap(org.id, plan=Plan.ENTERPRISE)

        if GOOGLE_REFRESH:
            # Stored exactly as the OAuth callback stores it: ENCRYPTED, with no
            # access token, so the very first call must exercise the real refresh
            # path against Google.
            s.add(Integration(
                organization_id=org.id, user_id=user.id,
                provider=IntegrationProvider.GMAIL,
                external_account=GOOGLE_ACCOUNT,
                access_token=None,
                refresh_token=token_cipher.encrypt(GOOGLE_REFRESH),
                token_expires_at=None,
                scopes=" ".join(GOOGLE_SCOPES),
                is_active=True, sync_state={},
            ))
        if SLACK_TOKEN:
            s.add(Integration(
                organization_id=org.id, user_id=user.id,
                provider=IntegrationProvider.SLACK,
                external_account="live-sandbox",
                access_token=token_cipher.encrypt(SLACK_TOKEN),
                scopes=",".join(SLACK_SCOPES),
                is_active=True, sync_state={},
            ))
        await s.commit()
        org_id, user_id = org.id, user.id

    yield org_id, user_id, Session

    async with Session() as s:
        # Cascades remove the integrations, executions, outbox and dead letters.
        obj = await s.get(Organization, org_id)
        if obj is not None:
            await s.delete(obj)
            await s.commit()


async def _execute_live(Session, org_id, user_id, action_type, payload):
    """Request -> approve -> run the REAL worker pass. Returns the finished row.

    Uses `production_providers()`, so the real senders and the real Google/Slack
    clients run. `session_scope` is NOT patched — the sender opens its own real
    sessions against the disposable database.
    """
    from navis.execution.providers import production_providers
    from navis.execution.repository import ExecutionRepository
    from navis.execution.service import ExecutionService
    from navis.execution.worker import execute_pass

    async with Session() as s:
        req, _ = await ExecutionService(s).request(
            org_id, uuid.uuid4(), action_type, payload,
            reasoning_signal_ids=[f"live-{uuid.uuid4().hex[:8]}"],
            reasoning_version=1, state_watermark="live",
        )
        await ExecutionService(s).approve(req.id, user_id)
        await s.commit()
        execution_id = req.id

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def unit():
        async with Session() as s:
            yield ExecutionRepository(s)
            await s.commit()

    await execute_pass(unit=unit, providers=production_providers(), limit=5)

    async with Session() as s:
        return await ExecutionRepository(s).get(execution_id)


async def _google_client(cls):
    """A real Google API client on the sandbox grant, for VERIFICATION reads.

    Deliberately separate from the sender's own credential path: verification
    must not depend on the machinery under test.
    """
    from navis.signals.google.oauth import TokenManager
    from navis.signals.google.transport import GoogleTransport

    grant = type("_G", (), {
        "access_token": None,
        "refresh_token": GOOGLE_REFRESH,
        "token_expires_at": None,
        "id": "verifier",
    })()
    tm = TokenManager(grant)
    transport = GoogleTransport(tm.token_provider())
    return cls(transport), transport


# ══ GMAIL ══════════════════════════════════════════════════════════════


@needs_google
async def test_live_gmail_send_email_really_arrives(live_org):
    """1, 4 — send_email, and the message provably exists in the real mailbox."""
    from navis.execution.providers._senders import navis_message_id
    from navis.execution.status import ExecutionStatus
    from navis.signals.gmail import GmailClient

    org_id, user_id, Session = live_org
    marker = f"navis-live-{uuid.uuid4().hex[:10]}"
    row = await _execute_live(
        Session, org_id, user_id, "send_email",
        {"to": GOOGLE_ACCOUNT, "subject": f"Navis live check {marker}",
         "body": f"Automated execution verification. Marker: {marker}"},
    )

    assert row.status == ExecutionStatus.EXECUTED.value, row.last_error
    assert row.provider_reference.get("provider") == "gmail"
    external_id = row.provider_reference.get("external_id")
    assert external_id and not row.provider_reference.get("dry_run")

    client, transport = await _google_client(GmailClient)
    try:
        # READ IT BACK. A provider reference is Navis's claim; this is Gmail's.
        got = await client.get_message(external_id)
        assert got.get("id") == external_id
        headers = {
            h["name"].lower(): h["value"]
            for h in (got.get("payload") or {}).get("headers", [])
        }
        assert marker in headers.get("subject", "")
        # 5 — the deterministic Message-ID really is on the delivered message.
        assert headers.get("message-id") == navis_message_id(row.id)
        # Clean up: move it to trash so the sandbox mailbox does not fill up.
        await client._t.request(
            "POST", f"https://gmail.googleapis.com/gmail/v1/users/me/messages/{external_id}/trash"
        )
    finally:
        await transport.aclose()


@needs_google
async def test_live_gmail_replay_does_not_send_twice(live_org):
    """6 — a REPLAY of the same execution must not deliver a second message.

    This is the one that unit tests genuinely cannot settle: it depends on how
    fast Gmail indexes a sent message for `rfc822msgid:` search. The guarantee is
    documented as at-least-once with a narrow window, and this measures whether
    the window is actually closed in practice for a back-to-back retry.
    """
    from navis.execution.providers import production_providers
    from navis.execution.providers._senders import navis_message_id
    from navis.signals.gmail import GmailClient

    org_id, user_id, Session = live_org
    marker = f"navis-replay-{uuid.uuid4().hex[:10]}"
    row = await _execute_live(
        Session, org_id, user_id, "send_email",
        {"to": GOOGLE_ACCOUNT, "subject": f"Navis replay check {marker}",
         "body": f"Replay verification. Marker: {marker}"},
    )
    first_id = row.provider_reference.get("external_id")
    assert first_id

    # Call the sender AGAIN with the same execution id — exactly what a worker
    # retry after a timeout does.
    from navis.execution.providers._senders import gmail_sender

    second_id = await gmail_sender({
        "__ctx__": {"org_id": str(org_id), "execution_id": str(row.id),
                    "decision_id": str(row.decision_id)},
        "to": GOOGLE_ACCOUNT,
        "subject": f"Navis replay check {marker}",
        "body": f"Replay verification. Marker: {marker}",
    })

    client, transport = await _google_client(GmailClient)
    try:
        found = await client.list_messages(
            query=f"rfc822msgid:{navis_message_id(row.id).strip('<>')}"
        )
        ids = [m["id"] for m in (found.get("messages") or [])]
        assert second_id == first_id, (
            "the duplicate pre-check did not catch the replay — a SECOND message "
            f"was delivered ({second_id} != {first_id})"
        )
        assert len(ids) == 1, f"{len(ids)} messages carry one Message-ID: {ids}"
        for mid in ids:
            await client._t.request(
                "POST",
                f"https://gmail.googleapis.com/gmail/v1/users/me/messages/{mid}/trash",
            )
    finally:
        await transport.aclose()
        assert production_providers()["send_email"].is_live


@needs_google
async def test_live_gmail_invalid_credential_fails_permanently(live_org):
    """7 — a revoked/garbage grant must dead-letter with a reconnect instruction,
    not burn the retry budget."""
    from sqlalchemy import select, update

    from navis.execution.status import ExecutionStatus
    from navis.identity.models import Integration, IntegrationProvider
    from navis.signals.crypto import token_cipher

    org_id, user_id, Session = live_org
    async with Session() as s:
        await s.execute(
            update(Integration)
            .where(Integration.organization_id == org_id,
                   Integration.provider == IntegrationProvider.GMAIL)
            .values(refresh_token=token_cipher.encrypt("1//0-definitely-not-valid"))
        )
        await s.commit()

    row = await _execute_live(
        Session, org_id, user_id, "send_email",
        {"to": GOOGLE_ACCOUNT, "subject": "should never send", "body": "should never send"},
    )
    assert row.status == ExecutionStatus.FAILED.value
    assert row.attempts == 1, "an unusable credential must not be retried"
    for secret in ("1//0-definitely-not-valid", "Bearer ", "ya29."):
        assert secret not in (row.last_error or "")

    async with Session() as s:
        from navis.execution.models import ExecutionDeadLetter

        dlq = (await s.scalars(
            select(ExecutionDeadLetter).where(ExecutionDeadLetter.execution_id == row.id)
        )).all()
        assert len(dlq) == 1
        assert "reconnect" in dlq[0].error.lower()


# ══ CALENDAR + MEET ════════════════════════════════════════════════════


@needs_google
async def test_live_calendar_full_lifecycle_with_meet(live_org):
    """1-5, 8 — create (with a real Google Meet), verify, reschedule, verify,
    cancel, verify it is gone."""
    from datetime import datetime, timedelta, timezone

    from navis.execution.status import ExecutionStatus
    from navis.signals.calendar import CalendarClient
    from navis.signals.google.transport import GoogleAPIError

    org_id, user_id, Session = live_org
    start = datetime.now(timezone.utc) + timedelta(days=3)
    end = start + timedelta(minutes=30)

    created = await _execute_live(
        Session, org_id, user_id, "schedule_meeting",
        {"title": f"Navis live check {uuid.uuid4().hex[:8]}",
         "start": start.isoformat(), "end": end.isoformat(),
         "time_zone": "UTC", "conference": True,
         # No invitation mail reaches a human.
         "send_google_invite": False, "attendees": []},
    )
    assert created.status == ExecutionStatus.EXECUTED.value, created.last_error
    event_id = created.provider_reference.get("external_id")
    assert event_id
    # 2 — Google Meet really was attached, via Calendar conferenceData.
    assert created.provider_reference.get("meet_url", "").startswith("https://meet.google.com/"), (
        f"no real Meet link on the reference: {created.provider_reference}"
    )

    client, transport = await _google_client(CalendarClient)
    try:
        live = await client.get_event(event_id)
        assert live["id"] == event_id and live.get("status") != "cancelled"
        assert live.get("hangoutLink"), "Google did not attach a conference"

        # 3 — reschedule, then verify the real event moved.
        new_start = start + timedelta(hours=2)
        moved = await _execute_live(
            Session, org_id, user_id, "reschedule_meeting",
            {"event_id": event_id, "start": new_start.isoformat(),
             "end": (new_start + timedelta(minutes=30)).isoformat(),
             "time_zone": "UTC", "send_google_invite": False},
        )
        assert moved.status == ExecutionStatus.EXECUTED.value, moved.last_error
        after = await client.get_event(event_id)
        assert after["start"]["dateTime"][:16] == new_start.isoformat()[:16]
        # The conference survives a PATCH — the reason no separate Meet action exists.
        assert after.get("hangoutLink") == live.get("hangoutLink")

        # 4 — cancel, then verify Google agrees it is gone.
        cancelled = await _execute_live(
            Session, org_id, user_id, "cancel_meeting",
            {"event_id": event_id, "send_google_invite": False},
        )
        assert cancelled.status == ExecutionStatus.EXECUTED.value, cancelled.last_error
        try:
            gone = await client.get_event(event_id)
            assert gone.get("status") == "cancelled", gone
        except GoogleAPIError as exc:
            assert exc.is_not_found or exc.is_gone
    finally:
        await transport.aclose()


@needs_google
async def test_live_calendar_replay_returns_the_same_event(live_org):
    """6, 7 — THE claim that has never been observed: replaying one execution
    must NOT book a second meeting.

    Calendar is described as exactly-once because `events.insert` takes a
    client-supplied id and 409s on replay. Every test so far asserted our
    HANDLING of a 409; none has ever seen Google produce one. This calls the
    sender twice with the same execution id and checks the calendar.
    """
    from datetime import datetime, timedelta, timezone

    from navis.execution.providers._senders import _google_event_id, calendar_sender
    from navis.execution.status import ExecutionStatus
    from navis.signals.calendar import CalendarClient

    org_id, user_id, Session = live_org
    start = datetime.now(timezone.utc) + timedelta(days=4)
    row = await _execute_live(
        Session, org_id, user_id, "schedule_meeting",
        {"title": f"Navis replay check {uuid.uuid4().hex[:8]}",
         "start": start.isoformat(), "end": (start + timedelta(minutes=30)).isoformat(),
         "time_zone": "UTC", "conference": False,
         "send_google_invite": False, "attendees": []},
    )
    assert row.status == ExecutionStatus.EXECUTED.value, row.last_error
    first_id = row.provider_reference["external_id"]
    assert first_id == _google_event_id(row.id), "the event id must be deterministic"

    payload = {
        "__ctx__": {"org_id": str(org_id), "execution_id": str(row.id),
                    "decision_id": str(row.decision_id)},
        "title": "Navis replay check", "start": start.isoformat(),
        "end": (start + timedelta(minutes=30)).isoformat(),
        "time_zone": "UTC", "conference": False,
        "send_google_invite": False, "attendees": [],
    }
    second_id, _extra = await calendar_sender(payload)

    client, transport = await _google_client(CalendarClient)
    try:
        assert second_id == first_id, "a replay created a SECOND calendar event"
        listing = await client._t.request(
            "GET", "https://www.googleapis.com/calendar/v3/calendars/primary/events",
            params={"q": "Navis replay check", "showDeleted": "false"},
        )
        matching = [e for e in listing.get("items", []) if e.get("id") == first_id]
        assert len(matching) == 1, f"{len(matching)} events share the execution id"
        await client.delete_event(first_id, send_updates="none")
    finally:
        await transport.aclose()


# ══ SLACK ══════════════════════════════════════════════════════════════


@needs_slack
async def test_live_slack_posts_and_the_message_exists(live_org):
    """1-3, 6 — post to a real workspace, verify it exists, verify Observation."""
    from navis.execution.status import ExecutionStatus
    from navis.platform.events import EventType
    from navis.signals.slack.client import SlackClient

    org_id, user_id, Session = live_org
    marker = f"navis-live-{uuid.uuid4().hex[:10]}"
    row = await _execute_live(
        Session, org_id, user_id, "send_slack",
        {"channel": SLACK_CHANNEL, "text": f"Navis execution verification {marker}"},
    )
    assert row.status == ExecutionStatus.EXECUTED.value, row.last_error
    ts = row.provider_reference.get("external_id")
    assert ts and not row.provider_reference.get("dry_run")

    client = SlackClient(SLACK_TOKEN or "")
    try:
        history = await client._request(
            "conversations.history",
            {"channel": SLACK_CHANNEL, "latest": ts, "inclusive": "true", "limit": "1"},
        )
        messages = history.get("messages") or []
        assert messages and marker in messages[0].get("text", "")
        await client._request("chat.delete", {"channel": SLACK_CHANNEL, "ts": ts})
    finally:
        await client.aclose()

    # 6 — the connector-neutral Observation envelope, from a REAL execution.
    from sqlalchemy import select

    from navis.execution.models import ExecutionOutbox

    async with Session() as s:
        events = (await s.scalars(
            select(ExecutionOutbox).where(ExecutionOutbox.execution_id == row.id)
        )).all()
    executed = [e for e in events if e.event_type == EventType.ACTION_EXECUTED.value]
    assert len(executed) == 1
    outcome = executed[0].payload["outcome"]
    assert outcome["org_id"] == str(org_id)
    assert outcome["execution_id"] == str(row.id)
    assert outcome["decision_id"] == str(row.decision_id)
    assert outcome["action_type"] == "send_slack"
    assert outcome["provider_reference"]["provider"] == "slack"
    for e in events:
        assert "xoxb" not in str(e.payload), "a Slack token reached the outbox"


# ══ SECURITY, on the live path ═════════════════════════════════════════


@needs_google
async def test_live_forged_payload_context_cannot_borrow_another_orgs_credential(
    engine, live_org
):
    """The cross-tenant attack, against the real credential path.

    A second organization with NO Google grant requests a send whose payload
    carries a forged `__ctx__.org_id` naming the org that DOES have one. The
    worker stamps the context from the persisted row, so the forged value is
    overwritten and the send fails for want of a credential — it must never
    borrow the sandbox's.
    """
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from navis.execution.status import ExecutionStatus
    from navis.identity.models import Organization, User
    from navis.organization.contract import Plan
    from navis.organization.service import OrganizationService

    victim_org, _victim_user, Session = live_org
    Other = async_sessionmaker(engine, expire_on_commit=False)
    async with Other() as s:
        org = Organization(name="Attacker", slug=f"atk-{uuid.uuid4().hex[:8]}")
        user = User(email=f"atk-{uuid.uuid4().hex[:6]}@example.com")
        s.add_all([org, user])
        await s.flush()
        await OrganizationService(s).bootstrap(org.id, plan=Plan.ENTERPRISE)
        await s.commit()
        attacker_org, attacker_user = org.id, user.id

    row = await _execute_live(
        Session, attacker_org, attacker_user, "send_email",
        {"to": GOOGLE_ACCOUNT, "subject": "forged", "body": "forged",
         "__ctx__": {"org_id": str(victim_org),
                     "execution_id": str(uuid.uuid4()),
                     "decision_id": str(uuid.uuid4())}},
    )
    assert row.organization_id == attacker_org
    assert row.status == ExecutionStatus.FAILED.value, (
        "an org with no Google grant SENT MAIL — the payload selected the tenant"
    )
    assert "not connected" in (row.last_error or "").lower()


@needs_google
async def test_live_revoked_capability_stops_a_real_send(live_org):
    """Post-approval revocation, on the real path: the row is APPROVED, then the
    connector capability is revoked, and no mail may leave."""
    from navis.execution.providers import production_providers
    from navis.execution.repository import ExecutionRepository
    from navis.execution.service import ExecutionService
    from navis.execution.status import ExecutionStatus
    from navis.execution.worker import execute_pass
    from navis.organization.contract import Capability
    from navis.organization.service import OrganizationService

    org_id, user_id, Session = live_org
    async with Session() as s:
        req, _ = await ExecutionService(s).request(
            org_id, uuid.uuid4(), "send_email",
            {"to": GOOGLE_ACCOUNT, "subject": "must not send", "body": "must not send"},
            reasoning_signal_ids=[], reasoning_version=1, state_watermark="w",
        )
        await ExecutionService(s).approve(req.id, user_id)
        await s.commit()
        execution_id = req.id

    async with Session() as s:
        await OrganizationService(s).set_capability(org_id, Capability.GMAIL, False)
        await s.commit()

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def unit():
        async with Session() as s:
            yield ExecutionRepository(s)
            await s.commit()

    await execute_pass(unit=unit, providers=production_providers(), limit=5)

    async with Session() as s:
        row = await ExecutionRepository(s).get(execution_id)
    assert row.status == ExecutionStatus.FAILED.value
    assert "gmail" in (row.last_error or "").lower()
    assert not row.provider_reference, "no external side effect may be recorded"


# ══ TRANSACTION BOUNDARY, on the live path ═════════════════════════════


@needs_google
async def test_live_send_holds_no_database_session_during_the_google_call(live_org):
    """The transaction-boundary invariant, measured against the REAL sender and
    the REAL Google API rather than a stubbed transport.

    Instruments the engine's connection checkout so the count is of genuine
    pooled connections, not of a substituted session factory.
    """
    from sqlalchemy import event as sa_event

    from navis.execution.status import ExecutionStatus
    from navis.platform.database import engine as prod_engine

    checked_out = {"n": 0}
    observed: list[int] = []

    def _on_checkout(*_a):
        checked_out["n"] += 1

    def _on_checkin(*_a):
        checked_out["n"] -= 1

    sa_event.listen(prod_engine.sync_engine, "checkout", _on_checkout)
    sa_event.listen(prod_engine.sync_engine, "checkin", _on_checkin)

    import navis.signals.google.transport as gt

    original = gt.GoogleTransport._do_request

    async def _record(self, method, url, **kw):
        observed.append(checked_out["n"])
        return await original(self, method, url, **kw)

    gt.GoogleTransport._do_request = _record
    try:
        org_id, user_id, Session = live_org
        marker = uuid.uuid4().hex[:10]
        row = await _execute_live(
            Session, org_id, user_id, "send_email",
            {"to": GOOGLE_ACCOUNT, "subject": f"Navis txn check {marker}",
             "body": f"Transaction boundary verification {marker}"},
        )
        assert row.status == ExecutionStatus.EXECUTED.value, row.last_error
        assert observed, "no Google request was observed"
        assert all(n == 0 for n in observed), (
            f"a pooled DB connection was held during a Google call: {observed}"
        )
    finally:
        gt.GoogleTransport._do_request = original
        sa_event.remove(prod_engine.sync_engine, "checkout", _on_checkout)
        sa_event.remove(prod_engine.sync_engine, "checkin", _on_checkin)


@needs_google
async def test_live_refreshed_token_is_persisted_encrypted(live_org):
    """The sandbox grant starts with NO access token, so the first send must
    refresh against Google. Afterwards the stored token must be present and
    ENCRYPTED — never the plaintext the 5.4.2 audit found being written."""
    from sqlalchemy import select

    from navis.execution.status import ExecutionStatus
    from navis.identity.models import Integration, IntegrationProvider
    from navis.signals.crypto import token_cipher

    org_id, user_id, Session = live_org
    row = await _execute_live(
        Session, org_id, user_id, "send_email",
        {"to": GOOGLE_ACCOUNT, "subject": "Navis refresh check",
         "body": "Token refresh persistence verification."},
    )
    assert row.status == ExecutionStatus.EXECUTED.value, row.last_error

    async with Session() as s:
        integ = (await s.scalars(
            select(Integration).where(
                Integration.organization_id == org_id,
                Integration.provider == IntegrationProvider.GMAIL,
            )
        )).first()
    assert integ is not None
    assert integ.access_token, "the refreshed access token was not persisted"
    assert token_cipher.is_encrypted(integ.access_token), (
        "the access token is stored in PLAINTEXT"
    )
    assert token_cipher.is_encrypted(integ.refresh_token)
    assert not str(integ.access_token).startswith("ya29."), "plaintext token at rest"
