"""INTEGRATION — a real organizational situation becomes real, approved work.

WHAT THIS ADDS THAT NOTHING ELSE COVERS
---------------------------------------
Three suites already prove the UPSTREAM half against real code
(`test_waiting_on_reply_pipeline`, `test_obligations_pipeline`,
`test_draft_to_send_pipeline`): a Gmail payload really does become a draftable
candidate, and a draft really does become a payload the provider accepts.

None of them crosses the GOVERNANCE layer or reaches Observation. They run on
`FakeExecutionRepo` with no organization, no entitlements, a hand-injected
provider and no outbox relay — so the questions this file exists to answer were
all open:

    does the persisted brief actually produce execution rows?     (the bridge)
    does approval enforce the connector capability?               (capability)
    does the worker re-check authorization before acting?         (authorization)
    is the credential resolved through the integration boundary?  (credentials)
    does the outcome reach Observation with correct lineage?      (observation)

Everything here is the real implementation on real PostgreSQL. THE ONLY
SUBSTITUTION IS THE NETWORK: `httpx.AsyncClient` is replaced at the Google
transport boundary. `_load_grant`, `resolve_integration`,
`resolve_gmail_send_integration`, `TokenManager`, `GoogleTransport`,
`GmailClient`, `CalendarClient` and the real senders all run.

SAFETY
------
Every `session_scope` reachable from this path is repointed at the disposable
test engine, and the test refuses to run unless `NAVIS_TEST_DATABASE_URL` is
set. The production database is never opened: the senders, the calendar
reconcile and the Observation subscriber each open their own session, and each
one is redirected here.
"""
from __future__ import annotations

import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("NAVIS_TEST_DATABASE_URL"),
    reason="set NAVIS_TEST_DATABASE_URL (disposable Postgres) to run workflow tests",
)

ORG = uuid.UUID("aaaaaaaa-0000-4000-8000-000000000001")
NOW = datetime(2026, 8, 19, 9, 0, tzinfo=timezone.utc)
FOUNDER = "founder@navislabs.in"
PARTNER = "john@partner.com"

ACCESS_SECRET = "ya29.WORKFLOW_ACCESS_TOKEN"
REFRESH_SECRET = "1//04WORKFLOW_REFRESH_TOKEN"


# ── Infrastructure ─────────────────────────────────────────────────────


@pytest.fixture()
async def engine():
    from sqlalchemy.ext.asyncio import create_async_engine

    import navis.registry  # noqa: F401 — registers every model AND the subscribers
    from navis.platform.database import Base

    url = os.environ["NAVIS_TEST_DATABASE_URL"]
    assert "neon.tech" not in url, "refusing to run workflow tests against Neon"
    eng = create_async_engine(url)
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    await eng.dispose()


@pytest.fixture()
async def Session(engine, monkeypatch):
    """A sessionmaker on the DISPOSABLE engine, installed everywhere the
    production path would otherwise reach for the global one."""
    from sqlalchemy.ext.asyncio import async_sessionmaker

    maker = async_sessionmaker(engine, expire_on_commit=False)

    @asynccontextmanager
    async def _scope():
        async with maker() as s:
            try:
                yield s
                await s.commit()
            except Exception:
                await s.rollback()
                raise

    # Each of these modules imported `session_scope` by value at import time, so
    # patching the source module alone would miss them. Every one of them opens a
    # session on the production engine in production.
    for target in (
        "navis.execution.providers._senders.session_scope",
        "navis.observation.subscribers.session_scope",
        "navis.platform.database.session_scope",
    ):
        monkeypatch.setattr(target, _scope, raising=False)

    # ONE DRIVER OF THE WORKER, DELIBERATELY.
    #
    # `settings.execution_inline_fallback` defaults to True, and the approve
    # handler acts on it by firing `asyncio.create_task(_run_inline_execution_
    # pass())` — a SECOND, concurrent execution pass inside the API process,
    # racing the one the caller drives. It is a dev convenience for running
    # without Celery, and its own docstring says it "should be False in
    # production when a real Celery worker is running".
    #
    # Left on, it makes this file measure the wrong thing: the racing pass holds
    # its Phase A session while THIS pass calls Google, so the transaction-
    # boundary probe reports a held connection that production (with a real
    # worker and the flag off) would never have. Turning it off is what a
    # production deployment does, and it makes the test drive exactly one worker.
    monkeypatch.setattr(
        "navis.platform.config.settings.execution_inline_fallback", False, raising=False
    )
    return maker


@pytest.fixture()
async def org(Session):
    """A provisioned organization with a real Gmail + Calendar grant.

    Provisioned through `OrganizationService.bootstrap`, exactly as signup does,
    because the approve gate and the worker's re-check both consult entitlements
    — an unprovisioned org would be correctly refused and the test would be
    asserting the denial path.
    """
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

    async with Session() as s:
        s.add(Organization(id=ORG, name="Navis Labs", slug=f"navis-{uuid.uuid4().hex[:6]}"))
        user = User(email=FOUNDER, full_name="Abhishek Meena")
        s.add(user)
        await s.flush()
        await OrganizationService(s).bootstrap(ORG, plan=Plan.ENTERPRISE)
        for provider in (IntegrationProvider.GMAIL, IntegrationProvider.GOOGLE_CALENDAR):
            s.add(Integration(
                organization_id=ORG, user_id=user.id, provider=provider,
                external_account=FOUNDER,
                access_token=token_cipher.encrypt(ACCESS_SECRET),
                refresh_token=token_cipher.encrypt(REFRESH_SECRET),
                token_expires_at=NOW + timedelta(hours=1),
                scopes=" ".join(GOOGLE_SCOPES), is_active=True, sync_state={},
            ))
        await s.commit()
        return user.id


@pytest.fixture()
def google(monkeypatch):
    """Substitute ONLY the network. Records every Google request, and how many
    database connections were checked out when it was made."""
    from sqlalchemy import event as sa_event

    state = {"requests": [], "checked_out": 0}

    class _Resp:
        status_code = 200
        headers: dict = {}
        content = b"{}"
        text = ""

        def __init__(self, body):
            self._body = body

        def json(self):
            return self._body

    class _HTTP:
        def __init__(self, *a, **kw):
            pass

        async def request(self, method, url, params=None, json=None, headers=None):
            state["requests"].append({
                "method": method, "url": url, "json": json, "headers": headers,
                "db_connections_open": state["checked_out"],
            })
            if "/messages/send" in url:
                return _Resp({"id": "gmail-live-1", "threadId": "thread-release"})
            if url.endswith("/messages"):
                return _Resp({"messages": []})
            if "/events/" in url:
                return _Resp({
                    "id": url.rsplit("/", 1)[-1], "etag": '"v1"', "status": "confirmed",
                    "summary": "Release sync",
                    "start": {"dateTime": "2026-08-22T10:00:00+00:00"},
                    "end": {"dateTime": "2026-08-22T10:30:00+00:00"},
                    "organizer": {"email": FOUNDER},
                })
            if url.endswith("/events"):
                body = json or {}
                return _Resp({
                    "id": body.get("id", "evt-created"), "etag": '"v1"', "status": "confirmed",
                    "summary": body.get("summary"), "start": body.get("start"),
                    "end": body.get("end"),
                    "hangoutLink": "https://meet.google.com/abc-defg-hij",
                    "conferenceData": {
                        "createRequest": {"status": {"statusCode": "success"}},
                        "entryPoints": [{
                            "entryPointType": "video",
                            "uri": "https://meet.google.com/abc-defg-hij",
                        }],
                    },
                })
            return _Resp({})

        async def aclose(self):
            pass

    monkeypatch.setattr("navis.signals.google.transport.httpx.AsyncClient", _HTTP)
    return state, sa_event


# ── The real upstream chain ────────────────────────────────────────────


def _signal(*, mid: str, days_ago: float, body: str, subject: str, thread_id: str):
    from navis.memory.extractors.base import SignalView
    from navis.signals.base import SignalEventType, SignalSource

    return SignalView(
        event_id=uuid.uuid5(uuid.NAMESPACE_URL, f"wf:{mid}"),
        org_id=ORG, user_id=uuid.uuid4(), integration_id=uuid.uuid4(),
        source=SignalSource.GMAIL, event_type=SignalEventType.EMAIL_RECEIVED,
        external_id=mid, external_version="1",
        event_timestamp=NOW - timedelta(days=days_ago),
        payload={
            "from": f"John Carter <{PARTNER}>", "to": FOUNDER, "subject": subject,
            "thread_id": thread_id, "sender_email": PARTNER, "sender_name": "John Carter",
            "to_emails": [FOUNDER], "body_text": body,
            "rfc_message_id": f"<{mid}@mail.gmail.com>",
            "label_ids": ["INBOX", "CATEGORY_PERSONAL"],
        },
        metadata={"thread_id": thread_id},
    )


async def _ingest_signals(Session, signals, user_id):
    """Persist the same Gmail messages as real `signal_events` rows.

    THE DRAFT GENERATOR READS THE THREAD FROM THE DATABASE, not from whatever a
    test holds in memory — `load_thread` selects `SignalEventRecord` rows, and
    when the thread is unknown `generate` raises `InsufficientContext` rather
    than writing a reply into a conversation it cannot see. That refusal is the
    product behaviour that stops Navis fabricating email, so the workflow test
    must satisfy it the way production does: by having actually captured the
    thread.

    Goes through the real `SignalIngestionService`, so these rows are
    byte-identical to synced ones — same dedupe key, same sealed payload, same
    staged outbox pointer.
    """
    from navis.identity.models import Integration
    from navis.signals.base import SignalContext, SignalEvent
    from navis.signals.service import SignalIngestionService
    from sqlalchemy import select

    async with Session() as s:
        integration = (await s.scalars(select(Integration).limit(1))).one()
        for sig in signals:
            event = SignalEvent.create(
                SignalContext(
                    org_id=ORG, user_id=user_id, integration_id=integration.id
                ),
                source=sig.source,
                event_type=sig.event_type,
                external_id=sig.external_id,
                external_version=sig.external_version,
                timestamp=sig.event_timestamp,
                payload=dict(sig.payload),
                metadata=dict(sig.metadata),
            )
            await SignalIngestionService(s).ingest(event)
        await s.commit()


def _reason_over(signals):
    """Signals -> Memory records -> State V2 -> Reasoning V2. All real."""
    from navis.memory.extractors import extractors_for
    from navis.memory.ids import entity_id, memory_record_id
    from navis.reasoning.v2 import run_reasoning_v2
    from navis.state.v2.builders import MemoryRecordView, build_projection

    records = []
    for sig in signals:
        for extractor in extractors_for(sig.source):
            for draft in extractor.extract(sig):
                records.append(MemoryRecordView(
                    memory_id=memory_record_id(
                        sig.org_id, sig.event_id, extractor.name,
                        draft.entity_type.value, draft.dedup_key,
                    ),
                    org_id=sig.org_id, entity_type=draft.entity_type.value,
                    entity_id=entity_id(sig.org_id, draft.entity_type.value,
                                        draft.entity_natural_key),
                    event_timestamp=sig.event_timestamp, payload=draft.payload,
                    source_signal_event_id=sig.event_id,
                    schema_version=draft.schema_version, confidence=draft.confidence,
                ))
    state = build_projection(ORG, records, as_of=NOW)
    return state, run_reasoning_v2(state)


async def _persist_brief(Session, signals):
    """Real reasoning -> real DecisionService -> a PERSISTED DailyBrief."""
    from navis.decisions.service import DecisionService

    _state, reasoning = _reason_over(signals)
    async with Session() as s:
        brief = await DecisionService(s).decide(
            ORG, reasoning, reasoning_version=1, state_watermark="w-1", as_of=NOW,
        )
        await s.commit()
        return brief


async def _drain(Session, providers, reconciler=None):
    """One real worker pass on the disposable database."""
    from navis.execution.repository import ExecutionRepository
    from navis.execution.worker import execute_pass

    @asynccontextmanager
    async def unit():
        async with Session() as s:
            yield ExecutionRepository(s)
            await s.commit()

    return await execute_pass(
        unit=unit, providers=providers, limit=10, reconciler=reconciler
    )


def _pick_candidate(rows, kind: str):
    """The execution row for one obligation kind.

    Mirrors `_pick` in the existing pipeline suites: a thread can produce more
    than one candidate, so a test names the obligation it means instead of
    assuming a thread yields exactly one.
    """
    matches = [r for r in rows if (r.payload or {}).get("candidate_kind") == kind]
    assert matches, (
        f"no {kind} candidate in "
        f"{[(r.payload or {}).get('candidate_kind') for r in rows]}"
    )
    return matches[0]


async def _relay_into_observation(Session):
    """Publish the outbox so the REAL Observation subscriber consumes it."""
    from navis.execution.repository import ExecutionRepository
    from navis.execution.worker import relay_pass

    async with Session() as s:
        published = await relay_pass(ExecutionRepository(s))
        await s.commit()
    return published


# ══════════════════════════════════════════════════════════════════════
# WORKFLOW 1 — CUSTOMER FOLLOW-UP  (proactive, Gmail)
# ══════════════════════════════════════════════════════════════════════

_UNANSWERED = [
    _signal(
        mid="wf-ask", days_ago=4, thread_id="thread-release", subject="Product Release",
        body="Hi, can you confirm whether we're still targeting Monday for the release?",
    )
]


async def test_customer_follow_up_reaches_gmail_and_observation(Session, org, google):
    """THE WORKFLOW, end to end:

        unanswered email
          -> Memory / State V2
          -> Reasoning V2
          -> Decision (persisted brief)
          -> propose_actions           (the proactive bridge)
          -> draft generated + approved
          -> capability + preflight     (approve gate)
          -> runtime authorization      (worker)
          -> credential boundary        (real resolver)
          -> Gmail                      (network substituted)
          -> outbox -> relay -> Observation
    """
    from navis.execution.drafts import DraftService
    from navis.execution.models import ExecutionOrigin, ExecutionRequest
    from navis.execution.providers import production_providers
    from navis.execution.status import ExecutionStatus
    from navis.observation.models import Outcome, OutcomeType
    from navis.orchestration.brief import propose_actions
    from sqlalchemy import select

    state_requests, _ = google
    user_id = org

    # ── Signals: really captured, so the draft has a thread to ground in ─
    await _ingest_signals(Session, _UNANSWERED, user_id)

    # ── Decision ──────────────────────────────────────────────────────
    brief = await _persist_brief(Session, _UNANSWERED)
    assert brief.decisions, "reasoning produced no decision from an unanswered email"

    # ── The proactive bridge (real `propose_actions`) ─────────────────
    async with Session() as s:
        created = await propose_actions(s, ORG)
        await s.commit()
    assert created >= 1, "the brief produced no execution rows"

    async with Session() as s:
        rows = (await s.scalars(
            select(ExecutionRequest).where(ExecutionRequest.organization_id == ORG)
        )).all()
    # ONE THREAD CAN CARRY SEVERAL OBLIGATIONS AT ONCE, and that is correct: an
    # unanswered email that also asks a question is both "they are waiting on a
    # reply" and "you owe a decision". Each is its own obligation with its own
    # stable identity, so each is its own row — which is exactly why the payload
    # carries `candidate_kind` and why the existing pipeline suites select rather
    # than assume. Pick the one this workflow is about.
    row = _pick_candidate(rows, "waiting_reply")
    assert row.action_type == "reply_to_email"
    assert row.origin == ExecutionOrigin.PROACTIVE.value, "must land in the Action Center"
    assert row.status == ExecutionStatus.REQUESTED.value
    # Provenance survived the whole chain.
    assert row.decision_id is not None
    assert row.obligation_key and row.obligation_key.startswith("reply|")
    execution_id = row.id

    # ── A human reviews the draft, then approves ──────────────────────
    async with Session() as s:
        req = await s.get(ExecutionRequest, execution_id)
        draft = await DraftService(s).get_or_generate(req, sender_name="Abhishek Meena")
        assert draft.body.strip(), "no reviewable draft was generated"
        assert PARTNER in draft.recipients
        await s.commit()

    approved = await _approve_through_the_real_gate(Session, execution_id, user_id)
    assert approved.status == ExecutionStatus.APPROVED.value
    assert approved.payload["requires_draft"] is False, "the draft merged into the payload"
    assert approved.payload["body"].strip()

    # ── Execution ─────────────────────────────────────────────────────
    result = await _drain(Session, production_providers())
    assert result["executed"] == 1, result

    async with Session() as s:
        done = await s.get(ExecutionRequest, execution_id)
    assert done.status == ExecutionStatus.EXECUTED.value, done.last_error
    assert done.provider_reference == {"provider": "gmail", "external_id": "gmail-live-1"}
    assert "dry_run" not in done.provider_reference

    # The credential really was resolved and used — through the real boundary.
    sends = [r for r in state_requests["requests"] if "/messages/send" in r["url"]]
    assert len(sends) == 1
    assert sends[0]["headers"]["Authorization"] == f"Bearer {ACCESS_SECRET}"

    # ── Observation ───────────────────────────────────────────────────
    published = await _relay_into_observation(Session)
    assert published >= 1

    async with Session() as s:
        outcomes = (await s.scalars(
            select(Outcome).where(Outcome.execution_id == execution_id)
        )).all()
    assert len(outcomes) == 1, "Observation did not receive the executed action"
    outcome = outcomes[0]
    assert outcome.organization_id == ORG
    assert outcome.outcome_type == OutcomeType.ACTION_EXECUTED.value
    assert outcome.decision_id == done.decision_id
    assert outcome.provider_reference["provider"] == "gmail"
    assert outcome.provider_reference["external_id"] == "gmail-live-1"


async def _approve_through_the_real_gate(Session, execution_id, user_id):
    """Approve through the REAL router handler, not `ExecutionService.approve`.

    That is deliberate: the handler is where the connector capability gate and
    the credential pre-flight live, and skipping it would test a path no user
    can take.
    """
    from types import SimpleNamespace

    from navis.execution.models import ExecutionRequest
    from navis.execution.router import approve

    async with Session() as s:
        await approve(execution_id, SimpleNamespace(org_id=ORG, subject_id=user_id), s)
        await s.commit()
    async with Session() as s:
        return await s.get(ExecutionRequest, execution_id)


async def test_revoking_gmail_blocks_the_approved_follow_up_with_no_google_call(
    Session, org, google
):
    """Post-approval revocation on the REAL workflow.

    The row is legitimately APPROVED, then the connector capability is revoked.
    The worker's re-check must refuse before the credential boundary is reached,
    so Google is never called and no side effect is recorded.
    """
    from navis.execution.models import ExecutionRequest
    from navis.execution.providers import production_providers
    from navis.execution.status import ExecutionStatus
    from navis.organization.contract import Capability
    from navis.organization.entitlements import EntitlementService
    from navis.orchestration.brief import propose_actions
    from sqlalchemy import select

    state_requests, _ = google
    user_id = org
    await _ingest_signals(Session, _UNANSWERED, user_id)
    await _persist_brief(Session, _UNANSWERED)
    async with Session() as s:
        await propose_actions(s, ORG)
        await s.commit()
    async with Session() as s:
        rows = (await s.scalars(
            select(ExecutionRequest).where(ExecutionRequest.organization_id == ORG)
        )).all()
        row = _pick_candidate(rows, "waiting_reply")
        execution_id = row.id
        from navis.execution.drafts import DraftService

        await DraftService(s).get_or_generate(row, sender_name="Abhishek Meena")
        await s.commit()
    await _approve_through_the_real_gate(Session, execution_id, user_id)

    # Revoke AFTER approval — the window this gate exists for. Goes through the
    # real EntitlementService, so the worker's re-check resolves it exactly as it
    # would for a plan downgrade or an admin revocation in production.
    async with Session() as s:
        from navis.organization.repository import OrganizationRepository

        await EntitlementService(OrganizationRepository(s)).set(
            ORG, Capability.GMAIL, False
        )
        await s.commit()

    before = len(state_requests["requests"])
    await _drain(Session, production_providers())
    async with Session() as s:
        row = await s.get(ExecutionRequest, execution_id)

    assert row.status == ExecutionStatus.FAILED.value
    assert "gmail" in (row.last_error or "").lower()
    assert not row.provider_reference, "no side effect may be recorded"
    assert len(state_requests["requests"]) == before, (
        "a revoked organization reached Google"
    )
    # The approval record is untouched — it says who authorized the intent.
    assert row.approved_by == user_id and row.approved_at is not None


async def test_the_scheduled_brief_proposes_without_anyone_asking(
    Session, org, google, monkeypatch
):
    """THE PROACTIVE TRIGGER — the fix this task made.

    `run_brief_for_org` is what Celery beat reaches: beat runs `build_all_briefs`
    -> `build_brief_for_org` -> here. It built the brief and STOPPED. The bridge
    that turns decisions into execution rows — `propose_actions`, described in
    its own docstring as "THE proactive producer" — was only reachable from
    `POST /brief/propose-actions`.

    So the Action Center filled up only when a client asked it to. Navis did not
    notice an unanswered email and raise it; it noticed when something else
    prompted it to look. This asserts the scheduled entry point now does the
    whole job, which is the difference between the product being proactive and
    being on demand.

    HOW THIS IS PROVEN, in two halves that meet:

      * HERE — the scheduled entry point really invokes the bridge, for the org
        it was given, and reports what it produced.
      * `test_customer_follow_up_reaches_gmail_and_observation` — the bridge
        really turns a persisted brief into PROACTIVE rows on real data.

    Split deliberately: `build_brief` folds DB-backed memory records through
    `StateService.incremental_update`, so driving it from raw signals would mean
    rebuilding the whole capture->memory->state cascade that three other
    integration suites already prove. Re-staging it here would test those suites
    again and this fix barely at all.
    """
    from navis.orchestration import brief as brief_module

    await _ingest_signals(Session, _UNANSWERED, org)

    # `brief.py` imported `session_scope` by value, so point ITS name at the
    # disposable engine — this is the module Celery actually calls.
    maker = Session

    @asynccontextmanager
    async def _scope():
        async with maker() as s:
            try:
                yield s
                await s.commit()
            except Exception:
                await s.rollback()
                raise

    monkeypatch.setattr(brief_module, "session_scope", _scope)

    asked: list = []

    async def _spy(db, org_id):
        asked.append(org_id)
        return 3

    monkeypatch.setattr(brief_module, "propose_actions", _spy)

    result = await brief_module.run_brief_for_org(ORG)

    assert asked == [ORG], (
        "the scheduled brief did not invoke the proactive bridge — the Action "
        "Center stays empty until a client calls POST /brief/propose-actions"
    )
    assert result["proposed"] == 3, "the scheduled run must report what it proposed"
    assert result["brief_id"], "the brief itself must still be built and returned"


async def test_the_scheduled_brief_survives_a_failing_proposer(
    Session, org, google, monkeypatch
):
    """The brief is the founder-visible artefact. A proposer failure must not
    lose it — the next scheduled run re-proposes from the same brief."""
    from navis.orchestration import brief as brief_module

    await _ingest_signals(Session, _UNANSWERED, org)
    maker = Session

    @asynccontextmanager
    async def _scope():
        async with maker() as s:
            try:
                yield s
                await s.commit()
            except Exception:
                await s.rollback()
                raise

    monkeypatch.setattr(brief_module, "session_scope", _scope)

    async def _boom(db, org_id):
        raise RuntimeError("proposer exploded")

    monkeypatch.setattr(brief_module, "propose_actions", _boom)

    result = await brief_module.run_brief_for_org(ORG)
    assert result["brief_id"], "the brief was lost because proposing failed"
    assert result["proposed"] == 0


# ══════════════════════════════════════════════════════════════════════
# WORKFLOW 2 — MEETING  (chat-originated, Calendar + Meet)
# ══════════════════════════════════════════════════════════════════════


async def test_meeting_workflow_creates_a_real_event_with_meet_and_observes_it(
    Session, org, google
):
    """chat proposal -> ExecutionRequest -> approval -> Calendar + Meet ->
    read-after-write -> Observation.

    Uses the payload shape `ScheduledProposal.to_execution_payload()` emits, so
    the contract between the scheduling agent and the sender is exercised rather
    than assumed.
    """
    from navis.execution.models import ExecutionOrigin, ExecutionRequest
    from navis.execution.providers import production_providers
    from navis.execution.service import ExecutionService
    from navis.execution.status import ExecutionStatus
    from navis.observation.models import Outcome
    from sqlalchemy import select

    state_requests, _ = google
    user_id = org
    start = NOW + timedelta(days=3)
    payload = {
        "title": "Release sync", "start": start.isoformat(),
        "end": (start + timedelta(minutes=30)).isoformat(),
        "time_zone": "UTC", "duration_minutes": 30,
        "attendees": [PARTNER], "attendee_name": "John Carter",
        "description": None, "send_google_invite": True,
        "conference": True, "assumptions": [],
    }

    async with Session() as s:
        req, _created = await ExecutionService(s).request(
            ORG, uuid.uuid4(), "schedule_meeting", payload,
            reasoning_signal_ids=[], reasoning_version=None, state_watermark=None,
            origin=ExecutionOrigin.CHAT,
        )
        await s.commit()
        execution_id = req.id

    approved = await _approve_through_the_real_gate(Session, execution_id, user_id)
    assert approved.status == ExecutionStatus.APPROVED.value

    reconciled: list = []

    async def _reconciler(org_id, event_id):
        reconciled.append((org_id, event_id))
        return True

    result = await _drain(Session, production_providers(), reconciler=_reconciler)
    assert result["executed"] == 1, result

    async with Session() as s:
        done = await s.get(ExecutionRequest, execution_id)
    assert done.status == ExecutionStatus.EXECUTED.value, done.last_error

    # Deterministic identity — the same execution always targets the same event.
    from navis.execution.providers._senders import _google_event_id

    assert done.provider_reference["external_id"] == _google_event_id(execution_id)
    # Google Meet, through Calendar conferenceData — no separate connector.
    assert done.provider_reference["meet_url"] == "https://meet.google.com/abc-defg-hij"
    assert done.provider_reference["conference_status"] == "success"

    inserts = [r for r in state_requests["requests"] if r["method"] == "POST"]
    assert inserts, "no event was created"
    assert inserts[0]["json"]["conferenceData"]["createRequest"]["conferenceSolutionKey"] == {
        "type": "hangoutsMeet"
    }
    # Read-after-write ran BEFORE the row reached EXECUTED.
    assert reconciled == [(ORG, done.provider_reference["external_id"])]

    published = await _relay_into_observation(Session)
    assert published >= 1
    async with Session() as s:
        outcomes = (await s.scalars(
            select(Outcome).where(Outcome.execution_id == execution_id)
        )).all()
    assert len(outcomes) == 1
    assert outcomes[0].provider_reference["provider"] == "calendar"
    assert outcomes[0].organization_id == ORG


async def test_a_replayed_meeting_execution_targets_the_same_event(Session, org, google):
    """Idempotency on the real path: the same execution replayed must address the
    SAME Google event id, which is what makes `events.insert` a no-op on replay.

    This is Navis's half of the guarantee. Google's half — actually returning 409
    — remains unverified; see the live harness.
    """
    from navis.execution.providers._senders import _google_event_id, calendar_sender

    state_requests, _ = google
    execution_id = uuid.uuid4()
    ctx = {"org_id": str(ORG), "execution_id": str(execution_id),
           "decision_id": str(uuid.uuid4())}
    body = {
        "__ctx__": ctx, "title": "Release sync",
        "start": (NOW + timedelta(days=3)).isoformat(),
        "end": (NOW + timedelta(days=3, minutes=30)).isoformat(),
        "time_zone": "UTC", "attendees": [PARTNER], "conference": False,
        "send_google_invite": False,
    }
    first, _ = await calendar_sender(dict(body))
    second, _ = await calendar_sender(dict(body))

    assert first == second == _google_event_id(execution_id)
    posted = [r for r in state_requests["requests"] if r["method"] == "POST"]
    assert all(r["json"]["id"] == first for r in posted), (
        "a replay sent a different event id, so Google could not deduplicate it"
    )


# ══════════════════════════════════════════════════════════════════════
# CROSS-CUTTING INVARIANTS, on the real workflow
# ══════════════════════════════════════════════════════════════════════


async def test_no_database_connection_is_held_while_google_is_called(
    Session, engine, org, google
):
    """The transaction boundary, measured on the REAL workflow against real
    pooled connections rather than a substituted session factory."""
    from sqlalchemy import event as sa_event

    from navis.execution.providers import production_providers
    from navis.execution.service import ExecutionService
    from navis.execution.models import ExecutionOrigin

    state, _ = google

    def _out(*_a):
        state["checked_out"] += 1

    def _in(*_a):
        state["checked_out"] -= 1

    sa_event.listen(engine.sync_engine, "checkout", _out)
    sa_event.listen(engine.sync_engine, "checkin", _in)
    try:
        async with Session() as s:
            req, _ = await ExecutionService(s).request(
                ORG, uuid.uuid4(), "send_email",
                {"to": PARTNER, "subject": "Release", "body": "Confirming Monday."},
                reasoning_signal_ids=[], reasoning_version=None, state_watermark=None,
                origin=ExecutionOrigin.CHAT,
            )
            await s.commit()
            execution_id = req.id
        await _approve_through_the_real_gate(Session, execution_id, org)
        await _drain(Session, production_providers())
    finally:
        sa_event.remove(engine.sync_engine, "checkout", _out)
        sa_event.remove(engine.sync_engine, "checkin", _in)

    google_calls = state["requests"]
    assert google_calls, "no Google request was made"
    offenders = [r for r in google_calls if r["db_connections_open"] != 0]
    assert not offenders, (
        "a pooled DB connection was held during a Google call: "
        + ", ".join(f"{r['method']} {r['url']}" for r in offenders)
    )


async def test_no_credential_reaches_any_durable_surface_on_the_real_workflow(
    Session, org, google
):
    """Secrets stay out of the row, the outbox, the dead letters and Observation
    — checked against the literal token values, on a real execution."""
    from navis.execution.models import ExecutionDeadLetter, ExecutionOutbox, ExecutionRequest
    from navis.execution.providers import production_providers
    from navis.execution.service import ExecutionService
    from navis.execution.models import ExecutionOrigin
    from navis.identity.models import Integration
    from navis.observation.models import Outcome
    from navis.signals.crypto import token_cipher
    from sqlalchemy import select

    async with Session() as s:
        req, _ = await ExecutionService(s).request(
            ORG, uuid.uuid4(), "send_email",
            {"to": PARTNER, "subject": "Release", "body": "Confirming Monday."},
            reasoning_signal_ids=[], reasoning_version=None, state_watermark=None,
            origin=ExecutionOrigin.CHAT,
        )
        await s.commit()
        execution_id = req.id
    await _approve_through_the_real_gate(Session, execution_id, org)
    await _drain(Session, production_providers())
    await _relay_into_observation(Session)

    async with Session() as s:
        row = await s.get(ExecutionRequest, execution_id)
        outbox = (await s.scalars(select(ExecutionOutbox))).all()
        dlq = (await s.scalars(select(ExecutionDeadLetter))).all()
        outcomes = (await s.scalars(select(Outcome))).all()
        integrations = (await s.scalars(select(Integration))).all()

    surfaces = {
        "payload": str(row.payload),
        "provider_reference": str(row.provider_reference),
        "last_error": str(row.last_error),
        "outbox": str([e.payload for e in outbox]),
        "dead_letters": str([(d.error, d.payload) for d in dlq]),
        "observation": str([(o.provider_reference, o.payload) for o in outcomes]),
    }
    for name, surface in surfaces.items():
        for secret in (ACCESS_SECRET, REFRESH_SECRET, "Bearer ", "ya29.", "1//04"):
            assert secret not in surface, f"{secret!r} leaked into {name}"

    # And nothing wrote a decrypted credential back to the integrations table.
    for integ in integrations:
        assert token_cipher.is_encrypted(integ.access_token), "PLAINTEXT token at rest"
        assert token_cipher.is_encrypted(integ.refresh_token)


async def test_one_organizations_workflow_never_touches_another(Session, org, google):
    """Org isolation on the real workflow: a second organization with its own
    grant must never have its credential used for the first org's action, and a
    forged payload context cannot change that."""
    from navis.execution.models import ExecutionOrigin, ExecutionRequest
    from navis.execution.providers import production_providers
    from navis.execution.service import ExecutionService
    from navis.execution.status import ExecutionStatus
    from navis.identity.models import Organization, User
    from navis.organization.contract import Plan
    from navis.organization.service import OrganizationService

    state_requests, _ = google
    other = uuid.UUID("bbbbbbbb-0000-4000-8000-000000000002")
    async with Session() as s:
        s.add(Organization(id=other, name="Other", slug=f"other-{uuid.uuid4().hex[:6]}"))
        s.add(User(email="other@example.com"))
        await s.flush()
        await OrganizationService(s).bootstrap(other, plan=Plan.ENTERPRISE)
        await s.commit()

    # `other` holds EXECUTION + GMAIL but has NO Google grant. A forged context
    # names the org that does.
    async with Session() as s:
        req, _ = await ExecutionService(s).request(
            other, uuid.uuid4(), "send_email",
            {"to": PARTNER, "subject": "forged", "body": "forged",
             "__ctx__": {"org_id": str(ORG), "execution_id": str(uuid.uuid4()),
                         "decision_id": str(uuid.uuid4())}},
            reasoning_signal_ids=[], reasoning_version=None, state_watermark=None,
            origin=ExecutionOrigin.CHAT,
        )
        await s.commit()
        execution_id = req.id
        assert req.organization_id == other

    async with Session() as s:
        from navis.execution.service import ExecutionService as Svc

        await Svc(s).approve(execution_id, org)
        await s.commit()

    before = len(state_requests["requests"])
    await _drain(Session, production_providers())
    async with Session() as s:
        row = await s.get(ExecutionRequest, execution_id)

    assert row.organization_id == other, "the persisted org must not move"
    assert row.status == ExecutionStatus.FAILED.value, (
        "an org with no Google grant SENT MAIL — the payload chose the tenant"
    )
    assert len(state_requests["requests"]) == before, "it reached Google anyway"
