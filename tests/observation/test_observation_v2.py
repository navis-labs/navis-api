"""Observation V2: capture, lineage, idempotency, webhooks, timeline, replay."""
from __future__ import annotations

from navis.observation.inbox import process_once
from navis.observation.models import Outcome, OutcomeType, compute_outcome_id
from navis.observation.service import ObservationService
from navis.observation.timeline import reconstruct
from navis.observation.webhooks.calendar import parse_calendar
from navis.observation.webhooks.gmail import parse_gmail
from navis.platform.events import EventType
from tests.observation.fakes import (
    DECISION,
    EXECUTION,
    ORG_A,
    ORG_B,
    FakeOutcomeRepo,
    FakePublisher,
    action_event,
)


def _svc(repo=None, pub=None):
    return ObservationService(repo=repo or FakeOutcomeRepo(), publisher=pub or FakePublisher())


# ---- capture from execution events (loop closed) ----

async def test_records_outcome_from_action_executed_with_full_lineage():
    repo, pub = FakeOutcomeRepo(), FakePublisher()
    svc = ObservationService(repo=repo, publisher=pub)
    event = action_event(EventType.ACTION_EXECUTED)

    outcome = await svc.record_from_execution(event, OutcomeType.ACTION_EXECUTED, "evt-1")

    assert outcome.organization_id == ORG_A
    assert outcome.execution_id == EXECUTION and outcome.decision_id == DECISION
    assert outcome.reasoning_signal_ids == ["s1", "s2"]
    assert outcome.reasoning_version == 1 and outcome.state_watermark == "wm-9"
    assert outcome.provider_reference == {"provider": "gmail", "external_id": "m-1"}
    assert outcome.outcome_type == OutcomeType.ACTION_EXECUTED.value
    # OUTCOME_OBSERVED carries the lineage Learning needs (not a subject_ref prefix).
    emitted = pub.events[0]
    assert emitted.type == EventType.OUTCOME_OBSERVED
    assert emitted.payload["reasoning_signal_ids"] == ["s1", "s2"]


async def test_action_failed_is_captured():
    svc = _svc()
    out = await svc.record_from_execution(
        action_event(EventType.ACTION_FAILED, error="provider down"),
        OutcomeType.ACTION_FAILED, "evt-fail",
    )
    assert out.outcome_type == OutcomeType.ACTION_FAILED.value
    assert out.payload["error"] == "provider down"


# ---- idempotency: at-least-once delivery collapses to one outcome ----

async def test_duplicate_delivery_collapses_to_one_outcome():
    repo = FakeOutcomeRepo()
    svc = ObservationService(repo=repo, publisher=FakePublisher())
    e = action_event(EventType.ACTION_EXECUTED, event_id="evt-x")

    first = await svc.record_from_execution(e, OutcomeType.ACTION_EXECUTED, "evt-x")
    second = await svc.record_from_execution(e, OutcomeType.ACTION_EXECUTED, "evt-x")  # redelivery
    assert first is not None and second is None  # deterministic id -> ON CONFLICT
    assert len(repo.rows) == 1


async def test_inbox_dedupes_consumer_processing():
    repo = FakeOutcomeRepo()
    calls = []

    async def handler():
        calls.append(1)

    assert await process_once(repo, "observation", "evt-1", handler) is True
    assert await process_once(repo, "observation", "evt-1", handler) is False  # duplicate
    assert len(calls) == 1


def test_outcome_id_is_deterministic():
    a = compute_outcome_id("evt-1", "action_executed")
    b = compute_outcome_id("evt-1", "action_executed")
    c = compute_outcome_id("evt-1", "reply_received")
    assert a == b and a != c


# ---- provider webhooks (reply / meeting) attach with inherited lineage ----

async def test_calendar_webhook_attaches_with_inherited_lineage():
    repo = FakeOutcomeRepo()
    svc = ObservationService(repo=repo, publisher=FakePublisher())
    # The scheduled meeting's outcome (provider_reference = calendar event id).
    await svc.record_from_execution(
        action_event(EventType.ACTION_EXECUTED, event_id="exec-1",
                     provider_ref={"provider": "calendar", "external_id": "cal-77"}),
        OutcomeType.ACTION_EXECUTED, "exec-1",
    )
    webhook = parse_calendar({"event_id": "cal-77", "status": "accepted"})
    out = await svc.record_from_webhook(webhook)

    assert out.outcome_type == OutcomeType.MEETING_ACCEPTED.value
    # Lineage inherited from the original action — traces to decision + signals.
    assert out.decision_id == DECISION
    assert out.reasoning_signal_ids == ["s1", "s2"]


async def test_webhook_unknown_reference_is_ignored():
    svc = _svc()
    webhook = parse_calendar({"event_id": "nope", "status": "accepted"})
    assert await svc.record_from_webhook(webhook) is None


def test_gmail_parser_maps_reply():
    we = parse_gmail({"in_reply_to": "m-1", "from": "a@x.com", "snippet": "ok"})
    assert we.provider == "gmail" and we.external_id == "m-1"
    assert we.outcome_type == OutcomeType.REPLY_RECEIVED


# ---- timeline reconstruction ----

async def test_timeline_reconstruction_orders_by_occurred_at():
    from datetime import datetime, timezone

    repo = FakeOutcomeRepo()
    svc = ObservationService(repo=repo, publisher=FakePublisher(),
                             clock=lambda: datetime(2026, 6, 1, tzinfo=timezone.utc))
    base = datetime(2026, 6, 1, tzinfo=timezone.utc)
    pr = {"provider": "calendar", "external_id": "cal-9"}
    await svc.record(org_id=ORG_A, source_event_id="e1", outcome_type=OutcomeType.ACTION_EXECUTED,
                     occurred_at=base, execution_id=EXECUTION, decision_id=DECISION,
                     reasoning_signal_ids=["s"], provider_reference=pr)
    await svc.record(org_id=ORG_A, source_event_id="e2", outcome_type=OutcomeType.MEETING_ACCEPTED,
                     occurred_at=base.replace(day=2), execution_id=EXECUTION, decision_id=DECISION)
    await svc.record(org_id=ORG_A, source_event_id="e3", outcome_type=OutcomeType.MEETING_COMPLETED,
                     occurred_at=base.replace(day=3), execution_id=EXECUTION, decision_id=DECISION)

    timeline = await reconstruct(repo, ORG_A, execution_id=EXECUTION)
    assert [t.outcome_type for t in timeline] == [
        "action_executed", "meeting_accepted", "meeting_completed",
    ]


# ---- replay (re-emit from the immutable log) ----

async def test_replay_reemits_each_outcome():
    repo, pub = FakeOutcomeRepo(), FakePublisher()
    svc = ObservationService(repo=repo, publisher=pub)
    await svc.record_from_execution(action_event(EventType.ACTION_EXECUTED, event_id="e1"),
                                    OutcomeType.ACTION_EXECUTED, "e1")
    pub.events.clear()
    rows = await repo.for_replay(ORG_A)
    for r in rows:
        await svc.republish(r)
    assert len(pub.events) == len(rows) == 1
    assert pub.events[0].type == EventType.OUTCOME_OBSERVED


# ---- multi-tenancy ----

async def test_multi_tenant_timeline_isolation():
    repo = FakeOutcomeRepo()
    svc = ObservationService(repo=repo, publisher=FakePublisher())
    await svc.record_from_execution(action_event(EventType.ACTION_EXECUTED, org_id=ORG_A, event_id="a"),
                                    OutcomeType.ACTION_EXECUTED, "a")
    await svc.record_from_execution(action_event(EventType.ACTION_EXECUTED, org_id=ORG_B, event_id="b"),
                                    OutcomeType.ACTION_EXECUTED, "b")
    a = await reconstruct(repo, ORG_A)
    b = await reconstruct(repo, ORG_B)
    assert len(a) == 1 and len(b) == 1
    assert a[0].outcome_id != b[0].outcome_id


# ---- law: Observation stores no success/confidence/risk ----

def test_outcome_model_stores_no_intelligence():
    cols = set(Outcome.__table__.columns.keys())
    forbidden = {"success", "confidence", "risk", "priority", "score", "recommendation", "prediction"}
    assert cols & forbidden == set()


# ---- wiring: subscribers registered (loop closed) ----

def test_observation_subscribers_registered():
    import navis.observation.subscribers  # noqa: F401  registers handlers
    from navis.platform.events import event_bus

    for et in (EventType.ACTION_EXECUTED, EventType.ACTION_FAILED):
        names = [h.__name__ for h in event_bus._handlers.get(et, [])]
        assert any("action" in n for n in names), f"no observation handler for {et}"
