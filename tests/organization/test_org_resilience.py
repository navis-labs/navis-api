# pyright: reportArgumentType=false
"""ORG-003 plan-change domain event (fake-based; runs in the default suite).

ORG-001 (CHECK constraints) and ORG-002 (bootstrap concurrency) need a live Postgres
and live there in tests/integration/test_organization_postgres.py.
"""
from __future__ import annotations

from navis.organization.billing import BillingService
from navis.organization.contract import Plan
from navis.organization.entitlements import EntitlementService
from navis.platform.events import EventType
from tests.organization.fakes import ACTOR, ORG_A, FakeOrgRepo


class _FakeAudit:
    def __init__(self) -> None:
        self.records: list[tuple] = []

    async def record(self, action, **kw):
        self.records.append((action, kw))


async def test_set_plan_emits_plan_changed_once_with_payload_and_audit():
    repo = FakeOrgRepo()
    ent = EntitlementService(repo)
    audit = _FakeAudit()
    events: list[dict] = []

    async def publisher(**kw):
        events.append(kw)

    billing = BillingService(repo, ent, audit=audit, publisher=publisher)
    await billing.set_plan(ORG_A, Plan.PRO, actor_id=ACTOR, billing_authorized=True)

    # event emitted exactly once, correct payload
    assert len(events) == 1
    e = events[0]
    assert e["org_id"] == ORG_A
    assert e["old_plan"] == "free"
    assert e["new_plan"] == "pro"
    assert e["actor_id"] == ACTOR
    assert e["changed_at"]  # ISO timestamp present
    # audit STILL written alongside the event
    assert len(audit.records) == 1
    assert audit.records[0][0] == "org.plan.pro"

    # no-op re-assert of the same plan: no duplicate event, no duplicate audit
    await billing.set_plan(ORG_A, Plan.PRO, actor_id=ACTOR, billing_authorized=True)
    assert len(events) == 1
    assert len(audit.records) == 1


async def test_set_plan_without_publisher_is_safe():
    # Default construction (publisher=None) must never error on a real change.
    billing = BillingService(FakeOrgRepo(), EntitlementService(FakeOrgRepo()))
    await billing.set_plan(ORG_A, Plan.ENTERPRISE, billing_authorized=True)  # no raise, no event


def test_plan_changed_event_type_exists():
    # Future consumers subscribe to this without touching BillingService.
    assert EventType.ORG_PLAN_CHANGED.value == "OrgPlanChanged"
