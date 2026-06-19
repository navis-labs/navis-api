"""In-memory OrganizationRepository fake (the services depend on the repo seam)."""
from __future__ import annotations

import uuid
from types import SimpleNamespace

from navis.organization.contract import LifecycleStatus, OnboardingStage, Plan

ORG_A = uuid.UUID("11111111-1111-1111-1111-111111111111")
ORG_B = uuid.UUID("22222222-2222-2222-2222-222222222222")
ACTOR = uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")


class FakeOrgRepo:
    def __init__(self) -> None:
        self.profiles: dict = {}
        self.lifecycles: dict = {}
        self.onboarding: dict = {}
        self.entitlements: dict = {}   # (org, cap) -> ns
        self.flags: list = []          # ns(organization_id, flag, kind, enabled)
        self.metrics: dict = {}

    # profile
    async def get_profile(self, org_id):
        return self.profiles.get(org_id)

    async def upsert_profile(self, org_id, fields):
        row = self.profiles.get(org_id) or SimpleNamespace(
            organization_id=org_id, company_name=None, website=None, industry=None,
            employee_count=None, hq_country=None,
        )
        for k, v in fields.items():
            setattr(row, k, v)
        self.profiles[org_id] = row
        return row

    # lifecycle
    async def get_lifecycle(self, org_id):
        return self.lifecycles.get(org_id)

    async def ensure_lifecycle(self, org_id):
        row = self.lifecycles.get(org_id)
        if row is None:
            row = SimpleNamespace(
                organization_id=org_id, status=LifecycleStatus.TRIAL.value, plan=Plan.FREE.value,
                activated_at=None, suspended_at=None, churned_at=None,
            )
            self.lifecycles[org_id] = row
        return row

    # onboarding
    async def get_onboarding(self, org_id):
        return self.onboarding.get(org_id)

    async def ensure_onboarding(self, org_id):
        row = self.onboarding.get(org_id)
        if row is None:
            row = SimpleNamespace(
                organization_id=org_id, stage=OnboardingStage.CREATED.value,
                signals_count=0, memory_count=0, stage_timestamps={}, completed_at=None,
            )
            self.onboarding[org_id] = row
        return row

    # entitlements
    async def get_entitlement(self, org_id, capability):
        return self.entitlements.get((org_id, capability))

    async def list_entitlements(self, org_id):
        return [v for (o, _), v in self.entitlements.items() if o == org_id]

    async def set_entitlement(self, org_id, capability, enabled):
        self.entitlements[(org_id, capability)] = SimpleNamespace(
            organization_id=org_id, capability=capability, enabled=enabled
        )

    # flags
    async def get_flag(self, org_id, flag):
        for f in self.flags:
            if f.organization_id == org_id and f.flag == flag:
                return f
        return None

    async def flags_for(self, org_id, flag):
        return [f for f in self.flags if f.flag == flag and f.organization_id in (org_id, None)]

    async def set_flag(self, org_id, flag, *, kind, enabled):
        existing = await self.get_flag(org_id, flag)
        if existing is None:
            self.flags.append(SimpleNamespace(organization_id=org_id, flag=flag, kind=kind, enabled=enabled))
        else:
            existing.kind, existing.enabled = kind, enabled

    # metrics
    async def get_metrics(self, org_id):
        return self.metrics.get(org_id)

    async def upsert_metrics(self, org_id, *, ttfi_seconds, activation_score):
        row = SimpleNamespace(organization_id=org_id, ttfi_seconds=ttfi_seconds, activation_score=activation_score)
        self.metrics[org_id] = row
        return row

    async def flush(self):
        pass


class FakePublisher:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    async def __call__(self, event: str, payload: dict) -> None:
        self.events.append((event, payload))
