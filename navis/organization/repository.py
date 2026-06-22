"""OrganizationRepository — persistence for the organization tables only.

Imports platform + its own models. Never a business domain.
"""
from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import func, or_, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import RowMapping
from sqlalchemy.ext.asyncio import AsyncSession

from navis.organization.models import (
    OrganizationEntitlement,
    OrganizationFeatureFlag,
    OrganizationLifecycle,
    OrganizationMetrics,
    OrganizationOnboarding,
    OrganizationProfile,
)


class OrganizationRepository:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    # ---- profile ----
    async def get_profile(self, org_id: uuid.UUID) -> OrganizationProfile | None:
        return await self.db.get(OrganizationProfile, org_id)

    async def upsert_profile(self, org_id: uuid.UUID, fields: dict[str, Any]) -> OrganizationProfile:
        """Atomic upsert by PK. The previous read-then-write get-or-create raced under
        concurrent first-touch (two PATCHes on a brand-new org both SELECT None, both
        INSERT the same PK -> IntegrityError, proven). ON CONFLICT DO UPDATE is the same
        write with no TOCTOU window; it updates ONLY the supplied columns, preserving the
        partial-update semantics. updated_at is set explicitly (the ORM onupdate hook does
        not fire for a core INSERT...ON CONFLICT)."""
        stmt = (
            pg_insert(OrganizationProfile)
            .values(organization_id=org_id, **fields)
            .on_conflict_do_update(
                index_elements=["organization_id"], set_={**fields, "updated_at": func.now()}
            )
        )
        await self.db.execute(stmt)
        row = await self.db.get(OrganizationProfile, org_id, populate_existing=True)
        assert row is not None  # just upserted by PK
        return row

    # ---- lifecycle ----
    async def get_lifecycle(self, org_id: uuid.UUID) -> OrganizationLifecycle | None:
        return await self.db.get(OrganizationLifecycle, org_id)

    async def ensure_lifecycle(self, org_id: uuid.UUID) -> OrganizationLifecycle:
        row = await self.get_lifecycle(org_id)
        if row is None:
            row = OrganizationLifecycle(organization_id=org_id)
            self.db.add(row)
            await self.db.flush()
        return row

    # ---- onboarding ----
    async def get_onboarding(self, org_id: uuid.UUID) -> OrganizationOnboarding | None:
        return await self.db.get(OrganizationOnboarding, org_id)

    async def ensure_onboarding(self, org_id: uuid.UUID, *, for_update: bool = False) -> OrganizationOnboarding:
        """for_update=True takes a row lock (SELECT … FOR UPDATE) so callers that
        read-modify-write the row serialize. advance() rewrites the stage_timestamps
        JSONB wholesale; without the lock two concurrent advances (independent domain
        events on separate workers) both read the old map and the later commit drops the
        other's stamp — a proven lost update. The lock makes the 2nd advance block, then
        re-read the committed map and merge. Per-org only; advances are infrequent."""
        row = await self.db.get(
            OrganizationOnboarding, org_id, with_for_update=True if for_update else False
        )
        if row is None:
            row = OrganizationOnboarding(organization_id=org_id)
            self.db.add(row)
            await self.db.flush()
        return row

    # ---- entitlements ----
    async def get_entitlement(self, org_id: uuid.UUID, capability: str) -> OrganizationEntitlement | None:
        return await self.db.scalar(
            select(OrganizationEntitlement).where(
                OrganizationEntitlement.organization_id == org_id,
                OrganizationEntitlement.capability == capability,
            )
        )

    async def list_entitlements(self, org_id: uuid.UUID) -> list[OrganizationEntitlement]:
        return list(
            (
                await self.db.scalars(
                    select(OrganizationEntitlement).where(OrganizationEntitlement.organization_id == org_id)
                )
            ).all()
        )

    async def set_entitlement(self, org_id: uuid.UUID, capability: str, enabled: bool) -> None:
        stmt = (
            pg_insert(OrganizationEntitlement)
            .values(organization_id=org_id, capability=capability, enabled=enabled)
            .on_conflict_do_update(
                index_elements=["organization_id", "capability"], set_={"enabled": enabled}
            )
        )
        await self.db.execute(stmt)

    async def set_entitlements(self, org_id: uuid.UUID, caps: dict[str, bool]) -> None:
        """Upsert many entitlement rows in ONE statement. The per-capability seed is the
        provisioning hot path; N sequential writes blew past the 15s event-handler
        timeout (left orgs unprovisioned). One multi-row upsert is the same idempotent
        (org, capability) conflict, scale-independent of capability count."""
        if not caps:
            return
        rows = [
            {"id": uuid.uuid4(), "organization_id": org_id, "capability": cap, "enabled": enabled}
            for cap, enabled in caps.items()
        ]
        stmt = pg_insert(OrganizationEntitlement).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=["organization_id", "capability"],
            set_={"enabled": stmt.excluded.enabled},
        )
        await self.db.execute(stmt)

    # ---- feature flags ----
    async def get_flag(self, org_id: uuid.UUID | None, flag: str) -> OrganizationFeatureFlag | None:
        return await self.db.scalar(
            select(OrganizationFeatureFlag).where(
                OrganizationFeatureFlag.organization_id == org_id,
                OrganizationFeatureFlag.flag == flag,
            )
        )

    async def flags_for(self, org_id: uuid.UUID, flag: str) -> list[OrganizationFeatureFlag]:
        """Both the org-specific row and the global default (org_id NULL)."""
        return list(
            (
                await self.db.scalars(
                    select(OrganizationFeatureFlag).where(
                        OrganizationFeatureFlag.flag == flag,
                        # NOT in_([org_id, None]): SQL `IN (x, NULL)` never matches NULL
                        # rows, so global (org_id IS NULL) flags would be silently dropped.
                        or_(
                            OrganizationFeatureFlag.organization_id == org_id,
                            OrganizationFeatureFlag.organization_id.is_(None),
                        ),
                    )
                )
            ).all()
        )

    async def set_flag(self, org_id: uuid.UUID | None, flag: str, *, kind: str, enabled: bool) -> None:
        existing = await self.get_flag(org_id, flag)
        if existing is None:
            self.db.add(OrganizationFeatureFlag(organization_id=org_id, flag=flag, kind=kind, enabled=enabled))
        else:
            existing.kind = kind
            existing.enabled = enabled
        await self.db.flush()

    # ---- metrics ----
    async def get_metrics(self, org_id: uuid.UUID) -> OrganizationMetrics | None:
        return await self.db.get(OrganizationMetrics, org_id)

    async def upsert_metrics(self, org_id: uuid.UUID, *, ttfi_seconds: int | None, activation_score: float) -> OrganizationMetrics:
        """Atomic upsert by PK. recompute() runs per onboarding advance, and advances are
        driven by independent domain events (FIRST_SIGNAL, MEMORY_READY, …) in separate
        sessions — two racing the first-touch both INSERT the same PK -> IntegrityError
        (proven). ON CONFLICT DO UPDATE is the same idempotent write without the TOCTOU
        window. updated_at is set explicitly (ORM onupdate does not fire for core upsert)."""
        stmt = (
            pg_insert(OrganizationMetrics)
            .values(organization_id=org_id, ttfi_seconds=ttfi_seconds, activation_score=activation_score)
            .on_conflict_do_update(
                index_elements=["organization_id"],
                set_={"ttfi_seconds": ttfi_seconds, "activation_score": activation_score, "updated_at": func.now()},
            )
        )
        await self.db.execute(stmt)
        row = await self.db.get(OrganizationMetrics, org_id, populate_existing=True)
        assert row is not None  # just upserted by PK
        return row

    # ---- capability resolution (hot path) ----
    async def resolve_capability(self, org_id: uuid.UUID, capability: str, *, kill_flag: str) -> RowMapping:
        """All inputs for one capability decision in a SINGLE round-trip: lifecycle
        status/plan, the explicit (org, capability) entitlement, and whether a kill switch
        (org-specific OR global) is enabled. Collapses can_use's 3 sequential reads into 1
        — the path every gated business request hits, so 3×→1× Neon RTT is the difference
        between viable and not at 1M orgs. Decision logic stays in EntitlementService; this
        only fetches. The kill EXISTS mirrors FeatureFlagService.is_killed's fail-safe OR."""
        stmt = text(
            "SELECT l.status AS status, l.plan AS plan, e.enabled AS explicit, "
            "EXISTS(SELECT 1 FROM organization_feature_flags f "
            "WHERE f.flag = :kill AND (f.organization_id = base.oid OR f.organization_id IS NULL) "
            "AND f.enabled) AS killed "
            "FROM (SELECT CAST(:org AS uuid) AS oid) base "
            "LEFT JOIN organization_lifecycles l ON l.organization_id = base.oid "
            "LEFT JOIN organization_entitlements e ON e.organization_id = base.oid AND e.capability = :cap"
        )
        # .one(): the synthetic base row guarantees exactly one result row, so this never
        # raises and returns a non-optional RowMapping (removes Optional-subscript at source).
        return (
            await self.db.execute(stmt, {"org": str(org_id), "cap": capability, "kill": kill_flag})
        ).mappings().one()

    # ---- read-side rollup ----
    async def rollup_row(self, org_id: uuid.UUID) -> RowMapping:
        """One round-trip rollup of the three org tables (all PK=organization_id).

        Replaces 3 sequential PK reads with a single anchored LEFT JOIN: a synthetic base
        row (SELECT :org_id) guarantees exactly one result row with NULLs for any absent
        table, so per-table presence semantics are identical to the independent reads — but
        at 1× Neon RTT instead of 3× on the user-facing metrics endpoint."""
        stmt = text(
            "SELECT l.status AS status, l.plan AS plan, "
            "o.stage AS stage, o.signals_count AS signals_count, o.memory_count AS memory_count, "
            "m.ttfi_seconds AS ttfi_seconds, m.activation_score AS activation_score "
            "FROM (SELECT CAST(:org_id AS uuid) AS oid) base "
            "LEFT JOIN organization_lifecycles l ON l.organization_id = base.oid "
            "LEFT JOIN organization_onboarding o ON o.organization_id = base.oid "
            "LEFT JOIN organization_metrics m ON m.organization_id = base.oid"
        )
        # .one(): synthetic base row guarantees exactly one result row (non-optional).
        return (await self.db.execute(stmt, {"org_id": str(org_id)})).mappings().one()

    async def flush(self) -> None:
        await self.db.flush()

    async def commit(self) -> None:
        await self.db.commit()
