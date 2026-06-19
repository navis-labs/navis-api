"""Organization V1 ORM. Answers only "what company is this?" and "what can it do?".

Owns profile / lifecycle / onboarding / entitlements / plan / feature flags /
metrics. Does NOT own users/sessions/roles/permissions/api keys/sso/scim — those
stay in Identity. All tables are org-scoped (one row per org, except entitlements/
flags which are per-(org,key)).
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from navis.organization.contract import LifecycleStatus, OnboardingStage, Plan
from navis.platform.database.base import Base


class OrganizationProfile(Base):
    __tablename__ = "organization_profiles"

    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), primary_key=True
    )
    company_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    website: Mapped[str | None] = mapped_column(String(255), nullable=True)
    industry: Mapped[str | None] = mapped_column(String(128), nullable=True)
    employee_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    hq_country: Mapped[str | None] = mapped_column(String(2), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class OrganizationLifecycle(Base):
    __tablename__ = "organization_lifecycles"

    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), primary_key=True
    )
    status: Mapped[str] = mapped_column(String(16), default=LifecycleStatus.TRIAL.value, nullable=False, index=True)
    plan: Mapped[str] = mapped_column(String(16), default=Plan.FREE.value, nullable=False)
    activated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    suspended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    churned_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class OrganizationOnboarding(Base):
    __tablename__ = "organization_onboarding"

    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), primary_key=True
    )
    stage: Mapped[str] = mapped_column(String(32), default=OnboardingStage.CREATED.value, nullable=False, index=True)
    signals_count: Mapped[int] = mapped_column(Integer, default=0)
    memory_count: Mapped[int] = mapped_column(Integer, default=0)
    # stage value -> ISO8601 timestamp the stage was first reached (drives TTFI).
    stage_timestamps: Mapped[dict] = mapped_column(JSONB, default=dict)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class OrganizationEntitlement(Base):
    __tablename__ = "organization_entitlements"
    __table_args__ = (UniqueConstraint("organization_id", "capability", name="uq_org_capability"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    capability: Mapped[str] = mapped_column(String(32), nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)


class OrganizationFeatureFlag(Base):
    __tablename__ = "organization_feature_flags"
    __table_args__ = (UniqueConstraint("organization_id", "flag", name="uq_org_flag"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # org_id NULL = a global default flag (applies to every org unless overridden).
    organization_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=True, index=True
    )
    flag: Mapped[str] = mapped_column(String(64), nullable=False)
    kind: Mapped[str] = mapped_column(String(16), default="release")  # release|permission|kill_switch
    enabled: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)


class OrganizationMetrics(Base):
    __tablename__ = "organization_metrics"

    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), primary_key=True
    )
    ttfi_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)
    activation_score: Mapped[float] = mapped_column(Float, default=0.0)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
