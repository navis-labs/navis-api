"""Adaptive V1 — append-only governance tables.

policy_proposals  : what Adaptive proposes (deterministic, never activates)
policy_snapshots  : versioned governing policy; exactly one ACTIVE per org
policy_approvals  : immutable record of who approved/rejected, when

All append-only and org-scoped (partition-ready on organization_id). Nothing here
mutates a prior row's meaning except a status transition on the proposal/snapshot
lifecycle (draft -> active/superseded), which is itself auditable.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from navis.adaptive.contract import ADAPTIVE_SCHEMA_VERSION
from navis.platform.database.base import Base


class PolicyProposal(Base):
    __tablename__ = "policy_proposals"

    proposal_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    adaptive_version: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False, index=True)

    proposed_changes: Mapped[list] = mapped_column(JSONB, default=list)
    proposed_payload: Mapped[dict] = mapped_column(JSONB, default=dict)
    supporting_insight_ids: Mapped[list] = mapped_column(JSONB, default=list)
    supporting_learning_ids: Mapped[list] = mapped_column(JSONB, default=list)

    # The draft snapshot this proposal would activate (deterministic id).
    policy_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    decided_by: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)

    schema_version: Mapped[int] = mapped_column(Integer, default=ADAPTIVE_SCHEMA_VERSION)
    created_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class PolicySnapshot(Base):
    __tablename__ = "policy_snapshots"
    __table_args__ = (
        # AT MOST ONE ACTIVE POLICY PER ORGANIZATION, enforced by the database.
        #
        # This index has existed in production since migration 0012
        # (`uq_ps_one_active`) but was never declared on the model, so
        # `Base.metadata.create_all` — which every test fixture and disposable
        # database uses — produced a schema WITHOUT it. The invariant that makes
        # concurrent approval safe was therefore real in production and absent
        # everywhere it could be tested, which is the worst possible split: a
        # double activation raises a unique violation live and silently corrupts
        # policy history in a test.
        #
        # Declared here so both paths agree. `get_active_snapshot` uses
        # `db.scalar()`, which relies on there being at most one row to return a
        # deterministic answer; this is what makes that true.
        Index(
            "uq_ps_one_active",
            "organization_id",
            unique=True,
            postgresql_where=text("status = 'active'"),
        ),
    )

    policy_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    policy_version: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False, index=True)

    payload: Mapped[dict] = mapped_column(JSONB, default=dict)
    source_intelligence_ids: Mapped[list] = mapped_column(JSONB, default=list)
    source_learning_event_ids: Mapped[list] = mapped_column(JSONB, default=list)

    created_by: Mapped[str | None] = mapped_column(String(64), nullable=True)  # "adaptive"/"system"/user
    approved_by: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)

    schema_version: Mapped[int] = mapped_column(Integer, default=ADAPTIVE_SCHEMA_VERSION)
    created_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    activated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class PolicyApproval(Base):
    __tablename__ = "policy_approvals"

    approval_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    proposal_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, index=True)
    decision: Mapped[str] = mapped_column(String(16), nullable=False)  # approved | rejected
    approved_by: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    approved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
