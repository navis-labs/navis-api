"""Adaptive contracts: the read-model it folds and the proposal it emits.

Pure: dataclasses, enums, and deterministic id functions. No DB, no clock, no
randomness. Deterministic ids are UUIDv5 keyed on (org, adaptive_version, ...),
so the same Intelligence always yields the same proposal/snapshot ids.
"""
from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

ADAPTIVE_SCHEMA_VERSION = 1
_ADAPTIVE_NS = uuid.UUID("c4e9a1b2-7f63-5d84-9a05-1e3c7b9d6f20")


class ProposalStatus(str, enum.Enum):
    DRAFT = "draft"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    REJECTED = "rejected"


class SnapshotStatus(str, enum.Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"


def compute_proposal_id(org_id: uuid.UUID, adaptive_version: int) -> uuid.UUID:
    return uuid.uuid5(_ADAPTIVE_NS, f"{org_id}|proposal|{adaptive_version}")


def compute_snapshot_id(org_id: uuid.UUID, policy_version: int, kind: str = "draft") -> uuid.UUID:
    return uuid.uuid5(_ADAPTIVE_NS, f"{org_id}|snapshot|{kind}|{policy_version}")


def compute_approval_id(proposal_id: uuid.UUID, approver_id: uuid.UUID, decision: str) -> uuid.UUID:
    return uuid.uuid5(_ADAPTIVE_NS, f"{proposal_id}|{approver_id}|{decision}")


# ---- read model (folded from Intelligence + Learning only) ----

@dataclass(frozen=True, slots=True)
class AggRow:
    key: str
    successes: int
    failures: int
    neutrals: int
    total: int


@dataclass(frozen=True, slots=True)
class PatternRow:
    signal_id: str
    kind: str
    success_rate: float
    samples: int


@dataclass(frozen=True, slots=True)
class InsightRow:
    insight_id: str
    insight_type: str
    key: str
    evidence: dict[str, Any]
    sample_size: int
    confidence: float
    supporting_learning_event_ids: list[str]
    learning_version: int


@dataclass(frozen=True, slots=True)
class AdaptiveView:
    org_id: uuid.UUID
    adaptive_version: int            # = the Learning version the Intelligence was built at
    as_of: datetime | None           # logical time (max insight created_at), never wall clock
    active_policy: dict[str, Any]    # current governing values (defaults if none active)
    insights: list[InsightRow]
    signal_aggs: list[AggRow] = field(default_factory=list)
    decision_aggs: list[AggRow] = field(default_factory=list)
    outcome_aggs: list[AggRow] = field(default_factory=list)
    patterns: list[PatternRow] = field(default_factory=list)


# ---- the proposal Adaptive emits (never activates) ----

@dataclass(frozen=True, slots=True)
class ProposedChange:
    field: str
    from_value: Any
    to_value: Any
    direction: str           # "increase" | "decrease"
    capped: bool             # magnitude cap bit (safety)
    reason: str
    supporting_insight_ids: list[str]
    supporting_learning_ids: list[str]


@dataclass(frozen=True, slots=True)
class ProposalDraft:
    proposal_id: uuid.UUID
    org_id: uuid.UUID
    adaptive_version: int
    as_of: datetime | None
    changes: list[ProposedChange]
    proposed_payload: dict[str, Any]
    supporting_insight_ids: list[str]
    supporting_learning_ids: list[str]
