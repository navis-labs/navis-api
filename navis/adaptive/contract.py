"""Adaptive contracts: the read-model it folds and the proposal it emits.

Pure: dataclasses, enums, and deterministic id functions. No DB, no clock, no
randomness. Deterministic ids are UUIDv5 keyed on (org, adaptive_version, ...),
so the same Intelligence always yields the same proposal/snapshot ids.
"""
from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass
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

# `AggRow` USED TO BE HERE. It existed only to carry `learning_aggregates` rows
# into the three `AdaptiveView` fields below, which nothing ever read; both are
# gone. Note this was a SECOND, independent copy of the same shape — Intelligence
# has its own `AggRow` in `intelligence/insights.py`, which is alive and stays.


# `PatternRow` USED TO BE HERE, alongside an `AdaptiveView.patterns` field fed by
# a direct `select(LearningPattern)`. Nothing read the field, and the query was
# Adaptive's last read of a Learning TABLE. Learned evidence now has exactly one
# way into Adaptive — `navis.learning.metrics.learning_snapshot` — which is what
# makes "Adaptive consumes the Learning contract" an enforceable statement rather
# than a convention.


@dataclass(frozen=True, slots=True)
class OutcomeEvidence:
    """One outcome-type counter, as Adaptive is allowed to see it.

    Flattened from `LearningSnapshot` by the repository — primitives only, no ORM
    row and no Learning type, so this module stays pure and Adaptive never learns
    Learning's storage shape.

    `outcome_type` is the only TYPE-STABLE key Learning exposes: signal and
    decision aggregates key on opaque uuids, which is exactly why `proposals.py`
    has never been able to tune a per-signal threshold. A named outcome type is
    something a rule can reason about without guessing.
    """

    outcome_type: str
    surface: str                 # delivery | judgement | world
    successes: int
    failures: int
    neutrals: int
    total: int
    supporting_learning_event_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class LearningEvidence:
    """What one organization has learned, as evidence for a policy candidate.

    Org-scoped by construction: it carries its own `org_id` and is built from a
    single `learning_snapshot(db, org_id)` call, so two organizations' evidence
    cannot be combined in one value without that being visible.
    """

    org_id: uuid.UUID
    learning_version: int
    as_of: datetime | None
    outcomes: tuple[OutcomeEvidence, ...] = ()

    def by_type(self, outcome_type: str) -> OutcomeEvidence | None:
        for o in self.outcomes:
            if o.outcome_type == outcome_type:
                return o
        return None


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
    # Learned evidence, straight from Learning's read contract. Optional so every
    # existing construction site (and the test fakes) stays valid and behaves
    # exactly as before: no evidence => the Learning-derived rules produce nothing.
    learning: LearningEvidence | None = None


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
