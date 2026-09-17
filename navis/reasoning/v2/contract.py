"""ReasoningSignal contract — fully provenanced, deterministic, replayable.

`signal_id` is a deterministic UUIDv5 of the reasoning identity tuple, so the same
State + same reasoning version reproduces the same id. `created_at` is the LOGICAL
time (= state.as_of), never wall-clock — that is how we keep determinism while
still carrying a timestamp.
"""
from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

# Bump when ANY engine's logic/thresholds change → signals get new ids, replay-safe.
# v2: added MAX_SURFACE_AGE_DAYS cap to risk.py engines so signals aren't
#     emitted for entities silent > 30 days (see engines/risk.py header).
# v3: OpportunityEngine decides organizational relevance from the SET of
#     signal_importance classifications observed across a relationship, instead
#     of emitting for every recent inbound address (see engines/meta.py).
# v4: RelationshipReasoningEngine applies that same relevance vocabulary, now
#     shared from base.py, and REQUIRES human-bearing evidence rather than
#     merely suppressing on evidence of irrelevance. Live, RELATIONSHIP_COOLING
#     was firing for newsletters, billing robots and no-reply senders, and
#     Intelligence aggregated them into organizational risk. PriorityEngine's
#     cooling count moved under the same gate so PRIORITY_LOAD cannot contradict
#     the signals. Historical v3 signal ids stay immutable and are not re-minted.
REASONING_VERSION = 4

_REASONING_NAMESPACE = uuid.UUID("7a2b9c14-3d5e-5f60-8b1a-9c0d2e4f6a81")


class SignalType(str, enum.Enum):
    COMMITMENT_SLIPPING = "commitment_slipping"
    RELATIONSHIP_COOLING = "relationship_cooling"
    PROJECT_STALLED = "project_stalled"
    CUSTOMER_ENGAGEMENT_DROP = "customer_engagement_drop"
    DECISION_UNRESOLVED = "decision_unresolved"
    # Someone emailed the founder and no reply from them exists in the
    # thread. Observed, never inferred — see engines/waiting.py.
    WAITING_ON_USER_REPLY = "waiting_on_user_reply"
    # A dated promise made TO the org whose deadline has passed with no
    # delivery. Distinct from COMMITMENT_SLIPPING, which covers UNDATED
    # promises by age alone; the two are mutually exclusive by design.
    COMMITMENT_FOLLOW_UP = "commitment_follow_up"
    # Somebody asked the founder to decide and no decision has landed.
    # Distinct from DECISION_UNRESOLVED, which covers decisions the org
    # RECORDED and never closed out.
    DECISION_OWED = "decision_owed"
    MEETING_OVERLOAD = "meeting_overload"
    PRIORITY_LOAD = "priority_load"
    OPPORTUNITY = "opportunity"
    LEVERAGE_POINT = "leverage_point"
    ORG_DRIFT = "org_drift"
    CONTEXT_SUMMARY = "context_summary"


@dataclass(frozen=True, slots=True)
class SignalDraft:
    """An engine's output before provenance is stamped on (mirrors memory's draft)."""

    signal_type: SignalType
    entity_type: str | None
    entity_id: str | None
    title: str
    why: str
    confidence: float
    severity: float
    dedup_key: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Structural validation at the engine-output boundary (fail-closed; the runner isolates a
        # bad engine, so the pass continues). confidence/severity feed Decision's weight
        # (confidence * severity), so an out-of-range value would silently corrupt ranking;
        # dedup_key anchors the content-addressed signal_id. Ranges only — no business logic. This
        # protects the reasoning->decision contract from any engine (deterministic OR future AI).
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"SignalDraft.confidence must be in [0, 1], got {self.confidence}")
        if not 0.0 <= self.severity <= 1.0:
            raise ValueError(f"SignalDraft.severity must be in [0, 1], got {self.severity}")
        if not self.dedup_key:
            raise ValueError("SignalDraft.dedup_key must be non-empty (it anchors the signal_id)")


@dataclass(frozen=True, slots=True)
class ReasoningSignal:
    signal_id: uuid.UUID
    org_id: uuid.UUID
    signal_type: SignalType
    entity_type: str | None
    entity_id: str | None
    title: str
    why: str
    evidence: dict[str, Any]
    confidence: float
    severity: float
    engine_name: str
    engine_version: str
    state_as_of: datetime | None
    state_watermark: str | None
    reasoning_version: int
    created_at: datetime | None  # logical time = state_as_of (deterministic)

    @property
    def weight(self) -> float:
        return self.confidence * self.severity

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal_id": str(self.signal_id),
            "org_id": str(self.org_id),
            "signal_type": self.signal_type.value,
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "title": self.title,
            "why": self.why,
            "evidence": self.evidence,
            "confidence": round(self.confidence, 4),
            "severity": round(self.severity, 4),
            "engine_name": self.engine_name,
            "engine_version": self.engine_version,
            "state_as_of": self.state_as_of.isoformat() if self.state_as_of else None,
            "state_watermark": self.state_watermark,
            "reasoning_version": self.reasoning_version,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


def compute_signal_id(
    org_id: uuid.UUID,
    reasoning_version: int,
    engine_name: str,
    engine_version: str,
    signal_type: SignalType,
    entity_id: str | None,
    dedup_key: str,
) -> uuid.UUID:
    key = (
        f"{org_id}|{reasoning_version}|{engine_name}|{engine_version}"
        f"|{signal_type.value}|{entity_id or '-'}|{dedup_key}"
    )
    return uuid.uuid5(_REASONING_NAMESPACE, key)
