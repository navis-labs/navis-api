"""ReasoningSignal contract + the ReasoningEngine protocol."""
from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from navis.state.aggregate import OrganizationalState


class SignalKind(str, enum.Enum):
    COMMITMENT_RISK = "commitment_risk"
    RELATIONSHIP_RISK = "relationship_risk"
    CUSTOMER_RISK = "customer_risk"
    HIRING_RISK = "hiring_risk"
    PROJECT_RISK = "project_risk"
    OPPORTUNITY = "opportunity"
    PRIORITY = "priority"
    LEVERAGE = "leverage"


@dataclass(slots=True)
class ReasoningSignal:
    """A single inference about the org. Confidence + a human 'why' are mandatory
    so the decision layer can stay trust-first (explainable, traceable)."""

    kind: SignalKind
    title: str
    why: str
    confidence: float  # 0..1
    severity: float = 0.5  # 0..1, how much it matters if true
    entity_ref: str | None = None  # e.g. "commitment:<uuid>"
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def weight(self) -> float:
        """Combined importance used for ranking in the decision layer."""
        return self.confidence * self.severity


class ReasoningEngine(ABC):
    """A focused engine. One concern, one kind of signal."""

    kind: SignalKind

    @abstractmethod
    def analyze(self, state: OrganizationalState) -> list[ReasoningSignal]:
        raise NotImplementedError
