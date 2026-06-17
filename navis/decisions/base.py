"""BriefItem — one card the founder sees, with its provenance."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class BriefItem:
    title: str
    why: str
    confidence: float
    category: str  # mirrors the SignalKind that produced it
    entity_ref: str | None = None
    suggested_action: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "why": self.why,
            "confidence": round(self.confidence, 3),
            "category": self.category,
            "entity_ref": self.entity_ref,
            "suggested_action": self.suggested_action,
            "evidence": self.evidence,
        }
