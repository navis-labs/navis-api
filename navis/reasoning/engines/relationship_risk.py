"""RelationshipRiskEngine — relationships going cold (investors, customers...)."""
from __future__ import annotations

from navis.reasoning.base import ReasoningEngine, ReasoningSignal, SignalKind
from navis.state.aggregate import OrganizationalState

# Investors/customers going silent matters faster than generic contacts.
_KIND_SEVERITY = {
    "investor": 0.9,
    "customer": 0.85,
    "candidate": 0.7,
    "vendor": 0.4,
    "internal": 0.5,
    "other": 0.4,
}


class RelationshipRiskEngine(ReasoningEngine):
    kind = SignalKind.RELATIONSHIP_RISK

    def analyze(self, state: OrganizationalState) -> list[ReasoningSignal]:
        out: list[ReasoningSignal] = []
        for r in state.relationships.going_cold(threshold_days=7.0):
            days = r.days_since_contact or 0.0
            severity = _KIND_SEVERITY.get(r.kind, 0.4)
            out.append(
                ReasoningSignal(
                    kind=self.kind,
                    title=f"{r.kind.title()} relationship going cold",
                    why=f"{days:.0f} days of silence with this {r.kind}.",
                    confidence=min(0.9, 0.5 + days / 30),
                    severity=severity,
                    entity_ref=f"relationship:{r.id}",
                    evidence={"days_since_contact": days, "kind": r.kind, "health": r.health},
                )
            )
        return out
