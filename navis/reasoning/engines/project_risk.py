"""ProjectRiskEngine — stalled or blocked projects losing momentum."""
from __future__ import annotations

from navis.reasoning.base import ReasoningEngine, ReasoningSignal, SignalKind
from navis.state.aggregate import OrganizationalState


class ProjectRiskEngine(ReasoningEngine):
    kind = SignalKind.PROJECT_RISK

    def analyze(self, state: OrganizationalState) -> list[ReasoningSignal]:
        out: list[ReasoningSignal] = []
        for p in state.projects.projects:
            if p.status == "blocked":
                out.append(
                    ReasoningSignal(
                        kind=self.kind,
                        title=f"Project blocked: {p.name}",
                        why="Marked blocked — needs an unblock decision.",
                        confidence=0.8,
                        severity=0.75,
                        entity_ref=f"project:{p.id}",
                    )
                )
            elif p.days_since_activity is not None and p.days_since_activity >= 10:
                out.append(
                    ReasoningSignal(
                        kind=self.kind,
                        title=f"Project stalling: {p.name}",
                        why=f"No activity for {p.days_since_activity:.0f} days.",
                        confidence=min(0.85, 0.4 + p.days_since_activity / 40),
                        severity=0.55,
                        entity_ref=f"project:{p.id}",
                        evidence={"days_since_activity": p.days_since_activity},
                    )
                )
        return out
