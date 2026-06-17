"""CommitmentRiskEngine — surfaces slipping/overdue promises."""
from __future__ import annotations

from navis.reasoning.base import ReasoningEngine, ReasoningSignal, SignalKind
from navis.state.aggregate import OrganizationalState

# Past this age an open commitment with no due date is "slipping".
_SLIPPING_AGE_DAYS = 5.0


class CommitmentRiskEngine(ReasoningEngine):
    kind = SignalKind.COMMITMENT_RISK

    def analyze(self, state: OrganizationalState) -> list[ReasoningSignal]:
        out: list[ReasoningSignal] = []

        for c in state.commitments.overdue:
            age = c.age_days or 0.0
            out.append(
                ReasoningSignal(
                    kind=self.kind,
                    title=f"Overdue commitment: {_clip(c.description)}",
                    why=f"Past due and still open ({age:.0f} days old).",
                    confidence=min(0.95, 0.6 + (c.age_days or 0) / 30),
                    severity=0.9,
                    entity_ref=f"commitment:{c.id}",
                    evidence={"age_days": age, "status": c.status},
                )
            )

        for c in state.commitments.open:
            age = c.age_days or 0.0
            if age >= _SLIPPING_AGE_DAYS:
                out.append(
                    ReasoningSignal(
                        kind=self.kind,
                        title=f"Commitment slipping: {_clip(c.description)}",
                        why=f"Open for {age:.0f} days with no delivery yet.",
                        confidence=min(0.9, 0.4 + age / 30),
                        severity=0.6,
                        entity_ref=f"commitment:{c.id}",
                        evidence={"age_days": age},
                    )
                )
        return out


def _clip(text: str, n: int = 80) -> str:
    text = text.strip().replace("\n", " ")
    return text if len(text) <= n else text[: n - 1] + "…"
