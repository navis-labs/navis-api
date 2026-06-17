"""PriorityEngine — a load-bearing meta signal about overall pressure.

Most engines find specific risks. This one reads the whole state and emits a
single 'where the day's pressure is' priority signal, used to frame the brief.
"""
from __future__ import annotations

from navis.reasoning.base import ReasoningEngine, ReasoningSignal, SignalKind
from navis.state.aggregate import OrganizationalState


class PriorityEngine(ReasoningEngine):
    kind = SignalKind.PRIORITY

    def analyze(self, state: OrganizationalState) -> list[ReasoningSignal]:
        overdue = len(state.commitments.overdue)
        cold = len(state.relationships.going_cold())
        stalled = len(state.projects.stalled())

        pressure = overdue * 3 + cold * 2 + stalled
        if pressure == 0:
            return []

        # Normalize to a confidence-ish 0..1 (caps at ~10 weighted issues).
        load = min(1.0, pressure / 10.0)
        return [
            ReasoningSignal(
                kind=self.kind,
                title="Today's load concentrated in open risks",
                why=(
                    f"{overdue} overdue commitments, {cold} cold relationships, "
                    f"{stalled} stalling projects."
                ),
                confidence=0.7,
                severity=load,
                evidence={"overdue": overdue, "cold": cold, "stalled": stalled},
            )
        ]
