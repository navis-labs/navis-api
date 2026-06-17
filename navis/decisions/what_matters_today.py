"""WhatMattersTodayEngine — rank reasoning signals into a founder's brief.

This is deliberately NOT a god object. It does one thing: take signals that
reasoning already produced and order/curate them. All the domain knowledge of
*what is risky* lives in the engines; all the knowledge of *what to do* lives in
execution. This engine only answers "of everything true right now, what matters
most today?".
"""
from __future__ import annotations

from navis.decisions.base import BriefItem
from navis.reasoning.base import ReasoningSignal, SignalKind

# A nudge so two equally-weighted signals break ties by category importance.
_CATEGORY_BIAS = {
    SignalKind.COMMITMENT_RISK: 0.15,
    SignalKind.RELATIONSHIP_RISK: 0.12,
    SignalKind.CUSTOMER_RISK: 0.12,
    SignalKind.PROJECT_RISK: 0.08,
    SignalKind.HIRING_RISK: 0.06,
    SignalKind.OPPORTUNITY: 0.10,
    SignalKind.PRIORITY: 0.20,  # framing signal floats to the top
    SignalKind.LEVERAGE: 0.05,
}

# Lightweight action hints per category (execution proposes the real action).
_ACTION_HINT = {
    SignalKind.COMMITMENT_RISK: "Send a status update or close it out today.",
    SignalKind.RELATIONSHIP_RISK: "Send a short re-engagement note.",
    SignalKind.PROJECT_RISK: "Identify the blocker and the owner.",
    SignalKind.OPPORTUNITY: "Decide whether to pursue now.",
}


class WhatMattersTodayEngine:
    def __init__(self, max_items: int = 7, min_confidence: float = 0.45) -> None:
        self.max_items = max_items
        self.min_confidence = min_confidence

    def rank(self, signals: list[ReasoningSignal]) -> list[BriefItem]:
        scored = [
            (s.weight + _CATEGORY_BIAS.get(s.kind, 0.0), s)
            for s in signals
            if s.confidence >= self.min_confidence
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)

        items: list[BriefItem] = []
        for _, s in scored[: self.max_items]:
            items.append(
                BriefItem(
                    title=s.title,
                    why=s.why,
                    confidence=s.confidence,
                    category=s.kind.value,
                    entity_ref=s.entity_ref,
                    suggested_action=_ACTION_HINT.get(s.kind),
                    evidence=s.evidence,
                )
            )
        return items
