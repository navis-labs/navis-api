"""WhatMattersTodayEngine — convert ReasoningSignals into ranked Decisions.

This is prioritization, the whole of Decision's job: of everything Reasoning says
is true, what matters most right now? It is deterministic (pure function of the
signals + provenance) and references the exact signals that produced each decision.

It does NOT recompute risk/confidence (those come from the signals) and does NOT
propose actions (that is Execution). Ranking policy lives here; nothing else.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from navis.decisions.base import (
    Decision,
    DecisionCategory,
    category_for,
    compute_decision_id,
)
from navis.reasoning.v2.contract import ReasoningSignal

# Category nudges so ties break by decision importance (framing floats up).
_CATEGORY_BIAS = {
    DecisionCategory.PRIORITY: 0.20,
    DecisionCategory.RISK: 0.12,
    DecisionCategory.OPPORTUNITY: 0.10,
    DecisionCategory.DELEGATION: 0.05,
}


class WhatMattersTodayEngine:
    def __init__(self, max_items: int = 7, min_confidence: float = 0.45) -> None:
        self.max_items = max_items
        self.min_confidence = min_confidence

    def decide(
        self,
        signals: list[ReasoningSignal],
        *,
        org_id: uuid.UUID,
        reasoning_version: int,
        state_watermark: str | None,
        as_of: datetime | None,
    ) -> list[Decision]:
        scored = []
        for s in signals:
            if s.confidence < self.min_confidence:
                continue
            category = category_for(s.signal_type)
            score = s.weight + _CATEGORY_BIAS.get(category, 0.0)
            scored.append((score, s, category))

        # Deterministic order: score desc, then signal_id as a stable tiebreak.
        scored.sort(key=lambda t: (-t[0], str(t[1].signal_id)))

        decisions: list[Decision] = []
        for rank, (_score, s, category) in enumerate(scored[: self.max_items], start=1):
            decisions.append(
                Decision(
                    decision_id=compute_decision_id(
                        org_id, reasoning_version, state_watermark, s.signal_id
                    ),
                    org_id=org_id,
                    title=s.title,
                    summary=s.why,
                    why=s.why,
                    evidence=s.evidence,
                    priority_rank=rank,
                    category=category,
                    reasoning_signal_ids=[str(s.signal_id)],
                    reasoning_version=reasoning_version,
                    state_watermark=state_watermark,
                    created_at=as_of,
                )
            )
        return decisions
