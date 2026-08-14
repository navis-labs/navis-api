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

# Category nudges so ties break by decision importance.
#
# PRIORITY / OPPORTUNITY / DELEGATION are gone: the only signal types
# that mapped to them (CONTEXT_SUMMARY, PRIORITY_LOAD, ORG_DRIFT,
# OPPORTUNITY, LEVERAGE_POINT) are interpretation, not decisions, and no
# longer reach this engine. PRIORITY additionally carried the LARGEST
# bias here, which is why a bare "Organizational context" recitation
# outranked real obligations in production briefs.
_CATEGORY_BIAS = {
    DecisionCategory.RISK: 0.12,
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
            # A SIGNAL IS NOT A DECISION. Interpretation signals — a state
            # recitation, a volume trend, an inbound-activity observation —
            # are valid reasoning output that asks nothing of anyone, so
            # they never enter the brief. `category_for` returns None for
            # them; see decisions/base.py for the full classification.
            #
            # They remain queryable as ReasoningSignals, so an engine or
            # surface that wants them as CONTEXT or EVIDENCE still can.
            # Nothing is discarded — it is just not called a decision.
            category = category_for(s.signal_type)
            if category is None:
                continue
            score = s.weight + _CATEGORY_BIAS.get(category, 0.0)
            scored.append((score, s, category))

        # Deterministic order: score desc, then signal_id as a stable tiebreak.
        scored.sort(key=lambda t: (-t[0], str(t[1].signal_id)))

        # One decision per underlying thing.
        #
        # Several engines can fire on the SAME entity — a customer who has
        # gone quiet can raise RELATIONSHIP_COOLING and
        # CUSTOMER_ENGAGEMENT_DROP off one relationship — which produced
        # two cards describing one situation. Keep the highest-scoring
        # signal for each entity; the list is already sorted, so the first
        # occurrence wins and the order stays deterministic.
        #
        # Signals with no entity are not collapsed: they are genuinely
        # org-wide and have no "same underlying thing" to share.
        seen_entities: set[tuple[str, str]] = set()
        deduped = []
        for entry in scored:
            s = entry[1]
            if s.entity_type and s.entity_id:
                key = (s.entity_type, s.entity_id)
                if key in seen_entities:
                    continue
                seen_entities.add(key)
            deduped.append(entry)

        decisions: list[Decision] = []
        for rank, (_score, s, category) in enumerate(deduped[: self.max_items], start=1):
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
