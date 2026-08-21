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
    IntelligenceFact,
    IntelligenceView,
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

# ── Intelligence corroboration ────────────────────────────────────────
#
# NOT A NEW SCORING DIMENSION. It is the same additive nudge the category
# system above already applies, and its ceiling IS that system's largest
# nudge — so an organizational fact can corroborate what Reasoning found
# and can never outweigh it. A decision whose signal weight leads by more
# than one bias unit cannot be displaced by Intelligence at all.
#
# Scaled by the insight's OWN confidence, which is the only strength
# measure an insight carries that is comparable across insight types.
# `sample_size` deliberately does NOT enter: it counts decisive learning
# samples on a Learning fold and affected ENTITIES on a reality fold, so
# treating it as evidential weight would let a 110-entity footprint
# recitation outrank a real obligation — the exact failure the category
# map was rewritten to remove.
_INTELLIGENCE_BIAS_CEILING = _CATEGORY_BIAS[DecisionCategory.RISK]


def _relevant_facts(
    signal: ReasoningSignal, intelligence: IntelligenceView | None
) -> tuple[IntelligenceFact, ...]:
    """The organizational facts that are ABOUT this signal. Deterministic.

    Two relationships, both already recorded upstream, neither inferred:

      1. the insight names this signal in `supporting_reasoning_signal_ids`
         (written by `intelligence.service._provenance`) — it was folded FROM
         this signal, so it is by construction about it;
      2. the insight names this signal's entity among the entities in its
         evidence (written by the reality engines' `_entities`).

    Everything else is unrelated and contributes nothing. No fuzzy matching,
    no text similarity, no model: an insight either names this signal or its
    entity, or it does not.
    """
    if intelligence is None or not intelligence.facts:
        return ()
    signal_id = str(signal.signal_id)
    out: list[IntelligenceFact] = []
    for fact in intelligence.facts:
        if signal_id in fact.supporting_reasoning_signal_ids:
            out.append(fact)
            continue
        if not (signal.entity_type and signal.entity_id):
            continue
        for entity in (fact.evidence or {}).get("entities") or ():
            if (
                isinstance(entity, dict)
                and entity.get("entity_type") == signal.entity_type
                and entity.get("entity_id") == signal.entity_id
            ):
                out.append(fact)
                break
    return tuple(out)


def _intelligence_bias(facts: tuple[IntelligenceFact, ...]) -> float:
    """How much the corroborating facts may move this decision: [0, ceiling].

    The STRONGEST relevant fact decides, not the sum. Summing would let a
    handful of weak insights stack past the ceiling and outweigh the primary
    evidence, which is precisely what this bound exists to prevent.
    """
    if not facts:
        return 0.0
    strongest = max(min(max(f.confidence, 0.0), 1.0) for f in facts)
    return _INTELLIGENCE_BIAS_CEILING * strongest


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
        intelligence: IntelligenceView | None = None,
    ) -> list[Decision]:
        """Rank signals into decisions, corroborated by what the org knows.

        `intelligence` is the already-materialized `IntelligenceView` built once
        per brief by orchestration. This engine performs NO I/O: it reads the
        view it is handed and nothing else, so it stays a pure function of
        (signals, policy, intelligence) and cannot add a query per decision.

        Keyword-only with a `None` default, so every existing caller — and an
        organization that knows nothing about itself yet — gets byte-identical
        output to before.
        """
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
            facts = _relevant_facts(s, intelligence)
            score = (
                s.weight
                + _CATEGORY_BIAS.get(category, 0.0)
                + _intelligence_bias(facts)
            )
            scored.append((score, s, category, facts))

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
        for rank, (_score, s, category, facts) in enumerate(
            deduped[: self.max_items], start=1
        ):
            decisions.append(
                Decision(
                    # UNCHANGED, and deliberately: identity is (org, reasoning
                    # version, watermark, signal). Intelligence corroborates a
                    # decision, it does not make it a different decision, so
                    # loading a view can never fork an id.
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
                    # Exactly the facts that were relevant to THIS signal —
                    # the same set `_intelligence_bias` scored, so the stored
                    # provenance and the applied nudge can never disagree.
                    supporting_insight_ids=tuple(
                        sorted(f.insight_id for f in facts)
                    ),
                    # The signal's own values, passed through unchanged. NOT the
                    # ranking score: `_score` additionally folds in the category
                    # bias and the intelligence nudge, which are this engine's
                    # opinion about ordering, not the detecting engine's
                    # statement about the finding.
                    confidence=s.confidence,
                    severity=s.severity,
                )
            )
        return decisions
