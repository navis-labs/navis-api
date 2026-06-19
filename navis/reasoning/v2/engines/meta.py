"""Meta engines — interpretations over the whole projection (priority, opportunity,
leverage, trend, context). Still pure, still State-only, still no LLM.

These produce interpretation, NOT decisions: PriorityEngine surfaces where pressure
is concentrated; it does NOT rank actions or recommend (that is Decision's job).
"""
from __future__ import annotations

from navis.reasoning.v2.base import ReasoningContext, ReasoningEngine, days_between
from navis.reasoning.v2.contract import SignalDraft, SignalType
from navis.reasoning.v2.engines.risk import (
    COMMITMENT_SLIPPING_DAYS,
    PROJECT_STALLED_DAYS,
    RELATIONSHIP_COOLING_DAYS,
)
from navis.state.v2.models import OrganizationalState

OPPORTUNITY_RECENT_DAYS = 2.0
LEVERAGE_MIN_PEOPLE = 2


def _count_slipping(state, as_of) -> int:
    return sum(
        1 for c in state.commitments()
        if c.status == "open"
        and (days_between(as_of, c.last_seen) or 0) >= COMMITMENT_SLIPPING_DAYS
    )


def _count_cooling(state, as_of) -> int:
    return sum(
        1 for r in state.relationships()
        if (days_between(as_of, r.last_interaction_at) or 0) >= RELATIONSHIP_COOLING_DAYS
    )


def _count_stalled(state, as_of) -> int:
    return sum(
        1 for p in state.projects()
        if (days_between(as_of, p.last_seen) or 0) >= PROJECT_STALLED_DAYS
    )


class PriorityEngine(ReasoningEngine):
    name = "priority"
    version = "2.0.0"

    def analyze(self, state: OrganizationalState, ctx: ReasoningContext) -> list[SignalDraft]:
        slipping = _count_slipping(state, ctx.as_of)
        cooling = _count_cooling(state, ctx.as_of)
        stalled = _count_stalled(state, ctx.as_of)
        pressure = slipping * 3 + cooling * 2 + stalled
        if pressure == 0:
            return []
        return [
            SignalDraft(
                signal_type=SignalType.PRIORITY_LOAD,
                entity_type=None,
                entity_id=None,
                title="Open-risk load concentrated",
                why=f"{slipping} slipping commitments, {cooling} cooling relationships, "
                    f"{stalled} stalled projects.",
                confidence=0.7,
                severity=min(1.0, pressure / 10.0),
                dedup_key="priority_load",
                evidence={"slipping": slipping, "cooling": cooling, "stalled": stalled},
            )
        ]


class OpportunityEngine(ReasoningEngine):
    name = "opportunity"
    version = "2.0.0"

    def analyze(self, state: OrganizationalState, ctx: ReasoningContext) -> list[SignalDraft]:
        out = []
        for r in state.relationships():
            age = days_between(ctx.as_of, r.last_interaction_at)
            if age is None or age > OPPORTUNITY_RECENT_DAYS:
                continue
            if "inbound" not in (r.directions or []):
                continue
            out.append(
                SignalDraft(
                    signal_type=SignalType.OPPORTUNITY,
                    entity_type="relationship",
                    entity_id=r.entity_id,
                    title="Active inbound engagement",
                    why=f"Inbound contact {age:.0f} days ago — warm window.",
                    confidence=0.6,
                    severity=0.5,
                    dedup_key=r.entity_id,
                    evidence={"days_since": round(age, 2), "person_email": r.person_email},
                )
            )
        return out


class LeverageEngine(ReasoningEngine):
    name = "leverage"
    version = "2.0.0"

    def analyze(self, state: OrganizationalState, ctx: ReasoningContext) -> list[SignalDraft]:
        out = []
        for co in state.companies():
            people = co.person_ids
            if len(people) < LEVERAGE_MIN_PEOPLE:
                continue
            out.append(
                SignalDraft(
                    signal_type=SignalType.LEVERAGE_POINT,
                    entity_type="company",
                    entity_id=co.entity_id,
                    title=f"Hub company: {co.domain}",
                    why=f"{len(people)} known people connect to this company.",
                    confidence=0.65,
                    severity=0.4,
                    dedup_key=co.entity_id,
                    evidence={"people": len(people), "domain": co.domain},
                )
            )
        return out


class TrendEngine(ReasoningEngine):
    name = "trend"
    version = "2.0.0"

    def analyze(self, state: OrganizationalState, ctx: ReasoningContext) -> list[SignalDraft]:
        if ctx.prior is None:
            return []  # trend needs a prior State snapshot (still State-only)
        cur = state.counts()
        prev = ctx.prior.counts()
        out = []
        for key in ("commitment", "meeting"):
            delta = cur.get(key, 0) - prev.get(key, 0)
            if delta > 0:
                out.append(
                    SignalDraft(
                        signal_type=SignalType.ORG_DRIFT,
                        entity_type=None,
                        entity_id=None,
                        title=f"Rising {key} volume",
                        why=f"{key} entities grew by {delta} since the prior state.",
                        confidence=0.6,
                        severity=min(0.7, delta / 10),
                        dedup_key=f"drift:{key}",
                        evidence={"delta": delta, "current": cur.get(key, 0)},
                    )
                )
        return out


class ContextEngine(ReasoningEngine):
    name = "context"
    version = "2.0.0"

    def analyze(self, state: OrganizationalState, ctx: ReasoningContext) -> list[SignalDraft]:
        counts = state.counts()
        if sum(counts.values()) == 0:
            return []
        return [
            SignalDraft(
                signal_type=SignalType.CONTEXT_SUMMARY,
                entity_type=None,
                entity_id=None,
                title="Organizational context",
                why="Current factual footprint of the organization's state.",
                confidence=1.0,
                severity=0.2,
                dedup_key="context_summary",
                evidence={"counts": counts},
            )
        ]
