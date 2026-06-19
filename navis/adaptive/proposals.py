"""Deterministic policy-change derivation — a pure fold over the AdaptiveView.

Adaptive proposes changes to *governing variables only*, grounded in what
Intelligence learned. Every change is bounded, magnitude-capped, and cites the
exact insight + learning events that justify it. No clock, no randomness.

V1 ships rules grounded in the one cross-org-safe, type-stable signal available:
the organization-level outcome profile (Organizational Intelligence). Per-signal
threshold tuning is deferred until signal_type propagates to Learning (the known
attribution gap), because patterns key on opaque signal_ids, not types.
"""
from __future__ import annotations

from navis.adaptive.contract import (
    AdaptiveView,
    InsightRow,
    ProposalDraft,
    ProposedChange,
    compute_proposal_id,
)
from navis.adaptive.policy import cap_change

MIN_SAMPLES = 5
TRUST_HIGH = 0.6   # earned trust -> surface more, lower the bar
TRUST_LOW = 0.3    # poor track record -> be conservative

_ORG_INSIGHT = "organizational_intelligence"


def _org_insight(view: AdaptiveView) -> InsightRow | None:
    candidates = [
        i for i in view.insights
        if i.insight_type == _ORG_INSIGHT and i.sample_size >= MIN_SAMPLES
    ]
    if not candidates:
        return None
    # Deterministic pick: most evidence, then id.
    return sorted(candidates, key=lambda i: (-i.sample_size, i.insight_id))[0]


def _positive_rate(insight: InsightRow) -> float | None:
    pos = insight.evidence.get("positive")
    neg = insight.evidence.get("negative")
    if pos is None or neg is None or (pos + neg) == 0:
        return None
    return round(pos / (pos + neg), 4)


def _direction(rate: float) -> int:
    """+1 raise (more trust), -1 lower, 0 hold."""
    if rate >= TRUST_HIGH:
        return 1
    if rate <= TRUST_LOW:
        return -1
    return 0


def _change(view: AdaptiveView, field: str, target: float, sign: int,
            insight: InsightRow, reason: str) -> ProposedChange | None:
    current = view.active_policy[field]
    capped_value, was_capped = cap_change(field, current, target)
    if capped_value == current:
        return None  # nothing meaningful to change
    return ProposedChange(
        field=field,
        from_value=current,
        to_value=capped_value,
        direction="increase" if capped_value > current else "decrease",
        capped=was_capped,
        reason=reason,
        supporting_insight_ids=[insight.insight_id],
        supporting_learning_ids=list(insight.supporting_learning_event_ids),
    )


def propose(view: AdaptiveView) -> ProposalDraft | None:
    """Return a single deterministic proposal bundling all justified changes."""
    org = _org_insight(view)
    if org is None:
        return None
    rate = _positive_rate(org)
    if rate is None:
        return None
    sign = _direction(rate)
    if sign == 0:
        return None  # track record is ambiguous -> propose nothing

    step = 1 + sign * 0.25  # move toward the magnitude cap; cap_change enforces safety
    reason = (
        f"Organizational outcome profile is {rate:.0%} positive over {org.sample_size} "
        f"decisive outcomes ({'trust earned' if sign > 0 else 'be conservative'})."
    )

    changes: list[ProposedChange] = []
    # Rule A — brief selectivity (min confidence). Trust earned => lower the bar.
    conf = view.active_policy["brief_min_confidence"]
    a = _change(view, "brief_min_confidence", conf * (1 - sign * 0.25), -sign, org, reason)
    if a:
        changes.append(a)
    # Rule B — brief breadth (max items). Trust earned => show more.
    items = view.active_policy["brief_max_items"]
    b = _change(view, "brief_max_items", items * step, sign, org, reason)
    if b:
        changes.append(b)

    if not changes:
        return None

    proposed_payload = dict(view.active_policy)
    insight_ids: set[str] = set()
    learning_ids: set[str] = set()
    for c in changes:
        proposed_payload[c.field] = c.to_value
        insight_ids.update(c.supporting_insight_ids)
        learning_ids.update(c.supporting_learning_ids)

    return ProposalDraft(
        proposal_id=compute_proposal_id(view.org_id, view.adaptive_version),
        org_id=view.org_id,
        adaptive_version=view.adaptive_version,
        as_of=view.as_of,
        changes=sorted(changes, key=lambda c: c.field),
        proposed_payload=proposed_payload,
        supporting_insight_ids=sorted(insight_ids),
        supporting_learning_ids=sorted(learning_ids),
    )
