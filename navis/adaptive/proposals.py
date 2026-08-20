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

# The minimum-evidence threshold, from Learning's read contract rather than a
# fourth private copy of the number. Learning already applies it when deriving
# `learning_patterns` and when bounding the DECISION aggregate read, so an
# organization that is "too thin to say anything" means the same thing here as
# it does two layers upstream.
from navis.learning.metrics import MIN_SAMPLES

TRUST_HIGH = 0.6   # earned trust -> surface more, lower the bar
TRUST_LOW = 0.3    # poor track record -> be conservative

_ORG_INSIGHT = "organizational_intelligence"

# The learned evidence a rule may act on. `action_rejected` is a JUDGEMENT
# outcome — a person's verdict on something Navis proposed — and `action_executed`
# is the DELIVERY record of a proposal that survived to run. Together they are the
# acceptance rate of what Navis chose to surface, which is the most direct
# evidence the product has about its own selectivity.
_PROPOSAL_ACCEPTED = "action_executed"
_PROPOSAL_REJECTED = "action_rejected"


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


def _intelligence_changes(view: AdaptiveView) -> list[ProposedChange]:
    """Rules grounded in Intelligence's organizational outcome profile (V1)."""
    org = _org_insight(view)
    if org is None:
        return []
    rate = _positive_rate(org)
    if rate is None:
        return []
    sign = _direction(rate)
    if sign == 0:
        return []  # track record is ambiguous -> propose nothing

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
    return changes


def _judgement_changes(view: AdaptiveView) -> list[ProposedChange]:
    """Rules grounded in LEARNED evidence: what the founder actually accepted.

    THE BRIDGE THIS ADDS. Every rule above folds an Intelligence insight, which
    is a projection of Learning. This one folds Learning's own outcome counters,
    reaching the one thing no insight expresses: how often a person said NO to
    what Navis proposed.

    ONE DIRECTION ONLY, and that asymmetry is the point. A high rejection rate is
    positive evidence that Navis is surfacing things this organization does not
    want, so the rule tightens selectivity. A LOW rejection rate is not evidence
    of anything — a founder who rejects nothing may be delighted or may simply
    not be looking — so it loosens nothing. Absence of complaint is not consent.

    Returns [] rather than guessing whenever the evidence is thin, absent or
    malformed.
    """
    evidence = view.learning
    if evidence is None or evidence.org_id != view.org_id:
        # A view whose evidence belongs to another organization is a programming
        # error, not an input to reason about. Fail closed.
        return []

    accepted = evidence.by_type(_PROPOSAL_ACCEPTED)
    rejected = evidence.by_type(_PROPOSAL_REJECTED)
    if accepted is None or rejected is None:
        return []
    decided = accepted.total + rejected.total
    if decided < MIN_SAMPLES:
        return []  # too thin to say anything

    acceptance = round(accepted.total / decided, 4)
    if _direction(acceptance) != -1:
        return []  # only a POOR acceptance rate justifies acting

    reason = (
        f"The founder rejected {rejected.total} of {decided} proposals "
        f"({1 - acceptance:.0%}); Navis is surfacing too freely."
    )
    # Cite the events behind BOTH counters — the rejections and the acceptances
    # they are a rate against.
    lineage = sorted(
        {*accepted.supporting_learning_event_ids, *rejected.supporting_learning_event_ids}
    )

    changes: list[ProposedChange] = []
    for field, target in (
        # Be more selective: raise the confidence floor.
        ("brief_min_confidence", view.active_policy["brief_min_confidence"] * 1.25),
        # And show less: lower the item cap.
        ("brief_max_items", view.active_policy["brief_max_items"] * 0.75),
    ):
        current = view.active_policy[field]
        capped, was_capped = cap_change(field, current, target)
        if capped == current:
            continue
        changes.append(
            ProposedChange(
                field=field,
                from_value=current,
                to_value=capped,
                direction="increase" if capped > current else "decrease",
                capped=was_capped,
                reason=reason,
                supporting_insight_ids=[],  # learned directly, not via an insight
                supporting_learning_ids=lineage,
            )
        )
    return changes


def _resolve(candidates: list[ProposedChange]) -> list[ProposedChange]:
    """One change per field, conservatively, when two rules touch the same one.

    Both rule families tune brief selectivity, so they can meet on a field. What
    they must never do is average, alternate, or let ordering decide:

      * OPPOSITE directions => the evidence conflicts, so neither fires. "Our
        outcomes are good, therefore surface more" and "the founder rejects most
        of what we surface" cannot both be acted on, and picking one would be a
        guess.
      * SAME direction => keep the change closest to the current value. Two
        independent reasons to move the same way is not a reason to move twice
        as far.
    """
    by_field: dict[str, list[ProposedChange]] = {}
    for change in candidates:
        by_field.setdefault(change.field, []).append(change)

    resolved: list[ProposedChange] = []
    for field in sorted(by_field):
        group = by_field[field]
        if len({c.direction for c in group}) > 1:
            continue  # conflicting evidence -> no change to this field
        # Deterministic: smallest movement, then reason text to break an exact tie.
        resolved.append(
            sorted(group, key=lambda c: (abs(c.to_value - c.from_value), c.reason))[0]
        )
    return resolved


def propose(view: AdaptiveView) -> ProposalDraft | None:
    """Return a single deterministic proposal bundling all justified changes."""
    changes = _resolve([*_intelligence_changes(view), *_judgement_changes(view)])
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
