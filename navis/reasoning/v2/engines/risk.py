"""Entity interpretation engines. Each owns its own judgment (slipping/cooling/
stalled/unresolved/...), computed from State V2 raw facts + the projection clock.
No thresholds live in State anymore — they live here and are versioned by
REASONING_VERSION.
"""
from __future__ import annotations

import re
from html import unescape

from navis.reasoning.v2.base import ReasoningContext, ReasoningEngine, days_between
from navis.reasoning.v2.contract import SignalDraft, SignalType
from navis.state.v2.models import OrganizationalState

# Versioned thresholds (bump REASONING_VERSION when these change).
COMMITMENT_SLIPPING_DAYS = 5.0
RELATIONSHIP_COOLING_DAYS = 7.0
PROJECT_STALLED_DAYS = 10.0
DECISION_UNRESOLVED_DAYS = 7.0
CUSTOMER_DROP_DAYS = 14.0
MEETING_OVERLOAD_COUNT = 5

# ── Upper age caps ─────────────────────────────────────────────────────
#
# There is no single correct ceiling, because the signals measure two
# different things and only one of them decays.
#
# DECAY signals measure SILENCE — relationship cooling, project stalled,
# customer engagement dropping. Silence genuinely stops being "cooling" at some
# point and becomes a lapsed relationship: a different, lower-urgency fact that
# a daily brief should not keep re-raising. One month is the founder's
# actionable window here, and past it the item is history they can neither
# salvage nor act on quickly.
#
# OBLIGATION signals measure an UNMET DUTY — a promise someone made, a decision
# you owe, a reply you never sent. These do NOT decay. "You promised me the deck
# 45 days ago" is more urgent than at 5 days, not less, and a Chief of Staff
# that silently forgets promises past an arbitrary horizon has the single worst
# failure mode this product can have. The 30-day cap applied to commitments was
# therefore a real (if quiet) regression — it made Navis forget.
#
# The obligation ceiling is still BOUNDED rather than absent: the scan stays
# sane, and a board carrying a year of debt is its own kind of useless. A
# quarter is long enough that nothing a founder would still act on is dropped.
MAX_DECAY_AGE_DAYS = 30.0
MAX_OBLIGATION_AGE_DAYS = 90.0

# Back-compat alias for the decay window. Kept so existing imports keep meaning
# what they always meant; new code should name the window it wants.
MAX_SURFACE_AGE_DAYS = MAX_DECAY_AGE_DAYS

# Defensive scrub for legacy Memory rows written before the extractor learned
# to strip HTML from Gmail bodies (extractors/email.py). New rows are already
# clean; this exists so 5+ day old commitments don't render <td> markup in
# today's brief title.
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _plain(text: str | None) -> str:
    if not text:
        return ""
    stripped = _HTML_TAG_RE.sub(" ", text)
    return re.sub(r"\s+", " ", unescape(stripped)).strip()


def _clip(text: str | None, n: int = 80) -> str:
    text = _plain(text).replace("\n", " ")
    return text if len(text) <= n else text[: n - 1] + "…"


class CommitmentReasoningEngine(ReasoningEngine):
    name = "commitment_reasoning"
    version = "2.0.0"

    def analyze(self, state: OrganizationalState, ctx: ReasoningContext) -> list[SignalDraft]:
        out = []
        for c in state.commitments():
            if c.status != "open":
                continue
            # A dated promise belongs to CommitmentFollowUpEngine, which can say
            # WHEN it was due. Skipping it here keeps the two mutually exclusive
            # so one promise never produces two cards.
            if c.due_at:
                continue
            age = days_between(ctx.as_of, c.last_seen)
            if age is None or age < COMMITMENT_SLIPPING_DAYS or age > MAX_OBLIGATION_AGE_DAYS:
                continue
            clean = _plain(c.text)
            if not clean:
                # Legacy row whose entire `text` was HTML markup — nothing
                # meaningful to surface. Skip rather than showing an empty
                # "Commitment slipping: " card. New rows never hit this path.
                continue
            out.append(
                SignalDraft(
                    signal_type=SignalType.COMMITMENT_SLIPPING,
                    entity_type="commitment",
                    entity_id=c.entity_id,
                    title=f"Commitment slipping: {_clip(c.text)}",
                    why=f"Open for {age:.0f} days with no newer fact in Memory.",
                    confidence=min(0.9, 0.5 + age / 30),
                    severity=min(0.9, 0.4 + age / 30),
                    dedup_key=c.entity_id,
                    evidence={"age_days": round(age, 2), "speaker_email": c.speaker_email},
                )
            )
        return out


class RelationshipReasoningEngine(ReasoningEngine):
    name = "relationship_reasoning"
    version = "2.0.0"

    def analyze(self, state: OrganizationalState, ctx: ReasoningContext) -> list[SignalDraft]:
        out = []
        for r in state.relationships():
            age = days_between(ctx.as_of, r.last_interaction_at)
            if age is None or age < RELATIONSHIP_COOLING_DAYS or age > MAX_SURFACE_AGE_DAYS:
                continue
            out.append(
                SignalDraft(
                    signal_type=SignalType.RELATIONSHIP_COOLING,
                    entity_type="relationship",
                    entity_id=r.entity_id,
                    title="Relationship cooling",
                    why=f"{age:.0f} days since the last interaction.",
                    confidence=min(0.9, 0.5 + age / 30),
                    severity=0.6,
                    dedup_key=r.entity_id,
                    evidence={"days_since_contact": round(age, 2), "person_email": r.person_email},
                )
            )
        return out


class ProjectReasoningEngine(ReasoningEngine):
    name = "project_reasoning"
    version = "2.0.0"

    def analyze(self, state: OrganizationalState, ctx: ReasoningContext) -> list[SignalDraft]:
        out = []
        for p in state.projects():
            age = days_between(ctx.as_of, p.last_seen)
            if age is None or age < PROJECT_STALLED_DAYS or age > MAX_SURFACE_AGE_DAYS:
                continue
            out.append(
                SignalDraft(
                    signal_type=SignalType.PROJECT_STALLED,
                    entity_type="project",
                    entity_id=p.entity_id,
                    title="Project stalled",
                    why=f"No document activity for {age:.0f} days.",
                    confidence=min(0.85, 0.4 + age / 40),
                    severity=0.55,
                    dedup_key=p.entity_id,
                    evidence={"days_since_activity": round(age, 2)},
                )
            )
        return out


class CustomerReasoningEngine(ReasoningEngine):
    name = "customer_reasoning"
    version = "2.0.0"

    def analyze(self, state: OrganizationalState, ctx: ReasoningContext) -> list[SignalDraft]:
        out = []
        for cu in state.customers():
            age = days_between(ctx.as_of, cu.last_seen)
            if age is None or age < CUSTOMER_DROP_DAYS or age > MAX_SURFACE_AGE_DAYS:
                continue
            out.append(
                SignalDraft(
                    signal_type=SignalType.CUSTOMER_ENGAGEMENT_DROP,
                    entity_type="customer",
                    entity_id=cu.entity_id,
                    title="Customer engagement dropping",
                    why=f"{age:.0f} days since the last customer signal.",
                    confidence=min(0.85, 0.5 + age / 30),
                    severity=0.7,
                    dedup_key=cu.entity_id,
                    evidence={"days_since": round(age, 2)},
                )
            )
        return out


class DecisionReasoningEngine(ReasoningEngine):
    name = "decision_reasoning"
    version = "2.0.0"

    def analyze(self, state: OrganizationalState, ctx: ReasoningContext) -> list[SignalDraft]:
        out = []
        for d in state.decisions():
            if d.status != "open":
                continue
            # Decision REQUESTS ("can you approve X?") are DecisionOwedEngine's
            # job — it knows who is waiting and can address the follow-up. This
            # engine covers decisions the org RECORDED and never closed out.
            if d.kind == "request":
                continue
            age = days_between(ctx.as_of, d.last_seen)
            if age is None or age < DECISION_UNRESOLVED_DAYS or age > MAX_OBLIGATION_AGE_DAYS:
                continue
            out.append(
                SignalDraft(
                    signal_type=SignalType.DECISION_UNRESOLVED,
                    entity_type="decision",
                    entity_id=d.entity_id,
                    title=f"Decision unresolved: {_clip(d.text)}",
                    why=f"Open for {age:.0f} days with no resolving fact.",
                    confidence=min(0.85, 0.4 + age / 30),
                    severity=0.65,
                    dedup_key=d.entity_id,
                    evidence={"age_days": round(age, 2)},
                )
            )
        return out


class MeetingReasoningEngine(ReasoningEngine):
    name = "meeting_reasoning"
    version = "2.0.0"

    def analyze(self, state: OrganizationalState, ctx: ReasoningContext) -> list[SignalDraft]:
        active = [m for m in state.meetings() if m.status != "cancelled"]
        if len(active) < MEETING_OVERLOAD_COUNT:
            return []
        return [
            SignalDraft(
                signal_type=SignalType.MEETING_OVERLOAD,
                entity_type=None,
                entity_id=None,
                title="Meeting overload",
                why=f"{len(active)} active meetings currently in the org's state.",
                confidence=0.75,
                severity=min(0.9, len(active) / 10),
                dedup_key="meeting_overload",
                evidence={"active_meetings": len(active)},
            )
        ]
