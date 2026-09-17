"""Engine base + context. The state's `as_of` is the only clock.

`ReasoningContext` carries the projection's logical clock and provenance so engines
can compute time-deltas deterministically (as_of - last_seen) without ever touching
datetime.now(). A `prior` OrganizationalState (also State V2) may be supplied for
trend reasoning — still State-only, never Memory/Signals.
"""
from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime

from navis.reasoning.v2.contract import SignalDraft
from navis.state.v2.models import OrganizationalState


@dataclass(frozen=True, slots=True)
class ReasoningContext:
    org_id: uuid.UUID
    as_of: datetime | None
    watermark: str | None
    reasoning_version: int
    prior: OrganizationalState | None = None
    # entity_id -> the SET of `signal_importance.classification` values observed
    # through that entity's memory lineage. Supplied by the CALLER, exactly like
    # `prior`: the composition root (orchestration / api) does the read, so
    # Reasoning itself still queries nothing and stays a pure function of its
    # inputs. Replaying the same (state, evidence) reproduces byte-identical
    # signals.
    #
    # An entity ABSENT from this map is UNKNOWN — never "human", never
    # "automated". Engines must preserve their existing behaviour for it rather
    # than manufacture or withhold relevance on missing evidence.
    evidence: Mapping[str, frozenset[str]] = field(default_factory=dict)


# ── Organizational relevance ──────────────────────────────────────────────────
#
# THE ONE definition of "does this represent a real human/business counterparty?",
# shared by every engine that needs it. It lived as a private constant inside
# `engines/meta.py` and served OpportunityEngine alone, which is why
# RelationshipReasoningEngine emitted "relationship cooling" for
# microsoft-noreply@microsoft.com, mail@mail.adobe.com and two Substack
# newsletters in production — and why Intelligence then aggregated those into
# organizational RISK insights.
#
# These are `navis.signals.importance.ImportanceClass` values written as
# literals, because `navis/reasoning/v2/**` may not import `navis.signals` (an
# AST invariant — see tests/reasoning/test_reasoning_v2.py). That vocabulary is
# declared "stable strings — they are persisted and queried", so this consumes a
# published taxonomy rather than forking one. Drift is not left to luck:
# tests/reasoning/test_opportunity_relevance.py asserts every member is a real
# ImportanceClass and fails if either side moves.
#
# `meeting` is the load-bearing member. Google Calendar sends an RSVP ON BEHALF
# of a human attendee, stamping `Auto-Submitted: auto-generated` while keeping
# that person's address in `From:`. Memory's `is_automated` is therefore True and
# `category` is "automated" — both correct about the MESSAGE, both wrong as a
# statement about the PERSON. Deciding from the accumulated classification SET
# (monotone: evidence is added, never retracted) is what makes the answer
# independent of which message happened to arrive last.
#
# Everything absent here — invoice, marketing, newsletter, automated_other, otp,
# social, spam, dev, document — stays first-class shared evidence for other
# engines. A billing- or security-oriented engine may legitimately treat
# `invoice` as its primary input. It simply is not evidence of a human
# relationship.
HUMAN_BEARING_EVIDENCE = frozenset({
    "commitment",
    "founder_decision",
    "human_reply",
    "action_request",
    "meeting",
    "human_other",
})


def has_human_bearing_evidence(observed: frozenset[str] | None) -> bool:
    """POSITIVE evidence that a human counterparty is on the other end.

    UNKNOWN (absent or empty evidence) is False — this asks "did we observe a
    human?", and we did not. Use for a claim that REQUIRES a real counterparty
    to be meaningful.
    """
    return bool(observed) and bool(observed & HUMAN_BEARING_EVIDENCE)


def is_observed_non_human(observed: frozenset[str] | None) -> bool:
    """POSITIVE evidence that nothing human has ever been observed here.

    UNKNOWN is False — missing data must never silently mute a real
    relationship. Use to SUPPRESS on evidence of irrelevance while leaving
    unobserved entities alone.

    The two predicates are deliberately not each other's negation, and the
    asymmetry is the point: an engine chooses which way UNKNOWN should fall
    based on what a false positive costs it. See their call sites.
    """
    return bool(observed) and not (observed & HUMAN_BEARING_EVIDENCE)


def days_between(as_of: datetime | None, last_seen_iso: str | None) -> float | None:
    """Deterministic age in days using the projection clock (NOT now())."""
    if as_of is None or not last_seen_iso:
        return None
    try:
        last = datetime.fromisoformat(last_seen_iso)
    except (TypeError, ValueError):
        return None
    return (as_of - last).total_seconds() / 86400.0


class ReasoningEngine(ABC):
    name: str
    version: str

    @abstractmethod
    def analyze(self, state: OrganizationalState, ctx: ReasoningContext) -> list[SignalDraft]:
        """Pure: OrganizationalState -> SignalDraft[]. No I/O, no clock, no LLM."""
        raise NotImplementedError
