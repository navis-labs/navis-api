"""The founder-facing projection of a Decision. Deterministic, cited, additive.

WHAT THIS IS FOR. Every layer below this one is correct and none of it is
readable. A card built straight off a stored decision shows the founder
`relationship_cooling`, a rank, a severity float and a UUID, and the three
questions a CEO actually asks — "why does this matter to my organization?",
"what should I do?", "can Navis do it?" — had no backend producer at all. The
frontend has a `whyItMatters` slot that is empty on every decision in
production, and a `whatNext` that renders the headline a second time.

WHAT IT IS NOT. Not an engine. It reaches no conclusion, computes no score,
consults no clock and performs no I/O. Every string it emits is either

  * copied verbatim from a producer (the engine's `title` / `why`, the
    insight's own `title`), or
  * filled from a fixed template using values the engines already recorded in
    `evidence`,

and when neither is possible the field is absent. There is no fallback prose:
"we could not explain this" is a real answer and is strictly better than a
sentence that sounds like an explanation.

THE ONE PLACE INTERNAL VOCABULARY IS TRANSLATED. `signal_type` is a stable
persisted string and stays exactly where it is — in the decision, in the trace
view, in the tests. `kind` is its founder-facing name, and the map below is the
only translation in the system. Six signal types collapse to `category: "risk"`
and are NOT collapsed here: a customer going quiet and a promise slipping are
different things to a person, and the whole point of publishing `signal_type`
was to stop calling both of them Risk.

RELATION SEMANTICS ARE ENFORCED, NOT DOCUMENTED. Organizational context is drawn
ONLY from an insight whose relation is `derived_from_same_signal` and which
describes this decision's own snapshot. `shares_entity` means "this insight
mentions the same person" and is never rendered as a reason; a stale generation
is never rendered as current. See `decisions.base.InsightRelation`.
"""
from __future__ import annotations

import enum
from typing import Any

from navis.decisions.base import InsightRelation


class DecisionKind(str, enum.Enum):
    """What a decision IS, in the language a founder uses.

    One member per decision-worthy `SignalType`. Deliberately not fewer: the
    six types that share `DecisionCategory.RISK` mean six different things to
    the person reading them.
    """

    REPLY_NEEDED = "reply_needed"
    DECISION_NEEDED = "decision_needed"
    COMMITMENT_OVERDUE = "commitment_overdue"
    COMMITMENT_UNRESOLVED = "commitment_unresolved"
    DECISION_UNRESOLVED = "decision_unresolved"
    CONTACT_LAPSED = "contact_lapsed"
    PROJECT_INACTIVE = "project_inactive"
    CUSTOMER_QUIET = "customer_quiet"
    CALENDAR_LOAD = "calendar_load"


class _Kind:
    """How one signal type is presented, and whether it can name its subject.

    `titled` records whether the ENGINE's own title already names what the
    decision is about. Three engines emit a bare category as their title —
    "Relationship cooling", "Project stalled", "Customer engagement dropping" —
    with no subject anywhere in the string, so a founder reads a card that
    cannot say who or what it concerns. Those are the only ones this module
    re-renders, and only when `evidence` actually carries a subject.

    `next_step` is a RECOMMENDATION TEMPLATE, not a claim about the world. It
    says what a person would do about this kind of thing; it never asserts an
    outcome, a deadline or an impact.
    """

    __slots__ = ("kind", "label", "titled", "headline", "next_step", "next_step_bare")

    def __init__(
        self,
        kind: DecisionKind,
        label: str,
        *,
        titled: bool,
        headline: str = "",
        next_step: str = "",
        next_step_bare: str = "",
    ) -> None:
        self.kind = kind
        self.label = label
        self.titled = titled
        self.headline = headline
        self.next_step = next_step
        self.next_step_bare = next_step_bare or next_step


# `SignalType` values as literals, matching the discipline `intelligence.insights`
# already uses for the same vocabulary. A test asserts every key is a real
# decision-worthy SignalType and that none is missing, so the two cannot drift.
_KINDS: dict[str, _Kind] = {
    "waiting_on_user_reply": _Kind(
        DecisionKind.REPLY_NEEDED, "Reply needed", titled=True,
        next_step="Reply to {subject}, or hand the thread off.",
        next_step_bare="Reply in this thread, or hand it off.",
    ),
    "decision_owed": _Kind(
        DecisionKind.DECISION_NEEDED, "Decision needed", titled=True,
        next_step="Give {subject} an answer, or say when you will.",
        next_step_bare="Give an answer, or say when you will.",
    ),
    "commitment_follow_up": _Kind(
        DecisionKind.COMMITMENT_OVERDUE, "Commitment overdue", titled=True,
        next_step="Follow up with {subject} on what was promised.",
        next_step_bare="Follow up on what was promised.",
    ),
    "commitment_slipping": _Kind(
        DecisionKind.COMMITMENT_UNRESOLVED, "Commitment unresolved", titled=True,
        next_step="Check with {subject} whether this is still on.",
        next_step_bare="Check whether this is still on, or close it out.",
    ),
    "decision_unresolved": _Kind(
        DecisionKind.DECISION_UNRESOLVED, "Open decision", titled=True,
        next_step_bare="Close this out, or record where it stands.",
    ),
    "relationship_cooling": _Kind(
        DecisionKind.CONTACT_LAPSED, "Contact lapsed", titled=False,
        headline="Communication with {subject} has dropped off",
        next_step="Reach out to {subject}, or decide the contact can lapse.",
        next_step_bare="Reach out, or decide the contact can lapse.",
    ),
    "project_stalled": _Kind(
        DecisionKind.PROJECT_INACTIVE, "Project inactive", titled=False,
        headline="{subject} has had no activity",
        next_step="Check where {subject} stands.",
        next_step_bare="Check where this project stands.",
    ),
    "customer_engagement_drop": _Kind(
        DecisionKind.CUSTOMER_QUIET, "Customer quiet", titled=False,
        headline="{subject} has gone quiet",
        next_step="Reconnect with {subject}, or find out why they went quiet.",
        next_step_bare="Find out why this customer went quiet.",
    ),
    "meeting_overload": _Kind(
        DecisionKind.CALENDAR_LOAD, "Calendar load", titled=True,
        next_step_bare="Review the calendar and drop what is not needed.",
    ),
}


# ── Subject: WHO or WHAT this decision is about ─────────────────────────────
#
# The precedence is deliberately IDENTICAL to `intelligence.insights._label`,
# which answers the same question about the same `evidence` dict one layer up.
# The two cannot be shared — `navis/decisions/**` may not import
# `navis.intelligence` — so a test asserts they agree on every input instead of
# leaving it to luck.
#
# A DISPLAY NAME BEATS AN ADDRESS, which is the one place this differs and the
# reason it does: an insight is describing a population and an address is a
# precise identifier there, while a card is addressing a person by name.
_SUBJECT_KEYS = ("person_name", "person_email", "speaker_email", "domain",
                 "subject", "commitment", "decision")
# NEVER a subject, even though the engines put them in `evidence`: an id is not
# a name, and rendering one is the exact defect this module exists to remove.
_NEVER_A_SUBJECT = frozenset({"thread_id", "rfc_message_id", "entity_id"})


def _text(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def subject_of(evidence: dict[str, Any]) -> str | None:
    """The human name of what this decision concerns, or None.

    None is a real answer. `project_stalled` and `customer_engagement_drop`
    record no name anywhere — their engines put only an age in `evidence` — so
    there is nothing to render and this returns None rather than the entity's
    UUID. The card then reads "This project has had no activity", which is less
    specific and true, instead of naming a uuid5 at a founder.
    """
    for key in _SUBJECT_KEYS:
        if key in _NEVER_A_SUBJECT:
            continue
        value = _text(evidence.get(key))
        if value:
            return value
    return None


# ── Evidence: the facts a person can check ─────────────────────────────────
#
# Only keys the engines actually write, each with the label a person would use.
# Anything absent from this map stays out of the founder-facing list and remains
# in `decision.evidence` for the trace view — an unlabelled key is by definition
# one nobody decided how to present.
_EVIDENCE_LABELS: dict[str, str] = {
    "subject": "Subject",
    "commitment": "What was promised",
    "decision": "What was asked",
    "person_name": "Person",
    "person_email": "Email",
    "speaker_email": "Promised by",
    "due_text": "Due",
    "domain": "Company",
}
# Values the engines record as ISO-8601 instants. Shown as the calendar DATE
# they carry and nothing more: a founder reading "they last wrote" wants a day,
# and rendering `2026-08-17T09:00:00+00:00` on a card is the same category of
# leak as rendering a UUID. The date is taken from the offset the producer
# recorded — no timezone is applied, because applying one would need a viewer
# locale this layer does not have and must not guess.
_DATE_LABELS: dict[str, str] = {
    "due_at": "Due date",
    "last_inbound_at": "They last wrote",
    "asked_at": "Asked on",
    "promised_at": "Promised on",
}
# Day counts, which read as a duration rather than a value. The engines each
# named theirs differently; the founder sees one phrasing.
_DAY_LABELS: dict[str, str] = {
    "waiting_days": "Waiting",
    "overdue_days": "Overdue",
    "age_days": "Open",
    "days_since_contact": "No contact for",
    "days_since_activity": "No activity for",
    "days_since": "Quiet for",
}


def _days(value: Any) -> str | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    whole = round(float(value))
    return f"{whole} day" if whole == 1 else f"{whole} days"


def _date(value: Any) -> str | None:
    """The calendar date an ISO-8601 value names, verbatim. No conversion.

    Returns None for anything that is not a date-leading string, so a malformed
    or already-humanised value is dropped rather than half-rendered.
    """
    text = _text(value)
    if not text or len(text) < 10:
        return None
    head = text[:10]
    parts = head.split("-")
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        return None
    return head


def evidence_items(evidence: dict[str, Any]) -> list[dict[str, str]]:
    """The decision's own evidence, labelled, in a fixed order.

    Every item names the `field` it came from, so a reader can always get back
    to the raw value the engine recorded. Nothing is computed, converted or
    combined — a value the engine did not write does not appear.
    """
    items: list[dict[str, str]] = []
    for key, label in _EVIDENCE_LABELS.items():
        value = _text(evidence.get(key))
        if value:
            items.append({"field": key, "label": label, "value": value})
    for key, label in _DATE_LABELS.items():
        value = _date(evidence.get(key))
        if value:
            items.append({"field": key, "label": label, "value": value})
    for key, label in _DAY_LABELS.items():
        value = _days(evidence.get(key))
        if value:
            items.append({"field": key, "label": label, "value": value})
    return items


# ── Organizational context: what Intelligence adds, and only when it may ────


# THE INSIGHT FAMILY THAT CONNECTS RATHER THAN COUNTS.
#
# A literal, not an import: `navis/decisions/**` may not import
# `navis.intelligence` (AST-pinned, tests/decisions/test_decision_semantics.py).
# It is safe to compare because `SupportingInsight.insight_type` exists for
# exactly this — it carries the family "so a client can name the FAMILY of
# evidence without a second lookup" — and it is pinned equal to
# `InsightType.SITUATION.value` by `tests/decisions/test_explanation.py`, which
# may import both layers.
_SITUATION_INSIGHT_TYPE = "situation_intelligence"


def _organizational_context(decision: dict[str, Any]) -> dict[str, Any] | None:
    """The one insight that may be presented as WHY this matters org-wide.

    THREE CONDITIONS, all read from stored provenance:

      1. `relation == derived_from_same_signal` — the insight was folded from
         the very signal this decision ranks. `shares_entity` is topical overlap
         ("this also mentions Nick") and `learned_from_outcomes` is a
         measurement over past executions; presenting either as the reason a
         decision exists is the causal overclaim the relation vocabulary was
         built to prevent.
      2. `describes_current_snapshot is True` — the insight describes THIS
         snapshot. A stale generation still matches by signal id, because
         `compute_signal_id` does not hash the watermark.
      3. it has a readable title.

    A SITUATION OUTRANKS A COUNT, and that preference is the difference between
    organizational context and a tally. Every other reality family answers "how
    many things of this kind are there?" — "1 open commitment slipping risk" —
    which restates the decision the founder is already reading, one altitude up.
    A situation answers "what is going on with this counterparty?", names the
    other conclusions about them, and is the only family whose claim the founder
    could not have made themselves from the card. When both qualify (they
    routinely do — both are folded from this decision's own signal), the
    connected one is strictly the more informative answer to "why does this
    matter", so it wins. Order is otherwise unchanged.

    The insight's title is passed through VERBATIM. Rewording it would give one
    organizational fact two voices, and the summary is deliberately not used —
    a reality summary quotes the engine's measurements ("peak severity 0.65"),
    which belongs in a trace view and not on a founder's screen.

    None when no insight qualifies, which is the common case.
    """
    qualifying = [
        raw
        for raw in decision.get("supporting_insights") or ()
        if isinstance(raw, dict)
        and raw.get("relation") == InsightRelation.DERIVED_FROM_SAME_SIGNAL.value
        and raw.get("describes_current_snapshot") is True
        and _text(raw.get("title"))
    ]
    if not qualifying:
        return None
    # Stable: `sorted` keeps the incoming order within each group, so among
    # several situations (or none) the first still wins exactly as before.
    chosen = sorted(
        qualifying,
        key=lambda raw: raw.get("insight_type") != _SITUATION_INSIGHT_TYPE,
    )[0]
    return {
        "statement": _text(chosen.get("title")),
        "insight_id": chosen.get("insight_id"),
        "insight_type": chosen.get("insight_type"),
        "affected": chosen.get("sample_size"),
    }


# ── The projection ──────────────────────────────────────────────────────────


def explain(decision: dict[str, Any]) -> dict[str, Any]:
    """One stored decision, as a founder-facing explanation. PURE.

    Input is the decision dict exactly as `daily_briefs.decisions` stores it
    (optionally already enriched with `action`, which this reads and never
    computes). Output is additive: nothing here replaces a stored field, and
    every stored field stays exactly where it was for the trace view.

    A decision whose `signal_type` this module has no presentation for gets an
    `explainable: false` explanation carrying its provenance and nothing else.
    That is the honest state for a vocabulary that moved ahead of this map, and
    it is what stops a new signal type from silently rendering as a blank card.
    """
    evidence = decision.get("evidence") or {}
    signal_type = decision.get("signal_type")
    spec = _KINDS.get(str(signal_type)) if signal_type else None
    subject = subject_of(evidence)
    action = decision.get("action")

    provenance = {
        "decision_id": decision.get("decision_id"),
        "signal_type": signal_type,
        "reasoning_signal_ids": list(decision.get("reasoning_signal_ids") or ()),
        "entity_type": decision.get("entity_type"),
        "entity_id": decision.get("entity_id"),
        "state_watermark": decision.get("state_watermark"),
        "reasoning_version": decision.get("reasoning_version"),
        "supporting_insight_ids": list(decision.get("supporting_insight_ids") or ()),
    }

    if spec is None:
        # No presentation exists for this conclusion. Say so; do not guess.
        return {
            "explainable": False,
            "kind": None,
            "kind_label": None,
            "headline": _text(decision.get("title")),
            "why": _text(decision.get("why")),
            "subject": None,
            "evidence": evidence_items(evidence),
            "organizational_context": None,
            "next_step": None,
            "can_navis_act": bool(action),
            "provenance": provenance,
        }

    # The engine's own title, unless the engine emitted a bare category and the
    # evidence can name what it is about. See `_Kind.titled`.
    headline = _text(decision.get("title"))
    if not spec.titled and subject and spec.headline:
        headline = spec.headline.format(subject=subject)

    next_step = (
        spec.next_step.format(subject=subject)
        if subject and spec.next_step
        else spec.next_step_bare
    ) or None

    return {
        "explainable": True,
        # Founder vocabulary. `signal_type` stays in `provenance` for the trace.
        "kind": spec.kind.value,
        "kind_label": spec.label,
        # WHAT is happening.
        "headline": headline,
        # WHY, in the detecting engine's own words — it quotes the message, the
        # promise or the elapsed days it actually measured.
        "why": _text(decision.get("why")),
        # WHO or WHAT this is about, with the ids a client navigates by.
        "subject": (
            {
                "label": subject,
                "email": _text(evidence.get("person_email"))
                or _text(evidence.get("speaker_email")),
                "entity_type": decision.get("entity_type"),
                "entity_id": decision.get("entity_id"),
                "thread_id": _text(evidence.get("thread_id")),
            }
            if subject or decision.get("entity_id")
            else None
        ),
        # The facts behind the conclusion, labelled.
        "evidence": evidence_items(evidence),
        # WHY IT MATTERS to the organization — cited, or absent.
        "organizational_context": _organizational_context(decision),
        # WHAT to do. A recommendation, never an assertion about the outcome.
        "next_step": next_step,
        # CAN NAVIS DO IT. Mirrors `action` exactly; the action block itself
        # carries provider, approval and risk. False means this is a finding
        # for a person to judge, and no execute affordance may be shown.
        "can_navis_act": bool(action),
        "provenance": provenance,
    }
