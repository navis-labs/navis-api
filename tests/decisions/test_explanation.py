"""A founder must be able to read a decision without reading the architecture.

Every layer below the explanation is correct and none of it is legible. A card
built straight off a stored decision shows `relationship_cooling`, a severity
float, a priority rank and a uuid5 — and the three questions a CEO actually asks
had no backend producer at all:

  * WHY does this matter to my organization?  The frontend's `whyItMatters` slot
    reads `decisionDetail`, which `/brief/today` does not serve. Empty on every
    decision in production.
  * WHAT should I do?  `whatNext` was assigned `d.action`, i.e. the headline
    rendered a second time.
  * WHO is this about?  Three engines emit a bare category as their title —
    "Relationship cooling", "Project stalled", "Customer engagement dropping" —
    with no subject anywhere in the string.

These pin the projection that answers them, and the two rules that keep it
honest: every string is copied from a producer or filled from a fixed template
with values a producer recorded, and a field with no grounded value is ABSENT
rather than filled with something that sounds like an answer.
"""
from __future__ import annotations

from typing import Any

import pytest

from navis.decisions.base import InsightRelation
from navis.decisions.explanation import (
    DecisionKind,
    _KINDS,
    evidence_items,
    explain,
    subject_of,
)

WATERMARK = "0198f2aa-4c31-7d2e-8b55-2a9c6e13f004"
SIGNAL = "36742121-349a-5236-9770-0b5d3f4d944c"
INSIGHT = "81ddb4cb-5307-509a-97ea-85ff834adf94"


def decision(**over: Any) -> dict[str, Any]:
    """A stored decision, exactly as `daily_briefs.decisions` holds one."""
    d: dict[str, Any] = {
        "decision_id": "2984d3d3-527e-50e3-ad99-a9a6c1e53556",
        "title": "Nick Alvarez is waiting for your reply",
        "why": 'They emailed "Re: pricing" 6 days ago and there is no reply from you.',
        "category": "waiting_reply",
        "signal_type": "waiting_on_user_reply",
        "priority_rank": 1,
        "confidence": 0.9,
        "severity": 0.65,
        "entity_type": "thread",
        "entity_id": "93f95896-75b2-50f4-a9c5-19819807edc5",
        "reasoning_signal_ids": [SIGNAL],
        "reasoning_version": 4,
        "state_watermark": WATERMARK,
        "created_at": "2026-08-23T09:00:00+00:00",
        "supporting_insight_ids": [],
        "supporting_insights": [],
        "evidence": {
            "person_email": "nick@acme.com",
            "person_name": "Nick Alvarez",
            "thread_id": "19fdb57e89581d43",
            "rfc_message_id": "<abc@mail.gmail.com>",
            "subject": "Re: pricing",
            "waiting_days": 6.0,
            "last_inbound_at": "2026-08-17T09:00:00+00:00",
        },
        "action": None,
    }
    d.update(over)
    return d


def insight(**over: Any) -> dict[str, Any]:
    ref = {
        "insight_id": INSIGHT,
        "relation": InsightRelation.DERIVED_FROM_SAME_SIGNAL.value,
        "insight_type": "obligation_intelligence",
        "title": "1 outstanding reply obligation",
        "sample_size": 1,
        "state_watermark": WATERMARK,
        "describes_current_snapshot": True,
    }
    ref.update(over)
    return ref


# ── 4. A signal type is never rendered as generic Risk ─────────────────────


def test_the_vocabulary_covers_every_decision_worthy_conclusion_and_no_more():
    """A conclusion with no presentation renders as a blank card. A presentation
    for a conclusion that cannot be decided is dead vocabulary."""
    from navis.decisions.base import _CATEGORY_FOR_SIGNAL

    assert set(_KINDS) == {t.value for t in _CATEGORY_FOR_SIGNAL}


@pytest.mark.parametrize(
    "signal_type,kind",
    [
        ("waiting_on_user_reply", DecisionKind.REPLY_NEEDED),
        ("decision_owed", DecisionKind.DECISION_NEEDED),
        ("commitment_follow_up", DecisionKind.COMMITMENT_OVERDUE),
        ("commitment_slipping", DecisionKind.COMMITMENT_UNRESOLVED),
        ("decision_unresolved", DecisionKind.DECISION_UNRESOLVED),
        ("relationship_cooling", DecisionKind.CONTACT_LAPSED),
        ("project_stalled", DecisionKind.PROJECT_INACTIVE),
        ("customer_engagement_drop", DecisionKind.CUSTOMER_QUIET),
        ("meeting_overload", DecisionKind.CALENDAR_LOAD),
    ],
)
def test_each_conclusion_keeps_its_own_identity(signal_type, kind):
    """Six of these share `category: "risk"`. A customer going quiet and a
    promise slipping are different things to the person reading them, and
    collapsing both into "Risk" is what publishing `signal_type` was for."""
    got = explain(decision(signal_type=signal_type))
    assert got["kind"] == kind.value
    assert got["kind_label"] and got["kind_label"][0].isupper()


def _rendered_text(explanation: dict[str, Any]) -> str:
    """Every string a founder actually READS. Citation ids and type names are
    machine keys a client navigates by, not text on the card, so they are not
    part of this and are asserted separately."""
    parts = [
        explanation.get("headline"),
        explanation.get("why"),
        explanation.get("next_step"),
        explanation.get("kind_label"),
        (explanation.get("organizational_context") or {}).get("statement"),
    ]
    parts += [f"{i['label']} {i['value']}" for i in explanation.get("evidence") or ()]
    subject = explanation.get("subject") or {}
    parts += [subject.get("label")]
    return " ".join(p for p in parts if p).lower()


def test_internal_vocabulary_never_reaches_text_a_founder_reads():
    """`signal_type`, `severity`, `state_watermark` and friends belong in the
    trace view. They may be carried as machine keys under `provenance` and the
    citation, and they may never appear in a sentence."""
    got = explain(decision(supporting_insights=[insight()]))
    text = _rendered_text(got)
    for internal in (
        "signal_type", "state_watermark", "severity", "priority_rank",
        "reasoning_version", "context_rank", "supporting_reasoning_signal_ids",
        "entity_id", "reasoning_signal", "insight_type", "_intelligence",
        "waiting_on_user_reply", "relationship_cooling", "contextentityrow",
    ):
        assert internal not in text, f"`{internal}` leaked into founder-facing text"


# ── 2. No UUID where a real label exists ───────────────────────────────────


def test_a_bare_category_title_is_re_rendered_with_the_subject_it_names():
    """`relationship_cooling` emits the title "Relationship cooling" and puts
    `person_email` in evidence. The founder was shown the category."""
    got = explain(
        decision(
            signal_type="relationship_cooling",
            title="Relationship cooling",
            why="21 days since the last interaction.",
            entity_type="relationship",
            entity_id="rel-1",
            evidence={"days_since_contact": 21.0, "person_email": "amy@beta.io"},
        )
    )
    assert got["headline"] == "Communication with amy@beta.io has dropped off"
    assert got["subject"]["label"] == "amy@beta.io"


def test_a_title_that_already_names_its_subject_is_left_exactly_alone():
    """The detecting engine's voice is the one that quoted the message. Where it
    already names the subject, nothing here re-words it."""
    got = explain(decision())
    assert got["headline"] == "Nick Alvarez is waiting for your reply"
    assert got["why"] == decision()["why"]


def test_no_subject_produces_no_subject_never_an_id():
    """`project_stalled` records only an age. There is no name to show, so the
    card degrades to a true sentence rather than rendering a uuid5."""
    got = explain(
        decision(
            signal_type="project_stalled",
            title="Project stalled",
            why="No document activity for 33 days.",
            entity_type="project",
            entity_id="93f95896-75b2-50f4-a9c5-19819807edc5",
            evidence={"days_since_activity": 33.0},
        )
    )
    assert got["headline"] == "Project stalled"
    assert got["subject"]["label"] is None
    assert got["next_step"] == "Check where this project stands."
    # The id is still reachable for navigation — it is simply never TEXT.
    assert got["subject"]["entity_id"] == "93f95896-75b2-50f4-a9c5-19819807edc5"
    for value in (got["headline"], got["next_step"], got["why"]):
        assert "93f95896" not in (value or "")


def test_an_id_is_never_promoted_to_a_subject():
    """`thread_id` and `rfc_message_id` are in `evidence` and are identifiers.
    Neither is a name, and rendering one is the defect this removes."""
    assert subject_of({"thread_id": "19fdb", "rfc_message_id": "<a@b>"}) is None


def test_a_display_name_is_preferred_over_an_address():
    assert subject_of({"person_email": "n@acme.com", "person_name": "Nick"}) == "Nick"


def test_subject_resolution_agrees_with_intelligence_labelling():
    """Two layers answer "what is this about?" over the SAME evidence dict and
    cannot share code — `navis/decisions/**` may not import
    `navis.intelligence`. So they are asserted to agree instead."""
    from navis.intelligence.insights import _label

    class _Sig:
        def __init__(self, evidence):
            self.evidence = evidence
            self.entity_id = "ent-1"
            self.signal_type = "x"

    for evidence in (
        {"person_email": "a@b.com"},
        {"speaker_email": "s@b.com"},
        {"domain": "acme.com"},
        {"subject": "Re: pricing"},
        {"commitment": "I'll send the deck"},
        {"decision": "ship or hold?"},
    ):
        assert subject_of(evidence) == _label(_Sig(evidence))


# ── 3 + 11. No unsupported claim; missing evidence is honest ───────────────


def test_no_role_company_or_customer_is_ever_inferred():
    """"customer", "partner", "investor", "employee" are organizational claims.
    Nothing here establishes one, so nothing here may say one."""
    text = _rendered_text(explain(decision(supporting_insights=[insight()])))
    for claim in ("customer", "partner", "investor", "employee", "vendor", "client"):
        assert claim not in text


def test_a_conclusion_with_no_presentation_says_so_rather_than_guessing():
    """A new SignalType that outruns this map must not render as a blank card
    with confident-looking empty fields."""
    got = explain(decision(signal_type="some_future_signal"))
    assert got["explainable"] is False
    assert got["kind"] is None and got["kind_label"] is None
    assert got["next_step"] is None
    # Provenance survives regardless — the trace view still works.
    assert got["provenance"]["reasoning_signal_ids"] == [SIGNAL]


def test_evidence_only_reports_values_a_producer_actually_recorded():
    """Nothing is computed, combined or converted. An absent key is absent."""
    items = evidence_items({"person_email": "n@acme.com"})
    assert [i["field"] for i in items] == ["person_email"]
    assert evidence_items({}) == []


def test_every_evidence_item_names_the_field_it_came_from():
    """The founder sees a label; a reader checking the claim can always get back
    to the raw value the engine wrote."""
    for item in explain(decision())["evidence"]:
        assert item["field"] in decision()["evidence"]
        assert item["label"] and item["value"]


# ── 13. No confidence becomes a percentage ─────────────────────────────────


def test_confidence_and_severity_are_not_rendered_at_all():
    """They are the engine's own 0..1 measures, not calibrated probabilities.
    A frontend that had them turned rank into a percentage; the fix is to not
    put them on the card in any form."""
    got = explain(decision(supporting_insights=[insight()]))
    text = _rendered_text(got)
    for banned in ("0.9", "0.65", "90%", "65%", "confidence", "severity", "%"):
        assert banned not in text


# ── 5 + 6. Relation semantics ──────────────────────────────────────────────


def test_organizational_context_comes_only_from_the_signal_it_was_folded_from():
    got = explain(decision(supporting_insights=[insight()]))
    context = got["organizational_context"]
    assert context["statement"] == "1 outstanding reply obligation"
    assert context["insight_id"] == INSIGHT
    assert context["affected"] == 1


def test_shares_entity_is_never_presented_as_a_reason():
    """"This insight also mentions Nick" is topical overlap. Rendering it as why
    a decision exists is the causal overclaim the relation vocabulary prevents."""
    got = explain(
        decision(supporting_insights=[insight(relation=InsightRelation.SHARES_ENTITY.value)])
    )
    assert got["organizational_context"] is None


def test_learned_from_outcomes_is_never_presented_as_a_reason():
    """A Learning insight measures what happened when the org ACTED on signals
    like this one. It is not a statement about what is true now."""
    got = explain(
        decision(
            supporting_insights=[
                insight(
                    relation=InsightRelation.LEARNED_FROM_OUTCOMES.value,
                    insight_type="signal_intelligence",
                    describes_current_snapshot=None,
                )
            ]
        )
    )
    assert got["organizational_context"] is None


def test_a_stale_generation_is_never_presented_as_current_context():
    """`compute_signal_id` does not hash the watermark, so an older generation
    still matches by signal id. It must not speak for this snapshot."""
    got = explain(
        decision(supporting_insights=[insight(describes_current_snapshot=False)])
    )
    assert got["organizational_context"] is None


def test_an_insight_with_no_readable_title_is_not_cited():
    """A citation nobody can read is a UUID on a card."""
    got = explain(decision(supporting_insights=[insight(title="")]))
    assert got["organizational_context"] is None


# ── 7 + 8. Execution affordance ────────────────────────────────────────────


def test_no_action_means_no_execution_affordance():
    """A finding is for a person to judge. `can_navis_act` false is what stops a
    card offering a button that would fail."""
    got = explain(decision(signal_type="relationship_cooling", action=None))
    assert got["can_navis_act"] is False


def test_an_action_is_mirrored_never_recomputed():
    """The explanation reads the `action` block; it never decides executability
    itself. One derivation — `execution.proposer.decision_action` — or the card
    and the worker can disagree."""
    from navis.execution.proposer import decision_action

    d = decision()
    action = decision_action(d)
    assert action is not None and action.requires_approval is True
    got = explain({**d, "action": action.to_dict()})
    assert got["can_navis_act"] is True
    # The explanation adds no second opinion about the action.
    assert "provider" not in repr(got["next_step"])


# ── 12. Provenance survives ────────────────────────────────────────────────


def test_every_explanation_carries_the_chain_back_to_its_producers():
    got = explain(decision(supporting_insights=[insight()], supporting_insight_ids=[INSIGHT]))
    p = got["provenance"]
    assert p["decision_id"] == decision()["decision_id"]
    assert p["signal_type"] == "waiting_on_user_reply"
    assert p["reasoning_signal_ids"] == [SIGNAL]
    assert p["entity_type"] == "thread" and p["entity_id"]
    assert p["state_watermark"] == WATERMARK
    assert p["reasoning_version"] == 4
    assert p["supporting_insight_ids"] == [INSIGHT]


def test_the_projection_is_pure_and_deterministic():
    """Same decision in, byte-identical explanation out — no clock, no ordering
    dependence, and the input is not mutated."""
    d = decision(supporting_insights=[insight()])
    before = dict(d)
    assert explain(d) == explain(d)
    assert d == before


# ── 14. No temporal claim without a temporal diff ──────────────────────────


def test_no_explanation_claims_something_changed():
    """Nothing in this pipeline computes a diff between two snapshots for a
    decision, so no card may say "changed", "increased" or "since yesterday"."""
    for signal_type in _KINDS:
        got = explain(decision(signal_type=signal_type))
        text = " ".join(
            str(v) for v in (got["headline"], got["next_step"], got["kind_label"]) if v
        ).lower()
        for claim in ("changed", "increased", "decreased", "since yesterday", "trending"):
            assert claim not in text


# ── 15 + 16. Boundaries this task must not cross ───────────────────────────


def test_the_explanation_layer_imports_no_model_and_no_infrastructure():
    """It is a projection, not a producer. An LLM here would make generated
    prose indistinguishable from a stored organizational fact."""
    import ast
    import pathlib

    module = pathlib.Path(__file__).resolve().parents[2] / "navis" / "decisions" / "explanation.py"
    source = module.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(module))

    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert imported == {"__future__", "navis.decisions.base", "enum", "typing"}, imported

    for banned in ("sqlalchemy", "await ", "async def", "datetime.now(", "random"):
        assert banned not in source, f"the projection gained `{banned}`"


# ══════════════════════════════════════════════════════════════════════════════
# 7.5.7 — a connected situation outranks a bucket count
#
# Both families qualify as organizational context (both are folded from this
# decision's own signal and both describe this snapshot), so without a
# preference the winner was whichever the repository happened to return first —
# in production, always a count. "1 open commitment slipping risk" restates the
# card the founder is already reading; "Ajitesh has a cooling relationship and an
# unresolved commitment" is the fact they could not have derived from it.
# ══════════════════════════════════════════════════════════════════════════════

SITUATION = "situation_intelligence"


def situation_insight(**over: Any) -> dict[str, Any]:
    return insight(
        insight_id="5a1e0d3b-19cb-5f47-8a2e-6d4f0b7c9e21",
        insight_type=SITUATION,
        title="ajitesh@example.com has a cooling relationship and an unresolved commitment",
        sample_size=2,
        **over,
    )


def test_the_situation_type_literal_matches_intelligences_own_vocabulary():
    """`decisions` may not import `navis.intelligence` (AST-pinned), so the
    family is compared as a string. This test may import both, and is what stops
    the two vocabularies from drifting apart silently."""
    from navis.decisions.explanation import _SITUATION_INSIGHT_TYPE
    from navis.intelligence.models import InsightType

    assert _SITUATION_INSIGHT_TYPE == InsightType.SITUATION.value


def test_a_situation_wins_over_a_count_as_organizational_context():
    got = explain(decision(supporting_insights=[insight(), situation_insight()]))
    assert got["organizational_context"]["insight_type"] == SITUATION


def test_a_situation_wins_regardless_of_the_order_it_arrives_in():
    got = explain(decision(supporting_insights=[situation_insight(), insight()]))
    assert got["organizational_context"]["insight_type"] == SITUATION


def test_a_count_is_still_used_when_no_situation_qualifies():
    """The preference is a tiebreak, not a gate. An org whose pressures do not
    overlap produces no situation, and the count is still better than silence."""
    got = explain(decision(supporting_insights=[insight()]))
    assert got["organizational_context"]["insight_type"] == "obligation_intelligence"


def test_a_situation_that_describes_another_snapshot_never_wins():
    """The three existing conditions still apply first: a stale situation is not
    promoted over a current count."""
    got = explain(decision(supporting_insights=[
        insight(),
        situation_insight(describes_current_snapshot=False),
    ]))
    assert got["organizational_context"]["insight_type"] == "obligation_intelligence"


def test_a_situation_related_only_by_shared_entity_never_wins():
    got = explain(decision(supporting_insights=[
        insight(),
        situation_insight(relation=InsightRelation.SHARES_ENTITY.value),
    ]))
    assert got["organizational_context"]["insight_type"] == "obligation_intelligence"


def test_the_situation_statement_is_the_insights_title_verbatim():
    """One organizational fact, one voice. No layer rewords Intelligence."""
    ins = situation_insight()
    got = explain(decision(supporting_insights=[ins]))
    assert got["organizational_context"]["statement"] == ins["title"]
