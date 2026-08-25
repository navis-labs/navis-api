"""TODAY IS A PROJECTION, NOT AN ENGINE.

Today answers "what should I see right now?" by SHOWING what the layers below it
already concluded. It may not conclude anything itself, and — the failure this
suite exists to prevent — neither may the client. Every number, every claim and
every button on that surface has to resolve to a row somebody else wrote.

The audit that produced these tests found the breakage was rarely a wrong value.
It was a MISSING one: a fact the backend held and did not publish, so the client
filled the gap by inference. Four cases, all pinned below.

  * HOW MANY. A brief is ranked, deduped by entity and truncated to
    `brief_max_items`. Nothing said so, so the only count a surface could form
    was `len(decisions)` — the size of a display window presented as the number
    of people waiting for an answer. `obligations` publishes the real count from
    the reasoning pass the brief was built from, and `available` says when that
    count could not be read at all, because "0 waiting" and "we could not look"
    must never render the same.

  * WHICH CONCLUSION. `category` buckets six signal types into RISK, so a client
    could not tell a commitment slipping from a customer going quiet without
    parsing the engine's prose. `signal_type` is the engine's own answer.

  * WHY THIS. `supporting_insight_ids` flattened two structurally different
    relationships into one list, and the natural rendering of either is "this
    insight is why Navis raised this" — which neither one is.
    `supporting_insights` keeps the distinction.

  * WHAT CAN BE DONE. Whether a decision has an executable action was computed
    on the read path and thrown away, so an Execute affordance could only be
    offered on every card or none. `action` publishes it, derived by the SAME
    function the proposer uses so a card cannot offer what the write path would
    refuse.

Everything here is a pure-function test: no database, no network, no clock. The
enrichment's DB half is exercised against real Postgres in
`tests/integration/test_brief_execution_enrichment_postgres.py`.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

import pytest

from navis.api.brief_router import _serialize
from navis.decisions.base import (
    InsightRelation,
    IntelligenceFact,
    IntelligenceView,
    obligation_counts,
)
from navis.decisions.service import DecisionService
from navis.decisions.what_matters_today import WhatMattersTodayEngine
from navis.execution.models import ActionType
from navis.execution.proposer import decision_action
from navis.memory.envelope import EntityType
from navis.memory.ids import entity_id as entity_id_for
from navis.reasoning.v2.contract import ReasoningSignal, SignalType
from navis.signals import read_router as R

ORG = uuid.UUID("3c1d0f88-5a44-4e6b-9c02-7f1b8d3a5e90")
AS_OF = datetime(2026, 8, 20, 9, 0, 0, tzinfo=timezone.utc)
WATERMARK = "0198f2aa-4c31-7d2e-8b55-2a9c6e13f004"
REASONING_VERSION = 4

THREAD = "19fdb57e89581d43"
MESSAGE = "<CACoxYfgQQVZ1K=2OQOXvYjZ6-ms9Gp4Fxowp=s9N@mail.gmail.com>"


# ── Fixtures: real contract objects, built without infrastructure ───────────


def signal(
    signal_type: SignalType,
    *,
    tag: str,
    entity_type: str | None = None,
    entity_id: str | None = None,
    confidence: float = 0.9,
    severity: float = 0.6,
    evidence: dict[str, Any] | None = None,
) -> ReasoningSignal:
    return ReasoningSignal(
        signal_id=uuid.uuid5(uuid.NAMESPACE_URL, f"today:{tag}"),
        org_id=ORG,
        signal_type=signal_type,
        entity_type=entity_type,
        entity_id=entity_id,
        title=f"{tag} title",
        why=f"{tag} why",
        evidence=evidence if evidence is not None else {},
        confidence=confidence,
        severity=severity,
        engine_name="test_engine",
        engine_version="1.0.0",
        state_as_of=AS_OF,
        state_watermark=WATERMARK,
        reasoning_version=REASONING_VERSION,
        created_at=AS_OF,
    )


def waiting_signal(tag: str = "wait", *, days: float = 4.0) -> ReasoningSignal:
    """The one signal type that can currently become an executable action."""
    return signal(
        SignalType.WAITING_ON_USER_REPLY,
        tag=tag,
        entity_type="thread",
        entity_id=str(entity_id_for(ORG, EntityType.THREAD.value, THREAD)),
        evidence={
            "person_email": "nick@acme.com",
            "person_name": "Nick",
            "thread_id": THREAD,
            "rfc_message_id": MESSAGE,
            "subject": "Re: pricing",
            "waiting_days": days,
            "last_inbound_at": "2026-08-16T09:00:00+00:00",
        },
    )


def commitment_signal(tag: str = "commit", *, age_days: float = 11.85) -> ReasoningSignal:
    """An undated promise going stale — the golden commitment's shape.

    Distinct entity per tag, so the ranker's per-entity dedupe keeps each as its
    own card. Confidence/severity use `CommitmentReasoningEngine`'s own
    transforms so the values are the engine's, not invented.
    """
    return signal(
        SignalType.COMMITMENT_SLIPPING,
        tag=tag,
        entity_type="commitment",
        entity_id=str(entity_id_for(ORG, EntityType.COMMITMENT.value, tag)),
        confidence=min(0.9, 0.5 + age_days / 30),
        severity=min(0.9, 0.4 + age_days / 30),
        evidence={"age_days": round(age_days, 2), "speaker_email": "nick@acme.com"},
    )


def decide(signals: list[ReasoningSignal], intelligence: IntelligenceView | None = None):
    return WhatMattersTodayEngine().decide(
        signals,
        org_id=ORG,
        reasoning_version=REASONING_VERSION,
        state_watermark=WATERMARK,
        as_of=AS_OF,
        intelligence=intelligence,
    )


class _Brief:
    """A DailyBrief row, as `_serialize` and `obligations` read one."""

    def __init__(self, decisions: list[dict[str, Any]] | None = None):
        self.id = uuid.uuid5(uuid.NAMESPACE_URL, "today:brief")
        self.organization_id = ORG
        self.reasoning_version = REASONING_VERSION
        self.state_watermark = WATERMARK
        self.as_of = AS_OF
        self.decisions = decisions or []


class _SignalTypeSession:
    """The one query `DecisionService.obligations` issues, and nothing else.

    Deliberately minimal: it returns the `(signal_type, count)` rows a real
    grouped read over `reasoning_signals` would return for this brief's
    snapshot. Anything richer would be testing the fake. The statement itself
    (org + version + watermark filtering) runs against real Postgres in the
    integration suite.
    """

    def __init__(self, emitted: dict[str, int]):
        self._emitted = emitted

    async def execute(self, _stmt):
        rows = list(self._emitted.items())

        class _Result:
            def all(self):
                return rows

        return _Result()


# ── 1. Today only exposes real decisions, and says how many there are ───────


async def test_the_brief_publishes_the_true_obligation_count_not_the_page_size():
    """"6 people are waiting on you" must come from six stored signals.

    THE DEFECT. A brief shows at most `brief_max_items` decisions across every
    category. A surface counting its `waiting_reply` cards therefore reports how
    many fitted on the page, which is a different number from how many people
    are actually waiting — and the gap grows exactly when the founder is
    busiest.
    """
    shown = decide([waiting_signal(f"w{i}") for i in range(2)])
    brief = _Brief([d.to_dict() for d in shown])
    stored = {
        SignalType.WAITING_ON_USER_REPLY.value: 6,
        SignalType.COMMITMENT_SLIPPING.value: 1,
        SignalType.CONTEXT_SUMMARY.value: 1,  # interpretation — not an obligation
    }

    got = await DecisionService(_SignalTypeSession(stored)).obligations(brief)

    assert got["available"] is True
    assert got["by_signal_type"] == {
        SignalType.WAITING_ON_USER_REPLY.value: 6,
        SignalType.COMMITMENT_SLIPPING.value: 1,
    }
    assert got["total"] == 7
    # ... and the brief is honest that it is showing a window onto them.
    assert got["shown"] == 1  # both waiting signals share a thread → one card
    assert got["shown"] < got["total"]


async def test_every_shown_obligation_is_counted_so_shown_never_exceeds_total():
    """THE `shown = 4, total = 2` DEFECT, closed at the seam that caused it.

    `shown` is the ranked decision list; `total` counts stored obligations at the
    brief's snapshot. They came from populations the write path did not keep in
    agreement: `persist_signals` used ON CONFLICT DO NOTHING on the identity row,
    so a conclusion first detected at an EARLIER fold was ranked into today's
    brief while its stored row still carried the older watermark — and the count,
    filtering on that watermark, could not see it.

    Now the observation is written per (conclusion, snapshot) BEFORE Decision
    ranks, so the population `total` reads over is a SUPERSET of the one `shown`
    is drawn from. This asserts that containment directly: every signal type the
    brief displays is present in the counts, and shown <= total.

    `orchestration.brief` guarantees the ordering — `persist_signals` runs before
    `decide`, pinned by `tests/decisions/test_decision_semantics.py`.
    """
    # Two carried-over commitments (first detected at older folds) plus a fresh
    # waiting-reply — exactly the shape of the production brief that broke.
    signals = [
        commitment_signal("c-old-1"),
        commitment_signal("c-old-2"),
        waiting_signal("w-new"),
    ]
    shown = decide(signals)
    brief = _Brief([d.to_dict() for d in shown])

    # What `persist_signals` recorded for THIS snapshot: one observation per
    # conclusion of the pass, regardless of when each was first detected.
    stored: dict[str, int] = {}
    for s in signals:
        stored[s.signal_type.value] = stored.get(s.signal_type.value, 0) + 1

    got = await DecisionService(_SignalTypeSession(stored)).obligations(brief)

    assert got["available"] is True
    assert got["shown"] <= got["total"], (
        f"shown={got['shown']} exceeded total={got['total']} — the brief is "
        "displaying obligations its own count cannot see"
    )
    # Containment, not just arithmetic: no displayed obligation may be missing
    # from the counts. This is what actually failed live — commitment_slipping
    # was the top-ranked decision and absent from `by_signal_type` entirely.
    displayed = {d["signal_type"] for d in brief.decisions if d.get("signal_type")}
    assert displayed <= set(got["by_signal_type"]), (
        f"displayed {displayed - set(got['by_signal_type'])} with no count behind it"
    )
    assert SignalType.COMMITMENT_SLIPPING.value in got["by_signal_type"]


async def test_an_unreadable_snapshot_reports_absence_not_zero():
    """No stored reasoning pass must never render as "nothing needs you".

    Briefs written before `reasoning_signals` existed resolve to no rows. The
    honest report is "we cannot see this", which is the same distinction the
    frontend already draws for thread activity — an unavailable source excluded
    from a count, never folded in as zero.
    """
    got = await DecisionService(_SignalTypeSession({})).obligations(_Brief())

    assert got["available"] is False
    assert got["by_signal_type"] == {}
    assert got["total"] == 0


def test_obligation_counts_exclude_interpretation():
    """A state recitation and an inbound-activity observation ask nothing of
    anyone, so counting them as obligations would inflate the number with things
    the founder cannot act on — the same overclaim `_INTERPRETATION_ONLY` keeps
    out of the brief itself."""
    counts = obligation_counts(
        {
            SignalType.CONTEXT_SUMMARY.value: 1,
            SignalType.PRIORITY_LOAD.value: 3,
            SignalType.ORG_DRIFT.value: 2,
            SignalType.OPPORTUNITY.value: 9,
            SignalType.LEVERAGE_POINT.value: 4,
            SignalType.DECISION_OWED.value: 1,
        }
    )
    assert counts == {SignalType.DECISION_OWED.value: 1}


def test_obligation_counts_skip_an_unknown_vocabulary_rather_than_guessing():
    """A signal stored under a retired type name cannot be classified, and
    guessing is the fallback-to-PRIORITY defect the category map was rewritten
    to remove. It contributes nothing."""
    assert obligation_counts({"some_retired_type": 5, "": 1}) == {}


# ── 2. Which conclusion this is ─────────────────────────────────────────────


@pytest.mark.parametrize(
    "signal_type",
    [
        SignalType.COMMITMENT_SLIPPING,
        SignalType.RELATIONSHIP_COOLING,
        SignalType.PROJECT_STALLED,
        SignalType.CUSTOMER_ENGAGEMENT_DROP,
        SignalType.DECISION_UNRESOLVED,
        SignalType.MEETING_OVERLOAD,
    ],
)
def test_signal_type_survives_into_the_decision(signal_type):
    """Six different findings all land in category RISK. Without `signal_type` a
    client cannot tell "a promise is slipping" from "a customer went quiet"
    except by parsing the engine's prose, which is not a contract."""
    [d] = decide([signal(signal_type, tag=signal_type.value)])
    assert d.signal_type == signal_type.value
    assert d.to_dict()["signal_type"] == signal_type.value
    assert d.to_dict()["category"] == "risk"


def test_every_serialized_decision_carries_its_own_provenance():
    """Reasoning, state and entity lineage must survive serialization — this is
    what makes "Why this?" resolvable rather than a restatement of the card."""
    [d] = decide([waiting_signal()])
    payload = d.to_dict()

    assert payload["reasoning_signal_ids"] == [str(uuid.uuid5(uuid.NAMESPACE_URL, "today:wait"))]
    assert payload["reasoning_version"] == REASONING_VERSION
    assert payload["state_watermark"] == WATERMARK
    assert payload["created_at"] == AS_OF.isoformat()  # logical time, never now()
    assert payload["entity_type"] == "thread"
    assert payload["entity_id"] == str(entity_id_for(ORG, EntityType.THREAD.value, THREAD))
    assert payload["signal_type"] == SignalType.WAITING_ON_USER_REPLY.value
    # The engine's own measurements, not the ranker's score.
    assert payload["confidence"] == 0.9
    assert payload["severity"] == 0.6


def test_an_org_wide_decision_reports_no_entity_rather_than_a_placeholder():
    """`meta.py` and `risk.py` emit signals with `entity_id=None`. Null means
    "org-wide", which is a real answer and must stay distinguishable from a
    fabricated id."""
    [d] = decide([signal(SignalType.MEETING_OVERLOAD, tag="overload")])
    payload = d.to_dict()
    assert payload["entity_type"] is None and payload["entity_id"] is None


# ── 3. Intelligence is never presented as cause ─────────────────────────────


def _fact(
    insight_id: str,
    *,
    signal_ids=(),
    entities=(),
    confidence: float = 0.8,
    insight_type: str = "risk_intelligence",
    describes_state: bool = True,
    watermark: str | None = WATERMARK,
    title: str = "3 open relationship cooling risks",
    sample_size: int = 3,
):
    """One insight as the composition root flattens it for Decision.

    `describes_state` mirrors `intelligence.insights.REALITY_INSIGHT_TYPES`
    membership, which `orchestration.brief._intelligence_view` resolves — it is
    what separates a description of the current snapshot from a measurement over
    past outcomes.
    """
    evidence: dict[str, Any] = {}
    if entities:
        evidence["entities"] = [dict(e) for e in entities]
    return IntelligenceFact(
        insight_id=insight_id,
        insight_type=insight_type,
        key="k",
        summary="s",
        confidence=confidence,
        sample_size=sample_size,
        evidence=evidence,
        supporting_reasoning_signal_ids=tuple(signal_ids),
        state_watermark=watermark,
        learning_version=3,
        describes_state=describes_state,
        title=title,
    )


def test_a_reality_insight_folded_from_this_signal_is_shared_origin_not_cause():
    s = waiting_signal()
    view = IntelligenceView(
        org_id=ORG, as_of=AS_OF, facts=(_fact("i-origin", signal_ids=(str(s.signal_id),)),)
    )
    [d] = decide([s], view)

    assert d.to_dict()["supporting_insights"] == [
        {
            "insight_id": "i-origin",
            "relation": InsightRelation.DERIVED_FROM_SAME_SIGNAL.value,
            "insight_type": "risk_intelligence",
            "title": "3 open relationship cooling risks",
            "sample_size": 3,
            "state_watermark": WATERMARK,
            "describes_current_snapshot": True,
        }
    ]


def test_a_learning_insight_naming_this_signal_is_outcome_history_not_reality():
    """Both families match on a signal id and mean different things.

    A `signal_intelligence` row measures what happened when the organization
    ACTED on signals like this one — folded from `learning_events`, versioned by
    `learning_version`. Reporting that as "folded from this signal" would claim
    it describes the organization right now, which it does not.
    """
    s = waiting_signal()
    view = IntelligenceView(
        org_id=ORG,
        as_of=AS_OF,
        facts=(
            _fact(
                "i-learned",
                signal_ids=(str(s.signal_id),),
                insight_type="signal_intelligence",
                describes_state=False,
                watermark="wm-from-some-execution",
            ),
        ),
    )
    [d] = decide([s], view)

    [got] = d.to_dict()["supporting_insights"]
    assert got["relation"] == InsightRelation.LEARNED_FROM_OUTCOMES.value
    assert got["insight_type"] == "signal_intelligence"
    # A Learning row's watermark is inherited from whichever execution recorded
    # one. It is NOT a claim about a snapshot, so no verdict is reported on it.
    assert got["describes_current_snapshot"] is None


def test_a_stale_reality_generation_cannot_pass_as_current_corroboration():
    """THE P0. `compute_signal_id` does not hash the state watermark, so the same
    cooling relationship carries the SAME signal id at every snapshot — and
    `_project_intelligence` is deliberately non-fatal, so a brief at W2 can be
    ranked against the newest STORED generation, still at W1. The insight then
    matched this pass's signals and was attached as corroboration while its
    content described a different organization. Both watermarks were in hand."""
    s = waiting_signal()
    view = IntelligenceView(
        org_id=ORG,
        as_of=AS_OF,
        facts=(
            _fact("i-stale", signal_ids=(str(s.signal_id),), watermark="an-older-watermark"),
        ),
    )
    [d] = decide([s], view)

    [got] = d.to_dict()["supporting_insights"]
    assert got["relation"] == InsightRelation.DERIVED_FROM_SAME_SIGNAL.value
    assert got["describes_current_snapshot"] is False
    assert got["state_watermark"] == "an-older-watermark"
    # Still present, still listed: the evidence is real and is not discarded —
    # it simply can no longer claim to be about this snapshot.
    assert d.to_dict()["supporting_insight_ids"] == ["i-stale"]


def test_the_relation_list_and_the_id_list_stay_positionally_identical():
    """A client zipping the two must never mismatch an id to a relation."""
    s = waiting_signal()
    view = IntelligenceView(
        org_id=ORG,
        as_of=AS_OF,
        facts=(
            _fact("i-zeta", signal_ids=(str(s.signal_id),)),
            _fact("i-alpha", entities=({"entity_type": s.entity_type, "entity_id": s.entity_id},)),
        ),
    )
    payload = decide([s], view)[0].to_dict()

    assert payload["supporting_insight_ids"] == ["i-alpha", "i-zeta"]
    assert [i["insight_id"] for i in payload["supporting_insights"]] == ["i-alpha", "i-zeta"]
    assert [i["relation"] for i in payload["supporting_insights"]] == [
        InsightRelation.SHARES_ENTITY.value,
        InsightRelation.DERIVED_FROM_SAME_SIGNAL.value,
    ]


def test_only_relationships_the_architecture_stores_are_expressible():
    """No `predicted_by_ml`, no `contradicts`, no `supported_by_state`.

    Nothing in this system computes contradiction between insights; no model
    contributes to Intelligence at all; and a state link would be a
    re-derivation of `entity_id`, which every decision already carries. A
    vocabulary member with no producer is fabricated provenance waiting to be
    rendered.
    """
    assert {r.value for r in InsightRelation} == {
        "derived_from_same_signal",
        "shares_entity",
        "learned_from_outcomes",
    }


def test_no_relevant_insight_produces_an_explicitly_empty_list():
    """Absence is a value. A client must be able to tell "no insight is about
    this" from "this build predates the field"."""
    payload = decide([waiting_signal()], IntelligenceView(org_id=ORG, as_of=AS_OF))[0].to_dict()
    assert payload["supporting_insights"] == []
    assert payload["supporting_insight_ids"] == []


# ── 4. What can actually be done ────────────────────────────────────────────


def test_an_executable_decision_names_the_existing_action_type():
    """Today does not get its own execution path. The action a card offers is
    the one `ExecutionService` would create, named by `ActionType`."""
    [d] = decide([waiting_signal()])
    action = decision_action(d.to_dict())

    assert action is not None
    assert action.action_type == ActionType.REPLY_TO_EMAIL.value
    assert action.requires_draft is True  # the proposer never writes a body
    assert action.obligation_key and THREAD in action.obligation_key


def test_a_finding_has_no_action_so_no_execute_button_can_be_offered():
    """`risk` decisions are findings to read. Proposing one would mean a blank
    email or a fake task — both of which shipped once and are why `_map`
    returns nothing for them."""
    [d] = decide([signal(SignalType.RELATIONSHIP_COOLING, tag="cool")])
    assert d.to_dict()["category"] == "risk"
    assert decision_action(d.to_dict()) is None


def test_a_draftable_decision_that_cannot_name_its_message_has_no_action():
    """No `rfc_message_id` means the reply obligation has no stable identity, so
    a rebuilt brief would mint a second "new action" for the same email. Not
    proposing it is the honest outcome; offering the button would not be."""
    s = signal(
        SignalType.WAITING_ON_USER_REPLY,
        tag="nomsg",
        entity_type="thread",
        entity_id="t-nomsg",
        evidence={"person_email": "nick@acme.com", "thread_id": THREAD, "subject": "x"},
    )
    [d] = decide([s])
    assert decision_action(d.to_dict()) is None


def test_the_read_path_and_the_write_path_agree_on_the_obligation():
    """One derivation, so a card can never offer an action the proposer refuses
    to create — or report "not sent yet" for work that was sent."""
    from navis.execution.proposer import decision_obligation_key, propose_from_decisions

    payload = decide([waiting_signal()])[0].to_dict()
    [proposed] = propose_from_decisions([payload], ORG)
    action = decision_action(payload)

    assert action is not None
    assert action.action_type == proposed.action_type
    assert action.obligation_key == decision_obligation_key(payload)


# ── 5. The brief envelope ───────────────────────────────────────────────────


def test_the_envelope_always_carries_the_snapshot_it_describes():
    """Freshness has to be checkable. `as_of`, `state_watermark` and
    `reasoning_version` are the tuple that says which organization, at which
    moment, under which reasoning this brief describes."""
    body = _serialize(_Brief([{"decision_id": "d"}]), obligations={"available": False})

    assert body["as_of"] == AS_OF.isoformat()
    assert body["state_watermark"] == WATERMARK
    assert body["reasoning_version"] == REASONING_VERSION
    assert body["obligations"] == {"available": False}


def test_every_decision_agrees_with_the_envelope_about_the_snapshot():
    """A decision claiming a different watermark than the brief that carries it
    is a lineage defect: the two would be describing different organizations."""
    brief = _Brief([d.to_dict() for d in decide([waiting_signal(), signal(SignalType.PROJECT_STALLED, tag="proj")])])
    body = _serialize(brief)

    assert body["decisions"]
    for d in body["decisions"]:
        assert d["state_watermark"] == body["state_watermark"]
        assert d["reasoning_version"] == body["reasoning_version"]
        assert d["created_at"] == body["as_of"]


def test_the_decision_count_is_the_decision_list():
    """"4 decisions" must be four returned records — nothing derived, nothing
    filtered client-side."""
    decisions = [d.to_dict() for d in decide([signal(SignalType.PROJECT_STALLED, tag=f"p{i}",
                                                     entity_type="project", entity_id=f"p-{i}")
                                              for i in range(4)])]
    body = _serialize(_Brief(decisions))
    assert len(body["decisions"]) == 4


# ── 6. The meeting contract ─────────────────────────────────────────────────


def _cal_row(**payload_overrides: Any):
    from dataclasses import dataclass

    @dataclass
    class _Row:
        id: uuid.UUID
        external_id: str
        occurred_at: datetime | None
        created_at: datetime | None
        payload: dict[str, Any]
        event_type: str

    payload: dict[str, Any] = {
        "summary": "Pricing review",
        "status": "confirmed",
        "start": {"dateTime": "2026-08-24T10:30:00+05:30", "timeZone": "Asia/Kolkata"},
        "end": {"dateTime": "2026-08-24T11:15:00+05:30", "timeZone": "Asia/Kolkata"},
        "attendees": [{"email": "nick@acme.com"}, {"email": "founder@navis.io"}],
        "attendee_count": 2,
        "organizer": {"email": "founder@navis.io"},
        "hangout_link": "https://meet.google.com/tap-kbov-nfz",
        "conference_data": {"conference_id": "tap-kbov-nfz", "has_video": True},
    }
    payload.update(payload_overrides)
    return _Row(
        id=uuid.uuid4(),
        external_id="oji40sl3uf61senmcc1a99k17k",
        occurred_at=datetime(2026, 8, 24, 5, 0, tzinfo=timezone.utc),
        created_at=datetime(2026, 8, 22, 6, 30, tzinfo=timezone.utc),
        payload=payload,
        event_type="calendar_event_updated",
    )


def test_a_meeting_publishes_the_same_field_names_the_rest_of_the_pipeline_uses():
    """One meeting, one set of names. Memory's `MeetingExtractor` writes
    `title`/`starts_at`/`ends_at`; this feed used to publish Google's
    `summary`/`start`/`end` and leave the reconciliation to the client."""
    m = R._shape_calendar(_cal_row(), org_id=ORG)

    assert m["title"] == "Pricing review"
    assert m["starts_at"] == "2026-08-24T10:30:00+05:30"
    assert m["ends_at"] == "2026-08-24T11:15:00+05:30"
    assert m["all_day"] is False
    assert m["meeting_url"] == "https://meet.google.com/tap-kbov-nfz"
    assert m["organizer_email"] == "founder@navis.io"
    assert m["attendees"] == ["nick@acme.com", "founder@navis.io"]


def test_the_meeting_entity_id_is_the_one_memory_and_context_use():
    """THE JOIN. `/context` publishes MEETING entities under this exact id, so
    a client can ask whether the organization knows anything about a meeting —
    and honestly say "no context available yet" when it does not."""
    m = R._shape_calendar(_cal_row(), org_id=ORG)

    assert m["entity_type"] == EntityType.MEETING.value
    assert m["entity_id"] == str(
        entity_id_for(ORG, EntityType.MEETING.value, "oji40sl3uf61senmcc1a99k17k")
    )


def test_the_same_meeting_in_two_organizations_never_shares_an_entity_id():
    """A calendar event id is a Google identifier, not a Navis one — two tenants
    can hold the same string. Identity is scoped by construction."""
    other = uuid.UUID("9b7e2a10-4c33-4f5d-8e21-0a6c9d4b7e11")
    assert (
        R._shape_calendar(_cal_row(), org_id=ORG)["entity_id"]
        != R._shape_calendar(_cal_row(), org_id=other)["entity_id"]
    )


def test_a_meeting_says_what_kind_of_thing_it_is_and_which_system_saw_it():
    """Today branches on the CONCEPT, never the provider. A future Outlook or
    Teams meeting arrives as `source_type: calendar` with a different
    `provider`, and needs no Today-specific renderer."""
    m = R._shape_calendar(_cal_row(), org_id=ORG)

    assert m["source_type"] == "calendar"
    assert m["provider"] == "google_calendar"
    assert m["external_id"] == "oji40sl3uf61senmcc1a99k17k"


def test_an_all_day_meeting_reports_all_day_rather_than_a_fabricated_time():
    """Google gives `{"date": ...}` with no instant. Inventing midnight in some
    timezone would place the meeting on the founder's timeline at a moment
    nobody scheduled."""
    m = R._shape_calendar(
        _cal_row(start={"date": "2026-08-24"}, end={"date": "2026-08-25"}), org_id=ORG
    )

    assert m["all_day"] is True
    assert m["starts_at"] is None and m["ends_at"] is None
    # The provider's own shape is still published, so the date is not lost.
    assert m["start"] == {"date": "2026-08-24"}


def test_when_it_happens_and_when_we_saw_it_are_separate_fields():
    """`occurred_at` for a calendar signal is the meeting's start; `observed_at`
    is ingestion. A surface that conflates them reorders the day."""
    m = R._shape_calendar(_cal_row(), org_id=ORG)

    assert m["occurred_at"] == "2026-08-24T05:00:00+00:00"
    assert m["observed_at"] == "2026-08-22T06:30:00+00:00"


def test_a_meeting_without_a_conference_link_reports_null_not_a_guess():
    """No `meeting_url` means there is no way to join, which a Join button must
    be able to see. Nothing here synthesises one from the conference id."""
    m = R._shape_calendar(_cal_row(hangout_link=None, conference_data=None), org_id=ORG)

    assert m["meeting_url"] is None
    assert m["has_conference"] is False


def test_the_meeting_feed_carries_no_organizational_interpretation():
    """A meeting reaching Today through Calendar → Signal → Today has no
    Memory, State, Intelligence or Decision behind it yet. This feed must
    therefore publish no importance, no purpose, no risk and no priority — the
    absence is what lets a client say "no meeting context available yet"
    instead of rendering an invented one."""
    m = R._shape_calendar(_cal_row(), org_id=ORG)

    invented = {
        "importance", "importance_score", "priority", "urgency", "risk",
        "purpose", "agenda", "why", "why_it_matters", "summary_text",
        "confidence", "decision", "decision_id", "insight", "recommendation",
    }
    assert not (invented & set(m)), f"the calendar feed is interpreting: {invented & set(m)}"


# ── 7. The execution boundary: who would do it, and may they ───────────────
#
# Phase 8's questions were all answerable from `execution.actions` and none of
# them were reachable from the decision that proposes the action. A surface could
# name an action and still not say which system would perform it, whether a human
# must approve it, whether it can be unsent, or whether performing it refreshes
# what Navis knows.


def test_the_action_is_described_by_the_registered_spec_not_by_the_decision():
    """Every field is the registry's, verified against the registry itself. A
    second opinion about what an action costs is how a card ends up offering
    something the worker refuses."""
    from navis.execution.actions import ApprovalPolicy, require_spec

    action = decision_action(decide([waiting_signal()])[0].to_dict())
    assert action is not None
    spec = require_spec(action.action_type)

    assert action.provider == spec.provider
    assert action.capability == spec.capability.value
    assert action.risk == spec.risk.value
    assert action.requires_approval is (spec.approval is ApprovalPolicy.REQUIRED)
    assert action.updates_read_model is (spec.reconciler is not None)


def test_a_reply_is_declared_irreversible_and_approval_gated():
    """The one thing a person needs before pressing approve. A reply is a message
    that cannot be unsent, and no proposal may execute without a human."""
    from navis.execution.models import ActionType

    action = decision_action(decide([waiting_signal()])[0].to_dict())
    assert action is not None
    assert action.action_type == ActionType.REPLY_TO_EMAIL.value
    assert action.provider == "gmail"
    assert action.risk == "irreversible_communication"
    assert action.requires_approval is True


def test_every_registered_action_that_touches_the_world_requires_approval():
    """Execution stays non-autonomous by declaration as well as by the status
    machine. Nothing Today can offer is exempt."""
    from navis.execution.actions import ApprovalPolicy, all_specs

    assert all_specs()
    for action_type, spec in all_specs().items():
        assert spec.approval is ApprovalPolicy.REQUIRED, (
            f"{action_type} would execute without a human"
        )


def test_the_read_model_edge_is_declared_per_action_not_assumed():
    """"When does Today stop showing this?" has two answers, and which one
    applies is a property of the action.

    A calendar verb reconciles its own result straight back into `signal_events`
    (`actions._calendar_reconcile`), so State reflects it on the next fold. A
    sent reply does not: it re-enters through the Gmail connector's next sync,
    which is what eventually flips `ThreadState.direction` to outbound and stops
    WAITING_ON_USER_REPLY firing. Both are real; conflating them would promise
    an immediacy the mail path does not have.
    """
    from navis.execution.actions import all_specs

    specs = all_specs()
    assert specs["reply_to_email"].reconciler is None
    for calendar_action in ("schedule_meeting", "reschedule_meeting", "cancel_meeting"):
        assert specs[calendar_action].reconciler is not None


# ── 8. Provider neutrality all the way down ────────────────────────────────


def test_no_layer_between_reasoning_and_today_names_a_provider():
    """A Gmail thread and an Outlook thread must reach the same reasoning.

    The pipeline domains reason over canonical entities (`thread`, `commitment`,
    `relationship`) and canonical signal types. Provider names belong at the two
    ends — the connector that captured the event, and the ActionSpec that would
    perform the action — and nowhere in between.
    """
    import ast
    import pathlib

    providers = ("gmail", "google_calendar", "outlook", "slack", "jira", "notion",
                 "hubspot", "salesforce", "linear")
    root = pathlib.Path(__file__).resolve().parents[2] / "navis"
    violations = []
    for domain in ("state", "reasoning/v2", "intelligence", "decisions"):
        for py in (root / domain).rglob("*.py"):
            tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
            for node in ast.walk(tree):
                # STRING LITERALS only — a comment naming Gmail as an example is
                # documentation, a literal is a branch waiting to happen.
                if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                    continue
                if node.value.lower() in providers:
                    violations.append(f"{domain}/{py.name}:{node.lineno} {node.value!r}")
    assert not violations, (
        "a provider name leaked into the reasoning pipeline:\n" + "\n".join(violations)
    )


def test_decision_entities_are_canonical_types_not_provider_objects():
    """`thread`, `commitment`, `relationship` — never `gmail_message`."""
    from navis.memory.envelope import EntityType

    canonical = {e.value for e in EntityType}
    for d in decide([waiting_signal(), signal(SignalType.PROJECT_STALLED, tag="p",
                                              entity_type="project", entity_id="p-1")]):
        assert d.entity_type in canonical


# ══════════════════════════════════════════════════════════════════════════════
# 7.5.7 — Today cannot silently serve a stale organization
#
# `latest_brief` returns the newest stored brief and `/brief/today` built one
# only when NONE existed, so once a brief existed the endpoint served it
# forever. Live, Today presented a picture ranked at watermark f8675a7f
# (2026-08-22) while State had advanced to f0102043, with 55 memory records
# folded in between and the lead commitment aged from 11.85 days to 13.89.
# ══════════════════════════════════════════════════════════════════════════════

NEWER_WATERMARK = "0198f3bb-5d42-7e3f-9c66-3b0d7f24a115"


def _freshness_of(brief_watermark, state_watermark, *, rebuilt=False):
    from navis.orchestration.brief import _freshness

    brief = _Brief()
    brief.state_watermark = brief_watermark
    return _freshness(brief, state_watermark, rebuilt=rebuilt)


def test_a_brief_at_states_own_watermark_reports_current():
    assert _freshness_of(WATERMARK, WATERMARK)["current"] is True


def test_a_brief_behind_state_reports_stale_rather_than_current():
    assert _freshness_of(WATERMARK, NEWER_WATERMARK)["current"] is False


def test_an_unbuildable_comparison_reports_unknown_and_never_current():
    """TRI-STATE. An org with no snapshot at the current projection version
    cannot be compared, and `None` must stay distinguishable from `True` — a
    client rendering "unknown" as "up to date" makes a claim this block
    deliberately refuses to make."""
    got = _freshness_of(WATERMARK, None)
    assert got["current"] is None
    assert got["current"] is not True


def test_freshness_names_both_watermarks_so_the_gap_is_auditable():
    got = _freshness_of(WATERMARK, NEWER_WATERMARK)
    assert got["brief_watermark"] == WATERMARK
    assert got["state_watermark"] == NEWER_WATERMARK


def test_freshness_says_whether_this_request_did_the_work():
    """Operationally the difference between "the scheduler is keeping up" and
    "reads are carrying it"."""
    assert _freshness_of(WATERMARK, WATERMARK, rebuilt=True)["rebuilt"] is True
    assert _freshness_of(WATERMARK, WATERMARK)["rebuilt"] is False


def test_the_envelope_carries_freshness_without_disturbing_any_existing_field():
    """BACKWARD COMPATIBILITY. Every field a client reads today must still be
    there, in place, with the same value — `freshness` is strictly additive."""
    brief = _Brief([{"decision_id": "d"}])
    before = _serialize(brief, obligations={"available": False})
    after = _serialize(
        brief,
        obligations={"available": False},
        freshness=_freshness_of(WATERMARK, WATERMARK),
    )

    assert set(after) - set(before) == set()
    for key in ("id", "org_id", "as_of", "reasoning_version",
                "state_watermark", "decisions", "obligations"):
        assert after[key] == before[key]
    assert after["freshness"]["current"] is True


def test_a_route_that_makes_no_comparison_reports_null_not_a_guess():
    """`/brief/generate` has just built the brief from the state it folded, so it
    states nothing rather than asserting freshness it did not check. That stays
    distinguishable from `current: null`, which means the check WAS attempted."""
    assert _serialize(_Brief())["freshness"] is None
