"""Locks the Decisions page read contracts: /learning/summary, /outcomes,
/brief/decisions/history, and the DECISION_DISMISSED -> Outcome -> Learning loop.

WHAT THESE EXIST TO PREVENT. The Decisions page previously read
`/decision-memory` — a Memory-layer feed of decision SENTENCES extracted from
email — and rendered its own zeros when that returned nothing: `Total decisions:
0`, `Average impact: +0%`, `Positive rate: 0%`. Every one of those numbers was
either about the wrong concept or invented. Meanwhile `outcomes` and
`learning_events` held real rows that no HTTP surface could reach.

So the properties under test are mostly about HONESTY:

  * a rate over zero decisive samples is null, never 0.0;
  * a stage that cannot happen says so, and is never a count;
  * every join runs on canonical lineage ids, never on title or subject text;
  * one reasoning conclusion is ONE history row however many watermarks it has
    spanned, and two conclusions that happen to share a title stay two.
"""
from __future__ import annotations

import inspect
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from navis.api import brief_router as BR
from navis.learning import router as LR
from navis.learning.metrics import AggregateFact, LearningSnapshot, SignalPattern
from navis.observation import router as OR
from navis.observation.models import OutcomeType
from navis.orchestration import decision_history as DH
from navis.organization.contract import Capability

ORG = uuid.UUID("11111111-1111-1111-1111-111111111111")
OTHER_ORG = uuid.UUID("22222222-2222-2222-2222-222222222222")
USER = uuid.UUID("33333333-3333-3333-3333-333333333333")

T0 = datetime(2026, 8, 24, 18, 46, tzinfo=timezone.utc)

# The two live commitment decisions from the audit: byte-identical titles, two
# different people, two different commitment entities, two different signals.
SIGNAL_AJITESH = "d9170d07-3936-5a2e-abae-b0282d91fb18"
SIGNAL_ASTRA = "0cadfe99-b9bb-56a7-b407-6223957a5b37"
ENTITY_AJITESH = "f33778d8-a54e-50bb-87b9-2d74d83584e2"
ENTITY_ASTRA = "02dbc756-1f71-5c82-afa8-44f08b59487c"
SAME_TITLE = "Commitment slipping: Thanks for following up on this — I'll come back to you on it."


@dataclass
class _Principal:
    org_id: uuid.UUID = ORG


def _decision(
    *,
    decision_id: str,
    signal_id: str,
    entity_id: str | None,
    title: str = SAME_TITLE,
    signal_type: str = "commitment_slipping",
    category: str = "risk",
    watermark: str = "wm-3",
    speaker: str | None = None,
) -> dict[str, Any]:
    """A stored decision, shaped exactly as `daily_briefs.decisions` holds one."""
    return {
        "decision_id": decision_id,
        "org_id": str(ORG),
        "title": title,
        "summary": "Open for 14 days with no newer fact in Memory.",
        "why": "Open for 14 days with no newer fact in Memory.",
        "evidence": {"age_days": 14.27, "speaker_email": speaker},
        "priority_rank": 1,
        "category": category,
        "reasoning_signal_ids": [signal_id],
        "reasoning_version": 4,
        "state_watermark": watermark,
        "created_at": T0.isoformat(),
        "supporting_insight_ids": [],
        "supporting_insights": [],
        "signal_type": signal_type,
        "confidence": 0.9,
        "severity": 0.8756,
        "entity_type": "commitment" if entity_id else None,
        "entity_id": entity_id,
    }


def _brief(decisions: list[dict], *, as_of: datetime, watermark: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid.uuid4(),
        organization_id=ORG,
        reasoning_version=4,
        state_watermark=watermark,
        as_of=as_of,
        decisions=decisions,
    )


def _execution(
    *,
    decision_id: str,
    signals: list[str],
    status: str = "executed",
    approved: bool = True,
    created_at: datetime = T0,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid.uuid4(),
        organization_id=ORG,
        decision_id=uuid.UUID(decision_id),
        reasoning_signal_ids=signals,
        action_type="reply_to_email",
        status=status,
        payload={"body": "drafted"},
        provider_reference={"provider": "gmail", "external_id": "m-1"},
        attempts=1,
        last_error=None,
        approved_at=T0 if approved else None,
        created_at=created_at,
    )


def _outcome(
    *,
    decision_id: str | None,
    signals: list[str],
    outcome_type: str = "action_executed",
    occurred_at: datetime = T0,
) -> SimpleNamespace:
    return SimpleNamespace(
        outcome_id=uuid.uuid4(),
        organization_id=ORG,
        execution_id=uuid.uuid4(),
        decision_id=uuid.UUID(decision_id) if decision_id else None,
        reasoning_signal_ids=signals,
        reasoning_version=4,
        state_watermark="wm-3",
        provider_reference={},
        outcome_type=outcome_type,
        payload={},
        occurred_at=occurred_at,
        recorded_at=occurred_at,
        created_at=occurred_at,
    )


def _learning(
    *, decision_id: str | None, signals: list[str], result: str, surface: str = "judgement"
) -> SimpleNamespace:
    return SimpleNamespace(
        learning_event_id=str(uuid.uuid4()),
        outcome_id=str(uuid.uuid4()),
        execution_id=None,
        decision_id=decision_id,
        reasoning_signal_ids=tuple(signals),
        outcome_type="decision_dismissed",
        outcome_surface=surface,
        result=result,
        occurred_at=T0,
    )


@dataclass
class _Dismissal:
    decision_ref: str
    reason: str = "explicit"
    created_at: datetime = T0
    organization_id: uuid.UUID = ORG


class _FakeDecisions:
    def __init__(self, briefs: list, dismissals: list | None = None) -> None:
        self.briefs = briefs
        self._dismissals = dismissals or []
        self.scoped: list[uuid.UUID] = []

    async def recent_briefs(self, org_id, *, limit=200):
        self.scoped.append(org_id)
        return [b for b in self.briefs if b.organization_id == org_id][:limit]

    async def dismissals(self, org_id, *, limit=500):
        self.scoped.append(org_id)
        return [d for d in self._dismissals if d.organization_id == org_id][:limit]


class _FakeExecutions:
    def __init__(self, rows: list) -> None:
        self.rows = rows
        self.scoped: list[uuid.UUID] = []

    async def recent_for_org(self, org_id, *, limit=200):
        self.scoped.append(org_id)
        return [r for r in self.rows if r.organization_id == org_id][:limit]


class _FakeOutcomes:
    def __init__(self, rows: list) -> None:
        self.rows = rows
        self.scoped: list[uuid.UUID] = []

    async def recent(self, org_id, *, limit=200):
        self.scoped.append(org_id)
        return [r for r in self.rows if r.organization_id == org_id][:limit]


def _install_history(
    monkeypatch,
    *,
    briefs: list | None = None,
    executions: list | None = None,
    outcomes: list | None = None,
    learning: list | None = None,
    dismissals: list | None = None,
):
    """Route the read model's four boundaries at in-memory fakes."""
    decisions = _FakeDecisions(briefs or [], dismissals or [])
    execs = _FakeExecutions(executions or [])
    outs = _FakeOutcomes(outcomes or [])
    seen_learning: list[uuid.UUID] = []

    async def _learning_events_for(db, org_id, *, limit=200):
        seen_learning.append(org_id)
        return tuple(learning or ())

    monkeypatch.setattr(DH, "DecisionService", lambda db: decisions)
    monkeypatch.setattr(DH, "ExecutionRepository", lambda db: execs)
    monkeypatch.setattr(DH, "OutcomeRepository", lambda db: outs)
    monkeypatch.setattr(DH, "learning_events_for", _learning_events_for)
    return SimpleNamespace(
        decisions=decisions, executions=execs, outcomes=outs, learning=seen_learning
    )


def _by_key(result: dict) -> dict[str, dict]:
    return {e["episode_key"]: e for e in result["episodes"]}


# ────────────────────────── /learning/summary ────────────────────────────────


def _snapshot(**kw) -> LearningSnapshot:
    return LearningSnapshot(
        org_id=kw.pop("org_id", ORG),
        learning_version=kw.pop("learning_version", 0),
        as_of=kw.pop("as_of", None),
        state_watermark=kw.pop("state_watermark", None),
        reasoning_version=kw.pop("reasoning_version", None),
        aggregates=kw.pop("aggregates", ()),
        patterns=kw.pop("patterns", ()),
        lineage=kw.pop("lineage", {}),
        signal_lineage=kw.pop("signal_lineage", {}),
    )


async def test_learning_summary_publishes_the_real_snapshot(monkeypatch):
    """Every number is a stored integer or Learning's own derivation of one."""
    snapshot = _snapshot(
        learning_version=4,
        as_of=T0,
        state_watermark="wm-3",
        reasoning_version=4,
        aggregates=(
            AggregateFact("outcome", "reply_received", 3, 0, 0, 3),
            AggregateFact("outcome", "decision_dismissed", 0, 1, 0, 1),
            AggregateFact("signal", SIGNAL_AJITESH, 1, 1, 0, 2),
        ),
        patterns=(SignalPattern(SIGNAL_AJITESH, "reliable", 0.75, 8),),
    )
    monkeypatch.setattr(LR, "learning_snapshot", lambda db, org: _snapshot_of(snapshot, org))

    body = await LR.learning_summary(p=_Principal(), db=None)  # type: ignore[arg-type]

    assert body["learning_version"] == 4
    assert body["as_of"] == T0.isoformat()
    assert body["reasoning_version"] == 4
    # Totals are summed over the OUTCOME dimension only — 3 successes + 1
    # failure. Summing SIGNAL rows too would double-count to 4/2.
    assert body["totals"]["successes"] == 3
    assert body["totals"]["failures"] == 1
    assert body["totals"]["decisive"] == 4
    assert body["totals"]["success_rate"] == 0.75
    # Learning's own delivery/judgement/world grouping, not a second copy here.
    assert set(body["by_surface"]) == {"world", "judgement"}
    assert body["by_signal"][0]["key"] == SIGNAL_AJITESH
    assert body["patterns"][0]["kind"] == "reliable"


def _snapshot_of(snapshot: LearningSnapshot, org_id):
    """Async shim that also proves the handler passed the principal's org."""
    async def _run():
        assert org_id == snapshot.org_id
        return snapshot
    return _run()


async def test_learning_summary_reports_null_not_zero_when_nothing_is_measured(monkeypatch):
    """THE headline defect: `Average impact +0%` / `Positive rate 0%` on an org
    that had never had a decisive outcome. Zero is a measurement and this is not
    one."""
    empty = _snapshot()
    monkeypatch.setattr(LR, "learning_snapshot", lambda db, org: _snapshot_of(empty, org))

    body = await LR.learning_summary(p=_Principal(), db=None)  # type: ignore[arg-type]

    assert body["totals"]["learning_events"] == 0
    assert body["totals"]["decisive"] == 0
    assert body["totals"]["success_rate"] is None  # NOT 0.0
    # A surface with no outcome is ABSENT, not a key holding a fabricated zero.
    assert body["by_surface"] == {}
    assert body["by_outcome_type"] == []
    assert body["patterns"] == []


async def test_learning_summary_rate_is_null_while_only_neutrals_exist(monkeypatch):
    """Neutrals are attempts, not verdicts. Three deliveries and no judgement is
    still 'not measured' — which is exactly the live org's position."""
    snapshot = _snapshot(
        learning_version=3,
        aggregates=(AggregateFact("outcome", "action_executed", 0, 0, 3, 3),),
    )
    monkeypatch.setattr(LR, "learning_snapshot", lambda db, org: _snapshot_of(snapshot, org))

    body = await LR.learning_summary(p=_Principal(), db=None)  # type: ignore[arg-type]

    assert body["totals"]["neutrals"] == 3
    assert body["totals"]["decisive"] == 0
    assert body["totals"]["success_rate"] is None


def _capabilities_gating(handler) -> set[Capability]:
    """Which Capability each `require_capability(...)` guard on a handler holds.

    The guard is the closure `require_capability` returns, so the capability it
    was built with is in its cell contents — the only way to read it back
    without standing up an app and a database.
    """
    found: set[Capability] = set()
    for param in inspect.signature(handler).parameters.values():
        dependency = getattr(param.default, "dependency", None)
        for cell in getattr(dependency, "__closure__", None) or ():
            if isinstance(cell.cell_contents, Capability):
                found.add(cell.cell_contents)
    return found


def test_learning_summary_is_org_scoped_and_capability_gated():
    """The tenant comes from the principal. A `user_id` / `org_id` query
    parameter would be a client-supplied authority — the exact shape
    `/decision-memory` accepted 'for FE compat'."""
    params = set(inspect.signature(LR.learning_summary).parameters)
    assert "org_id" not in params and "user_id" not in params
    assert _capabilities_gating(LR.learning_summary) == {Capability.LEARNING}


def test_outcomes_is_capability_gated():
    assert _capabilities_gating(OR.list_outcomes) == {Capability.OBSERVATION}


def test_decision_history_is_capability_gated():
    assert _capabilities_gating(BR.decisions_history) == {Capability.DECISION}


# ────────────────────────────── /outcomes ────────────────────────────────────


async def test_outcomes_returns_stored_rows_with_lineage(monkeypatch):
    rows = [_outcome(decision_id=str(uuid.uuid4()), signals=[SIGNAL_AJITESH])]
    fake = _FakeOutcomes(rows)
    monkeypatch.setattr(OR, "OutcomeRepository", lambda db: fake)

    body = await OR.list_outcomes(p=_Principal(), db=None)  # type: ignore[arg-type]

    assert body["total"] == 1
    row = body["outcomes"][0]
    assert row["outcome_type"] == "action_executed"
    assert row["reasoning_signal_ids"] == [SIGNAL_AJITESH]
    assert row["occurred_at"] == T0.isoformat()
    # Reality time and ingest time stay separate questions.
    assert "recorded_at" in row
    assert body["order"] == "occurred_at.desc"
    assert fake.scoped == [ORG]


async def test_outcomes_empty_org_returns_the_envelope_not_a_404(monkeypatch):
    fake = _FakeOutcomes([])
    monkeypatch.setattr(OR, "OutcomeRepository", lambda db: fake)

    body = await OR.list_outcomes(p=_Principal(), db=None)  # type: ignore[arg-type]

    assert body["outcomes"] == [] and body["total"] == 0
    assert body["truncated"] is False


async def test_outcomes_cannot_read_another_tenants_rows(monkeypatch):
    """Org scope is applied by the repository on the principal's id; a filter is
    applied ON TOP of it, never instead of it."""
    theirs = _outcome(decision_id=str(uuid.uuid4()), signals=["x"])
    theirs.organization_id = OTHER_ORG
    fake = _FakeOutcomes([theirs])
    monkeypatch.setattr(OR, "OutcomeRepository", lambda db: fake)

    body = await OR.list_outcomes(p=_Principal(), db=None)  # type: ignore[arg-type]

    assert body["outcomes"] == []
    assert fake.scoped == [ORG]


def test_outcomes_takes_no_tenant_parameter():
    params = set(inspect.signature(OR.list_outcomes).parameters)
    assert "org_id" not in params and "user_id" not in params


# ─────────────────── /brief/decisions/history — identity ─────────────────────


async def test_one_conclusion_across_many_watermarks_is_one_history_row(monkeypatch):
    """THE reason the episode key is not `decision_id`.

    Live, one commitment-slipping signal produced 18 distinct decision ids
    across 18 briefs because `compute_decision_id` hashes the state watermark.
    Keyed on the decision id this is 18 history rows for one thing that happened
    once."""
    briefs = [
        _brief(
            [_decision(
                decision_id=str(uuid.uuid5(uuid.NAMESPACE_DNS, f"wm{i}")),
                signal_id=SIGNAL_AJITESH,
                entity_id=ENTITY_AJITESH,
                watermark=f"wm-{i}",
            )],
            as_of=T0 - timedelta(hours=i),
            watermark=f"wm-{i}",
        )
        for i in range(5)
    ]
    _install_history(monkeypatch, briefs=briefs)

    result = await DH.decision_history(None, ORG)

    assert result["total"] == 1
    episode = result["episodes"][0]
    assert episode["episode_key"] == SIGNAL_AJITESH
    assert episode["episode_key_kind"] == "reasoning_signal"
    # The count is published rather than collapsed — it is how long this has
    # been on the founder's plate.
    assert episode["decision_instances"] == 5
    # Newest brief first, so the published snapshot is the current one.
    assert episode["latest_decision_id"] == str(uuid.uuid5(uuid.NAMESPACE_DNS, "wm0"))
    assert episode["is_current"] is True
    assert episode["first_seen_at"] == (T0 - timedelta(hours=4)).isoformat()
    assert episode["last_seen_at"] == T0.isoformat()


async def test_identical_titles_with_different_entities_stay_two_episodes(monkeypatch):
    """The two live commitment cards. Same sentence, two people, two commitment
    entities, two signals. NEVER deduplicated by title."""
    briefs = [
        _brief(
            [
                _decision(
                    decision_id=str(uuid.uuid4()),
                    signal_id=SIGNAL_AJITESH,
                    entity_id=ENTITY_AJITESH,
                    speaker="ajiteshkr533@gmail.com",
                ),
                _decision(
                    decision_id=str(uuid.uuid4()),
                    signal_id=SIGNAL_ASTRA,
                    entity_id=ENTITY_ASTRA,
                    speaker="astravision.global@gmail.com",
                ),
            ],
            as_of=T0,
            watermark="wm-3",
        )
    ]
    _install_history(monkeypatch, briefs=briefs)

    result = await DH.decision_history(None, ORG)

    assert result["total"] == 2
    episodes = _by_key(result)
    assert set(episodes) == {SIGNAL_AJITESH, SIGNAL_ASTRA}
    assert episodes[SIGNAL_AJITESH]["entity_id"] == ENTITY_AJITESH
    assert episodes[SIGNAL_ASTRA]["entity_id"] == ENTITY_ASTRA
    # Both carry the same headline — and the subject is what tells them apart,
    # from `decisions.explanation`, the same projection Today renders.
    subjects = {
        e["explanation"]["subject"]["email"] for e in result["episodes"]
    }
    assert subjects == {"ajiteshkr533@gmail.com", "astravision.global@gmail.com"}


async def test_a_decision_without_reasoning_falls_back_to_its_own_id(monkeypatch):
    """A chat proposal mints its own decision id and carries no signal. That id
    is not watermark-derived, so it is a stable one-shot episode of its own."""
    chat_decision_id = str(uuid.uuid4())
    execution = _execution(decision_id=chat_decision_id, signals=[])
    _install_history(monkeypatch, executions=[execution])

    result = await DH.decision_history(None, ORG)

    episode = result["episodes"][0]
    assert episode["episode_key"] == chat_decision_id
    assert episode["episode_key_kind"] == "decision"
    assert episode["signal_id"] is None
    assert episode["stages"]["execution"] == "executed"


# ─────────────────── /brief/decisions/history — the join ─────────────────────


async def test_history_joins_the_whole_lifecycle_on_canonical_lineage(monkeypatch):
    """Decision -> choice -> execution -> outcome -> learning, all reached by
    reasoning_signal_id. Nothing here matches on title, subject or body text —
    the execution's payload is deliberately unrelated to the decision's title."""
    decision_id = str(uuid.uuid4())
    briefs = [
        _brief(
            [_decision(
                decision_id=decision_id,
                signal_id=SIGNAL_AJITESH,
                entity_id=ENTITY_AJITESH,
                signal_type="waiting_on_user_reply",
                category="waiting_reply",
                title="Reply needed",
                speaker="ajiteshkr533@gmail.com",
            )],
            as_of=T0,
            watermark="wm-3",
        )
    ]
    _install_history(
        monkeypatch,
        briefs=briefs,
        executions=[_execution(decision_id=decision_id, signals=[SIGNAL_AJITESH])],
        outcomes=[_outcome(decision_id=decision_id, signals=[SIGNAL_AJITESH])],
        learning=[_learning(
            decision_id=decision_id, signals=[SIGNAL_AJITESH],
            result="neutral", surface="delivery",
        )],
    )

    result = await DH.decision_history(None, ORG)

    assert result["total"] == 1
    episode = result["episodes"][0]
    assert episode["stages"] == {
        "decision": "open",
        "choice": "acted",          # an execution was approved by a human
        "execution": "executed",
        "outcome": "observed",
        "learning": "neutral",       # Learning's own verdict, not a second one
    }
    assert episode["execution"]["status"] == "executed"
    assert len(episode["outcomes"]) == 1
    assert episode["learning"][0]["outcome_surface"] == "delivery"


async def test_unrelated_lineage_never_joins(monkeypatch):
    """An outcome for a different signal must not attach to this episode, even
    when everything else about the two looks alike."""
    decision_id = str(uuid.uuid4())
    briefs = [
        _brief(
            [_decision(decision_id=decision_id, signal_id=SIGNAL_AJITESH, entity_id=ENTITY_AJITESH)],
            as_of=T0, watermark="wm-3",
        )
    ]
    _install_history(
        monkeypatch,
        briefs=briefs,
        outcomes=[_outcome(decision_id=str(uuid.uuid4()), signals=[SIGNAL_ASTRA])],
    )

    result = await DH.decision_history(None, ORG)
    episodes = _by_key(result)

    assert episodes[SIGNAL_AJITESH]["outcomes"] == []
    assert episodes[SIGNAL_AJITESH]["stages"]["outcome"] == "none"
    # The stray outcome becomes its OWN episode rather than being dropped.
    assert SIGNAL_ASTRA in episodes


async def test_a_finding_reports_not_applicable_and_never_a_zero(monkeypatch):
    """A `risk` decision maps to no action type, so it has no execution and
    never will. Reporting that as 0 is indistinguishable from a reply that was
    proposed and never sent."""
    briefs = [
        _brief(
            [_decision(
                decision_id=str(uuid.uuid4()),
                signal_id=SIGNAL_AJITESH,
                entity_id=ENTITY_AJITESH,
            )],
            as_of=T0, watermark="wm-3",
        )
    ]
    _install_history(monkeypatch, briefs=briefs)

    episode = (await DH.decision_history(None, ORG))["episodes"][0]

    assert episode["stages"]["execution"] == "not_applicable"
    assert episode["stages"]["outcome"] == "none"
    assert episode["stages"]["learning"] == "none"
    assert episode["stages"]["choice"] == "pending"
    assert episode["execution"] is None
    assert episode["choice"] is None


async def test_dismissal_resolves_through_the_decision_id_index(monkeypatch):
    """`decision_ref` is one watermark's decision id; the episode is keyed on the
    signal. The index built while folding the briefs is what connects them."""
    old_id, new_id = str(uuid.uuid4()), str(uuid.uuid4())
    briefs = [
        _brief([_decision(decision_id=new_id, signal_id=SIGNAL_AJITESH,
                          entity_id=ENTITY_AJITESH, watermark="wm-4")],
               as_of=T0, watermark="wm-4"),
        _brief([_decision(decision_id=old_id, signal_id=SIGNAL_AJITESH,
                          entity_id=ENTITY_AJITESH, watermark="wm-3")],
               as_of=T0 - timedelta(days=1), watermark="wm-3"),
    ]
    # Dismissed at the OLDER watermark — the id the founder's tab was holding.
    _install_history(monkeypatch, briefs=briefs, dismissals=[_Dismissal(old_id, "stale")])

    episode = (await DH.decision_history(None, ORG))["episodes"][0]

    assert episode["stages"]["choice"] == "dismissed"
    assert episode["choice"]["reason"] == "stale"
    assert episode["choice"]["decision_ref"] == old_id
    # A dismissal is a judgement; its outcome has not been recorded yet here.
    assert episode["stages"]["outcome"] == "pending"
    assert episode["stages"]["learning"] == "pending"
    # Internal sort key never reaches the wire.
    assert not any(k.startswith("_") for k in episode["choice"])


async def test_history_is_org_scoped_at_every_read(monkeypatch):
    """All five boundaries receive the principal's org and nothing wider."""
    fakes = _install_history(
        monkeypatch,
        briefs=[_brief([_decision(decision_id=str(uuid.uuid4()), signal_id=SIGNAL_AJITESH,
                                  entity_id=ENTITY_AJITESH)], as_of=T0, watermark="wm-3")],
        dismissals=[_Dismissal(str(uuid.uuid4()), organization_id=OTHER_ORG)],
    )

    await DH.decision_history(None, ORG)

    assert fakes.decisions.scoped == [ORG, ORG]
    assert fakes.executions.scoped == [ORG]
    assert fakes.outcomes.scoped == [ORG]
    assert fakes.learning == [ORG]


async def test_history_reports_its_own_bounds(monkeypatch):
    """A truncated history must be visible on the wire, not look complete."""
    briefs = [
        _brief([_decision(decision_id=str(uuid.uuid4()),
                          signal_id=str(uuid.uuid5(uuid.NAMESPACE_DNS, f"s{i}")),
                          entity_id=f"e{i}")],
               as_of=T0 - timedelta(hours=i), watermark=f"wm-{i}")
        for i in range(4)
    ]
    _install_history(monkeypatch, briefs=briefs)

    result = await DH.decision_history(None, ORG, limit=2)

    assert result["total"] == 2
    assert result["available"] == 4
    assert result["scanned"]["briefs"] == 4
    assert result["scanned"]["brief_limit"] == DH.MAX_BRIEFS_SCANNED


async def test_history_never_rebuilds_a_brief(monkeypatch):
    """Opening the Decisions page must not fold State, run Reasoning or
    re-project Intelligence. `/brief/today` legitimately can; this must not."""
    import navis.orchestration.brief as brief_module

    called: list[str] = []
    monkeypatch.setattr(
        brief_module, "build_brief",
        lambda *a, **k: called.append("build") or pytest.fail("history rebuilt a brief"),
    )
    _install_history(monkeypatch)

    await DH.decision_history(None, ORG)

    assert called == []


# ───────────────── DECISION_DISMISSED -> Outcome -> Learning ─────────────────


def _dismiss_event(*, event_id: str, signals: list[str], decision_id: str, reason="explicit"):
    """The payload `brief_router.dismiss_decision` publishes."""
    from navis.platform.events import DomainEvent, EventType

    return DomainEvent(
        type=EventType.DECISION_DISMISSED,
        org_id=ORG,
        source="brief.router",
        payload={
            "event_id": event_id,
            "dismissal_id": event_id,
            "decision_ref": decision_id,
            "reason": reason,
            "user_id": str(USER),
            "decision_id": decision_id,
            "reasoning_signal_ids": signals,
            "reasoning_version": 4,
            "state_watermark": "wm-3",
        },
    )


async def test_dismiss_records_exactly_one_outcome_with_lineage():
    from tests.observation.fakes import FakeOutcomeRepo, FakePublisher
    from navis.observation.service import ObservationService

    repo = FakeOutcomeRepo()
    pub = FakePublisher()
    svc = ObservationService(repo=repo, publisher=pub)
    decision_id = str(uuid.uuid4())
    event = _dismiss_event(event_id="dis-1", signals=[SIGNAL_AJITESH], decision_id=decision_id)

    out = await svc.record_from_execution(event, OutcomeType.DECISION_DISMISSED, "dis-1")

    assert out is not None
    assert out.outcome_type == "decision_dismissed"
    assert out.reasoning_signal_ids == [SIGNAL_AJITESH]
    assert str(out.decision_id) == decision_id
    assert out.state_watermark == "wm-3"
    # The founder's stated grounds, preserved verbatim.
    assert out.payload["reason"] == "explicit"
    # A dismissed decision has no action. The key is ABSENT rather than null, so
    # "not applicable" stays distinguishable from "applicable but missing".
    assert "action_type" not in out.payload
    assert len(repo.rows) == 1
    assert pub.events[0].payload["outcome_type"] == "decision_dismissed"


async def test_re_dismissing_is_idempotent():
    """`dismissal_id` is uuid5(org|ref|user), so a re-click yields the same event
    id, the same outcome id, and no second row."""
    from tests.observation.fakes import FakeOutcomeRepo, FakePublisher
    from navis.observation.inbox import process_once
    from navis.observation.service import ObservationService

    repo = FakeOutcomeRepo()
    svc = ObservationService(repo=repo, publisher=FakePublisher())
    event = _dismiss_event(event_id="dis-1", signals=[SIGNAL_AJITESH],
                           decision_id=str(uuid.uuid4()))

    async def handler():
        await svc.record_from_execution(event, OutcomeType.DECISION_DISMISSED, "dis-1")

    assert await process_once(repo, "observation", "dis-1", handler) is True
    assert await process_once(repo, "observation", "dis-1", handler) is False
    assert len(repo.rows) == 1


async def test_dismiss_outcome_becomes_a_negative_judgement_in_learning():
    """Through the EXISTING Learning path — no shortcut from the router."""
    from tests.learning.fakes import FakeLearningRepo, FakePublisher, agg
    from navis.learning.aggregates import AggregateDimension
    from navis.learning.service import LearningService

    repo = FakeLearningRepo()
    svc = LearningService(repo=repo, publisher=FakePublisher())
    decision_id = uuid.uuid4()

    recorded = await svc.record_from_outcome({
        "outcome_id": str(uuid.uuid4()),
        "org_id": str(ORG),
        "outcome_type": "decision_dismissed",
        "execution_id": None,
        "decision_id": str(decision_id),
        "reasoning_signal_ids": [SIGNAL_AJITESH],
        "reasoning_version": 4,
        "state_watermark": "wm-3",
        "provider_reference": {},
        "occurred_at": T0.isoformat(),
        "payload": {"reason": "explicit"},
    })

    assert recorded is True
    event = next(iter(repo.events.values()))
    assert event.result == "failure"
    assert event.outcome_surface == "judgement"
    # A judgement is DECISIVE about the reasoning that raised the decision —
    # unlike a delivery outcome, which is only ever counted as an attempt.
    signal_agg = agg(repo, AggregateDimension.SIGNAL.value, SIGNAL_AJITESH, org=ORG)
    assert (signal_agg.successes, signal_agg.failures) == (0, 1)


def test_dismiss_subscriber_is_registered_and_brief_opened_is_not():
    """The loop is closed for a founder's verdict, and deliberately open for a
    page view — a brief open says nothing about whether its decisions were
    right, and would add one neutral sample per signal per open."""
    import navis.observation.subscribers  # noqa: F401  registers handlers
    from navis.platform.events import EventType, event_bus

    dismissed = event_bus._handlers.get(EventType.DECISION_DISMISSED, [])
    assert [h.__name__ for h in dismissed] == ["on_decision_dismissed"]
    assert event_bus._handlers.get(EventType.BRIEF_OPENED, []) == []


def test_learning_knows_the_whole_observation_vocabulary():
    """Adding an OutcomeType without teaching Learning what it means leaves it
    silently NEUTRAL — which is how `action_rejected` went unlearned for months."""
    from navis.learning.attribution import _OUTCOME_VOCABULARY

    assert {t.value for t in OutcomeType} == set(_OUTCOME_VOCABULARY)


# ───────────────────────── regression: /brief/today ──────────────────────────


def test_brief_today_serializer_contract_is_unchanged():
    """The Decisions page must not have moved Today's wire shape."""
    brief = _brief([], as_of=T0, watermark="wm-3")
    body = BR._serialize(brief, obligations={"available": False}, freshness={"current": True})

    assert set(body) == {
        "id", "org_id", "as_of", "reasoning_version", "state_watermark",
        "freshness", "decisions", "obligations",
    }


async def test_dismiss_response_shape_is_unchanged(monkeypatch):
    """The dismiss route gained lineage ON THE EVENT. Its persisted row and its
    HTTP response are untouched, so an existing client sees no change."""
    published: list[Any] = []

    class _DB:
        async def execute(self, *_a, **_k):
            return None

        async def commit(self):
            return None

    async def _publish(event):
        published.append(event)

    monkeypatch.setattr(BR.event_bus, "publish", _publish)
    monkeypatch.setattr(
        BR, "DecisionService",
        lambda db: SimpleNamespace(find_decision=_find_none),
    )
    ctx = SimpleNamespace(org_id=ORG, user=SimpleNamespace(id=USER))

    body = await BR.dismiss_decision(
        "dec-ref", BR.DismissDecisionBody(reason="duplicate"),
        ctx=ctx, _=None, db=_DB(),  # type: ignore[arg-type]
    )

    assert set(body) == {"status", "dismissal_id"}
    assert body["status"] == "dismissed"
    # Unresolvable ref: lineage is absent, never invented from a neighbour.
    payload = published[0].payload
    assert payload["decision_id"] is None
    assert payload["reasoning_signal_ids"] == []
    assert payload["event_id"] == body["dismissal_id"]


async def _find_none(org_id, decision_ref):
    return None
