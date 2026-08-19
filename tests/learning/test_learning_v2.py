"""Learning V2: attribution, idempotency, determinism, replay parity, lineage."""
from __future__ import annotations

import random

from navis.learning.aggregates import success_rate
from navis.learning.attribution import (
    classify,
    classify_surface,
    learning_event_values,
)
from navis.learning.models import (
    AggregateDimension,
    LearningEvent,
    LearningResult,
    OutcomeSurface,
    learning_event_id,
)
from navis.learning.replay import LearningReplayService
from navis.learning.service import LearningService
from tests.learning.fakes import (
    DECISION,
    ORG_A,
    ORG_B,
    FakeLearningRepo,
    FakePublisher,
    agg,
    outcome_payload,
)


def _svc(repo=None, pub=None):
    return LearningService(repo=repo or FakeLearningRepo(), publisher=pub or FakePublisher())


# ---- attribution correctness (the P0 fix) ----

def test_classify_maps_outcome_type_to_result():
    assert classify("reply_received") == LearningResult.SUCCESS
    assert classify("meeting_completed") == LearningResult.SUCCESS
    assert classify("action_failed") == LearningResult.FAILURE
    assert classify("action_executed") == LearningResult.NEUTRAL


def test_every_outcome_type_observation_emits_is_classified():
    """The vocabulary is Observation's, and Learning must know all of it.

    ACTION_REJECTED and ACTION_CANCELLED were emitted by Execution, recorded by
    Observation and absent from Learning's table, so both fell through to
    NEUTRAL — a founder rejecting a proposal was learned as "no information".
    This fails if either side adds a member the other has not been taught.
    """
    from navis.learning.attribution import _OUTCOME_VOCABULARY
    from navis.observation.models import OutcomeType

    emitted = {t.value for t in OutcomeType}
    assert emitted == set(_OUTCOME_VOCABULARY), (
        "Observation and Learning disagree about the outcome vocabulary: "
        f"unclassified={sorted(emitted - set(_OUTCOME_VOCABULARY))} "
        f"unknown_to_observation={sorted(set(_OUTCOME_VOCABULARY) - emitted)}"
    )


def test_a_human_rejection_is_a_decisive_failure_not_a_shrug():
    """The clearest negative signal the product can receive.

    Before this, `action_rejected` hit `classify`'s default and was recorded as
    NEUTRAL, so a signal whose every proposal a founder declined kept a perfect
    success rate.
    """
    assert classify("action_rejected") == LearningResult.FAILURE
    assert classify_surface("action_rejected") == OutcomeSurface.JUDGEMENT


def test_a_cancellation_is_not_scored_as_a_rejection():
    """Observation keeps these apart deliberately — a rejection is a verdict on
    the proposal, a cancellation is a withdrawal overtaken by events. Scoring the
    withdrawal as a failure would perform, one layer down, exactly the collapse
    Observation refused to make."""
    assert classify("action_cancelled") == LearningResult.NEUTRAL
    assert classify_surface("action_cancelled") == OutcomeSurface.JUDGEMENT


def test_an_outcome_type_learning_was_never_taught_stays_inert():
    """`outcome_type` is a plain String upstream so a new member needs no
    migration. It must be unable to move a rate until someone decides what it
    means."""
    assert classify("teleported_into_the_ocean") == LearningResult.NEUTRAL
    assert classify_surface("teleported_into_the_ocean") == OutcomeSurface.DELIVERY


async def test_attributes_to_each_reasoning_signal_not_subject_ref():
    repo = FakeLearningRepo()
    svc = LearningService(repo=repo, publisher=FakePublisher())
    await svc.record_from_outcome(
        outcome_payload(outcome_id="o1", outcome_type="reply_received", signals=["sig-A", "sig-B"])
    )
    # Each signal got attributed (no single "execution" bucket).
    assert agg(repo, AggregateDimension.SIGNAL.value, "sig-A").successes == 1
    assert agg(repo, AggregateDimension.SIGNAL.value, "sig-B").successes == 1
    assert agg(repo, AggregateDimension.DECISION.value, str(DECISION)).successes == 1
    assert agg(repo, AggregateDimension.OUTCOME.value, "reply_received").successes == 1


async def test_failure_and_success_accumulate_per_signal():
    repo = FakeLearningRepo()
    svc = LearningService(repo=repo, publisher=FakePublisher())
    await svc.record_from_outcome(outcome_payload(outcome_id="o1", outcome_type="reply_received", signals=["s"]))
    await svc.record_from_outcome(outcome_payload(outcome_id="o2", outcome_type="reply_received", signals=["s"]))
    # A WORLD failure — the customer never answered. Decisive about the signal,
    # unlike `action_failed`, which is decisive only about the channel.
    await svc.record_from_outcome(outcome_payload(outcome_id="o3", outcome_type="no_response", signals=["s"]))
    a = agg(repo, AggregateDimension.SIGNAL.value, "s")
    assert (a.successes, a.failures) == (2, 1)
    assert success_rate(a.successes, a.failures) == 0.6667


# ---- delivery outcome is NOT world outcome ----

async def test_a_channel_failure_does_not_count_against_the_signal():
    """THE SEPARATION. `action_failed` means the send broke — a lost Google
    credential, a dead channel, an exhausted retry budget. None of that is
    evidence that firing the signal was the wrong call, yet it used to increment
    `failures` on that signal, making a week of Gmail outages read identically to
    a week of customers saying no."""
    repo = FakeLearningRepo()
    svc = LearningService(repo=repo, publisher=FakePublisher())
    await svc.record_from_outcome(
        outcome_payload(outcome_id="o1", outcome_type="action_failed", signals=["s"],
                        payload={"failure_kind": "credential_unavailable"})
    )
    a = agg(repo, AggregateDimension.SIGNAL.value, "s")
    assert (a.successes, a.failures) == (0, 0), "a channel fault was blamed on the reasoning"
    # Recorded, never lost: the attempt still counts toward the sample.
    assert (a.neutrals, a.total) == (1, 1)
    assert success_rate(a.successes, a.failures) is None


async def test_a_delivery_failure_is_still_decisive_at_the_outcome_dimension():
    """Nothing is lost by not blaming the signal. The OUTCOME dimension is keyed
    ON the outcome_type, so delivery health stays fully measurable there."""
    repo = FakeLearningRepo()
    svc = LearningService(repo=repo, publisher=FakePublisher())
    await svc.record_from_outcome(
        outcome_payload(outcome_id="o1", outcome_type="action_failed", signals=["s"])
    )
    o = agg(repo, AggregateDimension.OUTCOME.value, "action_failed")
    assert (o.successes, o.failures, o.total) == (0, 1, 1)


async def test_executed_replied_and_accepted_stay_three_separate_facts():
    """Gmail: action_executed != customer_replied != customer_accepted.

    One execution produces three outcomes over time. Each must remain its own
    learning event and its own outcome counter — never folded into one score.
    """
    repo = FakeLearningRepo()
    svc = LearningService(repo=repo, publisher=FakePublisher())
    for oid, otype in (
        ("o1", "action_executed"), ("o2", "reply_received"), ("o3", "meeting_accepted"),
    ):
        await svc.record_from_outcome(
            outcome_payload(outcome_id=oid, outcome_type=otype, signals=["s"])
        )

    assert len(repo.events) == 3, "the three facts collapsed into one"
    surfaces = {e.outcome_type: e.outcome_surface for e in repo.events.values()}
    assert surfaces == {
        "action_executed": OutcomeSurface.DELIVERY.value,
        "reply_received": OutcomeSurface.WORLD.value,
        "meeting_accepted": OutcomeSurface.WORLD.value,
    }
    # Three separate outcome counters, not one aggregate "success".
    for otype in ("action_executed", "reply_received", "meeting_accepted"):
        assert agg(repo, AggregateDimension.OUTCOME.value, otype).total == 1
    # And only the two WORLD facts moved the signal's rate.
    s = agg(repo, AggregateDimension.SIGNAL.value, "s")
    assert (s.successes, s.failures, s.neutrals) == (2, 0, 1)


# ---- idempotency: duplicate delivery collapses ----

async def test_duplicate_outcome_does_not_double_count():
    repo = FakeLearningRepo()
    svc = LearningService(repo=repo, publisher=FakePublisher())
    p = outcome_payload(outcome_id="dup", outcome_type="reply_received", signals=["s"])
    first = await svc.record_from_outcome(p)
    second = await svc.record_from_outcome(p)  # at-least-once redelivery
    assert first is True and second is False
    assert agg(repo, AggregateDimension.SIGNAL.value, "s").successes == 1
    assert len(repo.events) == 1


def test_learning_event_id_is_deterministic():
    assert learning_event_id("o1") == learning_event_id("o1")
    assert learning_event_id("o1") != learning_event_id("o2")


# ---- determinism: replay/rebuild parity + ordering independence ----

async def test_rebuild_matches_live_aggregates():
    repo = FakeLearningRepo()
    svc = LearningService(repo=repo, publisher=FakePublisher())
    outcomes = [
        ("o1", "reply_received", ["s1"]),
        ("o2", "action_failed", ["s1", "s2"]),
        ("o3", "meeting_accepted", ["s2"]),
        ("o4", "action_executed", ["s1"]),
    ]
    for oid, ot, sig in outcomes:
        await svc.record_from_outcome(outcome_payload(outcome_id=oid, outcome_type=ot, signals=sig))
    live = {(a.dimension, a.key): (a.successes, a.failures, a.neutrals, a.total)
            for a in repo.aggs.values()}

    await LearningReplayService(db=None, repo=repo).rebuild(ORG_A)  # delete + replay
    rebuilt = {(a.dimension, a.key): (a.successes, a.failures, a.neutrals, a.total)
               for a in repo.aggs.values()}
    assert rebuilt == live  # identical projection from the immutable log


async def test_ordering_independence():
    outcomes = [
        ("o1", "reply_received", ["s1"]),
        ("o2", "action_failed", ["s1"]),
        ("o3", "meeting_completed", ["s1"]),
    ]

    async def run(order):
        repo = FakeLearningRepo()
        svc = LearningService(repo=repo, publisher=FakePublisher())
        for oid, ot, sig in order:
            await svc.record_from_outcome(outcome_payload(outcome_id=oid, outcome_type=ot, signals=sig))
        a = agg(repo, AggregateDimension.SIGNAL.value, "s1")
        return (a.successes, a.failures, a.neutrals)

    shuffled = outcomes[:]
    random.Random(3).shuffle(shuffled)
    assert await run(outcomes) == await run(shuffled)  # counts commute


# ---- multi-tenancy ----

async def test_multi_tenant_isolation():
    repo = FakeLearningRepo()
    svc = LearningService(repo=repo, publisher=FakePublisher())
    await svc.record_from_outcome(outcome_payload(org_id=ORG_A, outcome_id="a", outcome_type="reply_received", signals=["s"]))
    await svc.record_from_outcome(outcome_payload(org_id=ORG_B, outcome_id="b", outcome_type="no_response", signals=["s"]))
    # Same signal key in both orgs; the counters must not touch.
    assert agg(repo, AggregateDimension.SIGNAL.value, "s", org=ORG_A).successes == 1
    assert agg(repo, AggregateDimension.SIGNAL.value, "s", org=ORG_A).failures == 0
    assert agg(repo, AggregateDimension.SIGNAL.value, "s", org=ORG_B).failures == 1
    assert agg(repo, AggregateDimension.SIGNAL.value, "s", org=ORG_B).successes == 0


# ---- lineage completeness ----

def test_learning_event_carries_full_lineage():
    import uuid

    values = learning_event_values(
        outcome_payload(outcome_id=str(uuid.uuid4()), outcome_type="reply_received", signals=["s1"])
    )
    for field in ("outcome_id", "execution_id", "decision_id", "reasoning_signal_ids",
                  "reasoning_version", "state_watermark", "occurred_at", "result"):
        assert values.get(field) is not None


def test_event_without_occurred_at_or_lineage_is_skipped():
    bad = outcome_payload(outcome_id="o1", outcome_type="reply_received")
    bad.pop("occurred_at")
    assert learning_event_values(bad) is None  # determinism guard: no now() fallback


def test_learning_keeps_the_provider_receipt_and_the_failure_cause():
    """Both were published by Observation and dropped by Learning.

    Without them a learning event cannot answer "which message was that?" or
    "why did it fail?", and the only way to recover either was to read the
    `outcomes` table — which Learning is forbidden from doing.
    """
    import uuid

    values = learning_event_values(
        outcome_payload(
            outcome_id=str(uuid.uuid4()),
            outcome_type="action_failed",
            provider_reference={"provider": "gmail", "external_id": "msg-77"},
            payload={"action_type": "send_email", "failure_kind": "retry_exhausted",
                     "error": "gmail 503"},
        )
    )
    assert values["provider_reference"] == {"provider": "gmail", "external_id": "msg-77"}
    assert values["failure_kind"] == "retry_exhausted"


def test_occurred_at_is_business_time_never_ingestion_time():
    """Observation separates reality time from ingest time; Learning must record
    the reality one. A now() fallback here would silently restamp a webhook that
    arrived three days late."""
    from datetime import datetime, timezone

    values = learning_event_values(
        outcome_payload(outcome_id="o1", outcome_type="reply_received",
                        occurred_at="2026-06-01T12:00:00+00:00")
    )
    assert values["occurred_at"] == datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)


def test_the_migration_backfill_matches_the_live_classifier():
    """The 0040 backfill derives `outcome_surface` from `outcome_type` with its
    own CASE. If the two ever disagree, historical rows get a surface the running
    code would never assign."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "migrations" / "versions" / "0040_learning_outcome_surface.py"
    spec = importlib.util.spec_from_file_location("m0040", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for outcome_type, surface in module._SURFACE_BY_TYPE.items():
        assert classify_surface(outcome_type).value == surface, outcome_type
    from navis.learning.attribution import _OUTCOME_VOCABULARY

    assert set(module._SURFACE_BY_TYPE) == set(_OUTCOME_VOCABULARY)


# ---- patterns ----

async def test_patterns_materialize_for_reliable_signals():
    repo = FakeLearningRepo()
    svc = LearningService(repo=repo, publisher=FakePublisher())
    for i in range(6):  # >= MIN_SAMPLES, all successes -> reliable
        await svc.record_from_outcome(
            outcome_payload(outcome_id=f"o{i}", outcome_type="reply_received", signals=["s"])
        )
    facts = list(repo.patterns.values())
    assert facts and facts[0].kind == "reliable" and facts[0].success_rate == 1.0


# ---- no-intelligence guard ----

def test_learning_stores_no_intelligence():
    cols = set(LearningEvent.__table__.columns.keys())
    forbidden = {"risk", "priority", "recommendation", "intelligence", "prediction", "confidence", "score"}
    assert cols & forbidden == set()


def test_learning_subscribes_outcome_observed():
    import navis.learning.subscribers  # noqa: F401  registers handler
    from navis.platform.events import event_bus
    from navis.platform.events.events import EventType

    names = [h.__name__ for h in event_bus._handlers.get(EventType.OUTCOME_OBSERVED, [])]
    assert "on_outcome_observed" in names


# ---- OUTCOME_OBSERVED is the ONLY input ----

def test_learning_never_reaches_back_into_execution_observation_or_a_provider():
    """Learning reconstructs nothing. Everything it records arrives on the event,
    which is what makes replaying that event reproduce the same row.

    An AST check rather than a mock, because the failure mode is a single
    convenience import added later ("just look up the execution row") and that
    would not fail any behavioural test until a replay diverged in production.
    """
    import ast
    from pathlib import Path

    import navis.learning as pkg

    forbidden = (
        "navis.execution", "navis.observation", "navis.signals", "navis.memory",
        "navis.state", "navis.reasoning", "navis.decisions", "navis.intelligence",
        "navis.adaptive", "navis.ml",
    )
    violations = []
    for py in Path(pkg.__file__).parent.rglob("*.py"):
        tree = ast.parse(py.read_text(), filename=str(py))
        for node in ast.walk(tree):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods = [node.module]
            for mod in mods:
                if mod.startswith(forbidden):
                    violations.append(f"{py.name}:{node.lineno} imports {mod}")
    assert not violations, (
        "Learning may only consume OUTCOME_OBSERVED:\n" + "\n".join(violations)
    )


def test_intelligence_consumes_learning_through_the_contract_not_the_tables():
    """Learning's storage is Learning's business.

    `IntelligenceRepository` used to SELECT learning_events / learning_aggregates
    / learning_patterns itself, so a second model of Learning's schema lived in
    another layer. It must go through `navis.learning.metrics` instead.
    """
    import ast
    from pathlib import Path

    import navis.intelligence.repository as repo_mod

    source = Path(repo_mod.__file__).read_text()
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    assert "navis.learning.metrics" in imported, "Intelligence bypassed the contract"
    for orm in ("LearningEvent", "LearningAggregate", "LearningPattern"):
        assert orm not in source, (
            f"Intelligence still reads Learning's {orm} table directly"
        )
