"""Learning V2: attribution, idempotency, determinism, replay parity, lineage."""
from __future__ import annotations

import random

from navis.learning.aggregates import success_rate
from navis.learning.attribution import classify, learning_event_values
from navis.learning.models import (
    AggregateDimension,
    LearningEvent,
    LearningResult,
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
    await svc.record_from_outcome(outcome_payload(outcome_id="o3", outcome_type="action_failed", signals=["s"]))
    a = agg(repo, AggregateDimension.SIGNAL.value, "s")
    assert (a.successes, a.failures) == (2, 1)
    assert success_rate(a.successes, a.failures) == 0.6667


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
    await svc.record_from_outcome(outcome_payload(org_id=ORG_B, outcome_id="b", outcome_type="action_failed", signals=["s"]))
    assert agg(repo, AggregateDimension.SIGNAL.value, "s", org=ORG_A).successes == 1
    assert agg(repo, AggregateDimension.SIGNAL.value, "s", org=ORG_A).failures == 0
    assert agg(repo, AggregateDimension.SIGNAL.value, "s", org=ORG_B).failures == 1


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
