"""PostgreSQL integration tests for the stable-signal / temporal-observation split.

Runs only with NAVIS_TEST_DATABASE_URL. Exercises what the in-memory fakes cannot:
the real ON CONFLICT DO NOTHING behaviour on two different primary keys, the
foreign key from observation to signal, the NULL-watermark idempotency that a
plain UNIQUE index would not provide, and the snapshot-scoped obligation count.

THE GOLDEN COMMITMENT drives most of it — "Thanks for following up on this — I'll
come back to you on it." (org 071ae55b, commitment f33778d8). Production stores
that conclusion once, at 7.16 days old and confidence 0.7386, while the decision
citing that very row reports 11.85 days and 0.8951. Both were computed by the
engines; ON CONFLICT DO NOTHING on a single row discarded the second. These tests
pin the behaviour that makes that impossible.

A new file rather than an addition to an existing one: the table, its identity
rule and its conflict semantics are new, no existing integration module owns the
reasoning store, and folding them into the brief/execution enrichment suite would
put schema-level guarantees under a subject that is not about them.
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from navis.reasoning.store import (
    ReasoningSignalObservation,
    compute_observation_id,
    observation_history,
    observations_at,
    persist_signals,
    resolve_signals,
    signal_type_counts_at,
)
from navis.reasoning.v2.contract import (
    REASONING_VERSION,
    ReasoningSignal,
    SignalType,
    compute_signal_id,
)

pytestmark = pytest.mark.skipif(
    not os.environ.get("NAVIS_TEST_DATABASE_URL"),
    reason="set NAVIS_TEST_DATABASE_URL to run reasoning-observation Postgres tests",
)

ORG = uuid.UUID("071ae55b-4654-499b-ae38-0054226231ec")
COMMITMENT = "f33778d8-a54e-50bb-87b9-2d74d83584e2"

# The two folds the production data actually recorded.
AS_OF_1 = datetime(2026, 8, 17, 16, 11, 12, tzinfo=timezone.utc)
AS_OF_2 = datetime(2026, 8, 22, 8, 48, 21, tzinfo=timezone.utc)
WM_1 = "f7b48012-d304-5827-bda6-c8094890c7a4"
WM_2 = "f8675a7f-77cc-538d-9bd3-f198344f4b53"

# The TRUE ages, recovered from the stored confidences rather than from the
# rounded `evidence.age_days`. `age = (confidence - 0.5) * 30` inverts
# `CommitmentReasoningEngine`'s transform, so 0.7386435185 -> 7.1593 and
# 0.8951 -> 11.853. Using the display-rounded 7.16 / 11.85 as INPUT reproduces
# neither stored confidence (7.16 yields 0.7387, not 0.7386) — the two folds are
# 4.6925 days apart, which is exactly AS_OF_2 - AS_OF_1.
AGE_1 = 7.1593
AGE_2 = 11.853
# What the engine's transforms produce from those ages — the values production
# actually holds for signal d9170d07.
CONF_1, SEV_1 = 0.7386, 0.6386
CONF_2, SEV_2 = 0.8951, 0.7951


@pytest.fixture()
async def db():
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    import navis.registry  # noqa: F401  (register every ORM table)
    from navis.platform.database import Base

    engine = create_async_engine(os.environ["NAVIS_TEST_DATABASE_URL"])
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    async with Session() as session:
        yield session
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


async def _org(db, org_id: uuid.UUID = ORG) -> None:
    from navis.identity.models import Organization

    db.add(Organization(id=org_id, name="Acme", slug=f"acme-{uuid.uuid4().hex[:8]}"))
    await db.flush()


def _commitment_signal(
    *, as_of: datetime, watermark: str | None, age_days: float, org_id: uuid.UUID = ORG
) -> ReasoningSignal:
    """The golden commitment as `CommitmentReasoningEngine` emits it at one fold.

    Confidence and severity use the engine's own transforms
    (`min(0.9, 0.5 + age/30)` / `min(0.9, 0.4 + age/30)`) so the values here are
    the ones production actually stored, not invented stand-ins.
    """
    return ReasoningSignal(
        signal_id=compute_signal_id(
            org_id, REASONING_VERSION, "commitment_reasoning", "2.0.0",
            SignalType.COMMITMENT_SLIPPING, COMMITMENT, COMMITMENT,
        ),
        org_id=org_id,
        signal_type=SignalType.COMMITMENT_SLIPPING,
        entity_type="commitment",
        entity_id=COMMITMENT,
        title="Commitment slipping: Thanks for following up on this",
        why=f"Open for {age_days:.0f} days with no newer fact in Memory.",
        evidence={"age_days": round(age_days, 2), "speaker_email": "x@example.com"},
        confidence=min(0.9, 0.5 + age_days / 30),
        severity=min(0.9, 0.4 + age_days / 30),
        engine_name="commitment_reasoning",
        engine_version="2.0.0",
        state_as_of=as_of,
        state_watermark=watermark,
        reasoning_version=REASONING_VERSION,
        created_at=as_of,
    )


async def test_golden_commitment_keeps_one_identity_and_gains_an_observation_per_fold(db):
    """THE PHASE 2 ACCEPTANCE TEST.

    Same conclusion, two folds, four and a bit days apart. One stable id; two
    observations; both queryable; the newer one no longer discarded.
    """
    await _org(db)
    first = _commitment_signal(as_of=AS_OF_1, watermark=WM_1, age_days=AGE_1)
    second = _commitment_signal(as_of=AS_OF_2, watermark=WM_2, age_days=AGE_2)

    assert await persist_signals(db, [first]) == 1
    assert await persist_signals(db, [second]) == 1
    await db.flush()

    # IDENTITY IS STABLE. This is what keeps every stored reference in
    # execution_requests / learning_events / intelligence_insights resolvable.
    assert first.signal_id == second.signal_id

    # ...and exactly one identity row exists for it.
    resolved = await resolve_signals(db, ORG, [first.signal_id])
    assert list(resolved) == [str(first.signal_id)]
    # The identity row keeps FIRST-detection values — documented, not current.
    assert resolved[str(first.signal_id)].state_watermark == WM_1
    assert round(resolved[str(first.signal_id)].confidence, 4) == CONF_1

    # OBSERVATIONS ARE DISTINCT, and both survive.
    history = await observation_history(db, ORG, first.signal_id)
    assert len(history) == 2
    assert [h.state_watermark for h in history] == [WM_1, WM_2]     # oldest first
    assert [h.evidence["age_days"] for h in history] == [round(AGE_1, 2), round(AGE_2, 2)]
    assert round(history[0].confidence, 4) == CONF_1
    assert round(history[1].confidence, 4) == CONF_2
    assert history[0].why.startswith("Open for 7 days")
    assert history[1].why.startswith("Open for 12 days")

    # THE LATEST OBSERVATION IS DETERMINISTIC — ordered by logical time with the
    # id as a total-order tiebreak, so it is the same row on every query.
    assert history[-1].state_as_of == AS_OF_2
    assert (await observation_history(db, ORG, first.signal_id))[-1].observation_id == (
        history[-1].observation_id
    )

    # SNAPSHOT-SPECIFIC READS RESOLVE TO THE RIGHT ONE.
    at_1 = await observations_at(db, ORG, [first.signal_id], state_watermark=WM_1)
    at_2 = await observations_at(db, ORG, [first.signal_id], state_watermark=WM_2)
    assert round(at_1[str(first.signal_id)].confidence, 4) == CONF_1
    assert round(at_2[str(first.signal_id)].confidence, 4) == CONF_2

    # A snapshot at which this conclusion was never observed is ABSENT, not
    # silently backfilled with a neighbouring observation.
    assert await observations_at(
        db, ORG, [first.signal_id], state_watermark="never-folded"
    ) == {}


async def test_replaying_one_reasoning_pass_writes_nothing_the_second_time(db):
    """Idempotency at the database, not in application bookkeeping. Retries,
    at-least-once redelivery and a regenerated brief all land here."""
    await _org(db)
    signal = _commitment_signal(as_of=AS_OF_1, watermark=WM_1, age_days=AGE_1)

    assert await persist_signals(db, [signal]) == 1
    assert await persist_signals(db, [signal]) == 0
    assert await persist_signals(db, [signal]) == 0
    await db.flush()

    assert len(await observation_history(db, ORG, signal.signal_id)) == 1


async def test_a_null_watermark_does_not_accumulate_duplicate_observations(db):
    """The reason identity is a hashed PK and not a UNIQUE index.

    Postgres treats NULLs as distinct, so `UNIQUE (signal_id, state_watermark)`
    would have let an organization with no folded memory grow one duplicate
    observation per pass, forever.
    """
    await _org(db)
    signal = _commitment_signal(as_of=None, watermark=None, age_days=AGE_1)

    assert await persist_signals(db, [signal]) == 1
    assert await persist_signals(db, [signal]) == 0
    await db.flush()

    history = await observation_history(db, ORG, signal.signal_id)
    assert len(history) == 1
    assert history[0].state_watermark is None
    assert history[0].observation_id == compute_observation_id(signal.signal_id, None)


async def test_observations_never_overwrite_history(db):
    """Append-only, enforced. A second observation of the same snapshot is
    rejected outright rather than replacing the recorded one — the values a
    decision was built from must stay exactly as they were."""
    await _org(db)
    original = _commitment_signal(as_of=AS_OF_1, watermark=WM_1, age_days=AGE_1)
    await persist_signals(db, [original])
    await db.flush()

    # The same (conclusion, snapshot) recomputed with different numbers — e.g. a
    # code change that alters the confidence transform.
    tampered = _commitment_signal(as_of=AS_OF_1, watermark=WM_1, age_days=99.0)
    assert await persist_signals(db, [tampered]) == 0
    await db.flush()

    (row,) = await observation_history(db, ORG, original.signal_id)
    assert round(row.confidence, 4) == CONF_1, "recorded history was overwritten"
    assert row.evidence["age_days"] == round(AGE_1, 2)


async def test_obligation_counts_see_a_conclusion_carried_over_from_an_earlier_fold(db):
    """THE `shown > total` DEFECT, end to end.

    The commitment is FIRST detected at WM_1 and still true at WM_2. Counting
    `reasoning_signals.state_watermark` finds it only at WM_1 — which is how a
    brief at WM_2 showed it as its top-ranked decision while reporting it in no
    obligation count at all.
    """
    await _org(db)
    await persist_signals(db, [_commitment_signal(as_of=AS_OF_1, watermark=WM_1, age_days=AGE_1)])
    await persist_signals(db, [_commitment_signal(as_of=AS_OF_2, watermark=WM_2, age_days=AGE_2)])
    await db.flush()

    at_1 = await signal_type_counts_at(
        db, ORG, reasoning_version=REASONING_VERSION, state_watermark=WM_1
    )
    at_2 = await signal_type_counts_at(
        db, ORG, reasoning_version=REASONING_VERSION, state_watermark=WM_2
    )
    assert at_1 == {"commitment_slipping": 1}
    assert at_2 == {"commitment_slipping": 1}, (
        "an obligation carried over from an earlier fold vanished from the count"
    )

    # A snapshot nothing was recorded at reports EMPTY, which the caller surfaces
    # as `available: false` — never as "you owe nothing".
    assert await signal_type_counts_at(
        db, ORG, reasoning_version=REASONING_VERSION, state_watermark="unfolded"
    ) == {}


async def test_counts_are_per_conclusion_not_per_observation(db):
    """One thing owed is one obligation, however many passes observed it."""
    await _org(db)
    for wm, as_of, age in (
        (WM_1, AS_OF_1, AGE_1), (WM_2, AS_OF_2, AGE_2), ("wm-3", AS_OF_2, AGE_2 + 0.15),
    ):
        await persist_signals(db, [_commitment_signal(as_of=as_of, watermark=wm, age_days=age)])
    await db.flush()

    for wm in (WM_1, WM_2, "wm-3"):
        counts = await signal_type_counts_at(
            db, ORG, reasoning_version=REASONING_VERSION, state_watermark=wm
        )
        assert counts == {"commitment_slipping": 1}, f"count inflated at {wm}"


async def test_observation_reads_are_tenant_scoped(db):
    """These ids travel through Execution/Observation/Learning payloads. An id
    alone must never resolve another organization's observation."""
    other_org = uuid.UUID("df7d1c0a-ce08-4f52-85ce-4afc4088847c")
    await _org(db)
    await _org(db, other_org)

    mine = _commitment_signal(as_of=AS_OF_1, watermark=WM_1, age_days=AGE_1)
    await persist_signals(db, [mine])
    await db.flush()

    assert await observations_at(
        db, other_org, [mine.signal_id], state_watermark=WM_1
    ) == {}
    assert await observation_history(db, other_org, mine.signal_id) == []
    assert await signal_type_counts_at(
        db, other_org, reasoning_version=REASONING_VERSION, state_watermark=WM_1
    ) == {}


async def test_a_decision_and_its_cited_observation_describe_the_same_snapshot(db):
    """THE P0-3 INVARIANT, end to end against real Postgres.

    decision.state_watermark == observation.state_watermark, and confidence,
    severity, why and evidence all agree. Live, decision 4373a0b3 published
    11.85 days / 0.8951 while the signal it cites recorded 7.16 / 0.7386, and
    nothing compared them.
    """
    from navis.decisions.service import DecisionService
    from navis.decisions.what_matters_today import WhatMattersTodayEngine

    await _org(db)
    # Two folds: the conclusion is FIRST detected at WM_1 and still true at WM_2.
    # The brief is built at WM_2 — the exact shape that broke.
    await persist_signals(db, [_commitment_signal(as_of=AS_OF_1, watermark=WM_1, age_days=AGE_1)])
    current = _commitment_signal(as_of=AS_OF_2, watermark=WM_2, age_days=AGE_2)
    await persist_signals(db, [current])
    await db.flush()

    decisions = WhatMattersTodayEngine().decide(
        [current], org_id=ORG, reasoning_version=REASONING_VERSION,
        state_watermark=WM_2, as_of=AS_OF_2,
    )
    brief = await DecisionService(db).decide(
        ORG, [current], reasoning_version=REASONING_VERSION,
        state_watermark=WM_2, as_of=AS_OF_2,
    )
    await db.flush()

    checks = await DecisionService(db).provenance(brief)
    assert checks, "the brief produced no decisions to verify"
    for decision_id, check in checks.items():
        assert check["verified"] is True, f"{decision_id}: {check['reason']}"
        assert check["state_watermark"] == WM_2

    # The published values are the CURRENT fold's, not first detection's.
    (published,) = brief.decisions
    assert published["confidence"] == CONF_2
    assert published["severity"] == SEV_2
    assert published["evidence"]["age_days"] == round(AGE_2, 2)
    assert decisions[0].decision_id == uuid.UUID(published["decision_id"])


async def test_a_decision_whose_snapshot_was_never_observed_fails_closed(db):
    """No observation at the decision's watermark => `verified: false` with a
    reason. NEVER silently backed by the neighbouring snapshot's observation,
    which is exactly what would rebuild the defect one layer up."""
    from navis.decisions.service import DecisionService

    await _org(db)
    # Recorded at WM_1 only...
    signal = _commitment_signal(as_of=AS_OF_1, watermark=WM_1, age_days=AGE_1)
    await persist_signals(db, [signal])
    await db.flush()

    # ...but the brief claims WM_2, where nothing was ever observed.
    brief = await DecisionService(db).decide(
        ORG, [_commitment_signal(as_of=AS_OF_2, watermark=WM_2, age_days=AGE_2)],
        reasoning_version=REASONING_VERSION, state_watermark=WM_2, as_of=AS_OF_2,
    )
    await db.flush()
    # Remove the WM_2 observation the brief pass wrote, leaving only WM_1's —
    # the state a failed/partial reasoning persist would leave behind.
    from sqlalchemy import delete

    await db.execute(
        delete(ReasoningSignalObservation).where(
            ReasoningSignalObservation.state_watermark == WM_2
        )
    )
    await db.flush()

    (check,) = (await DecisionService(db).provenance(brief)).values()
    assert check["verified"] is False
    assert "no reasoning observation recorded" in check["reason"]
    assert check["state_watermark"] == WM_2
    # The WM_1 observation still exists and was NOT substituted.
    assert len(await observation_history(db, ORG, signal.signal_id)) == 1


async def test_an_observation_cannot_exist_without_its_conclusion(db):
    """The FK is the guarantee that provenance always resolves: there is no such
    thing as an observation of a conclusion Navis never recorded."""
    from sqlalchemy.exc import IntegrityError

    await _org(db)
    orphan = uuid.uuid4()
    db.add(
        ReasoningSignalObservation(
            observation_id=compute_observation_id(orphan, WM_1),
            signal_id=orphan,
            organization_id=ORG,
            state_watermark=WM_1,
            state_as_of=AS_OF_1,
            confidence=0.5,
            severity=0.5,
            why="orphaned",
            evidence={},
            created_at=AS_OF_1,
        )
    )
    with pytest.raises(IntegrityError):
        await db.flush()
