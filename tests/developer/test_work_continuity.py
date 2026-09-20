"""The three guarantees a work-continuity client depends on from Core.

Everything here is the DURABILITY half of Developer Work State: a coding agent
loses its conversation to a compaction and asks Core what it was doing. Three
things have to hold for that answer to be trustworthy, and each of them failed
on the live ledger before it was pinned here.

  EXACT RETRIEVAL   The newest checkpoint of a task is addressable. Without it
                    a client scans runs, picks a depth, and anything deeper
                    reads as absent — silently, and identically to work that
                    never existed.

  MONOTONIC WRITES  A late or retried write cannot replace a newer statement
                    about the same run.

  HONEST SCRUBBING  A credential is destroyed; an ordinary key is not. Both
                    halves are the guarantee, and the second half is the one
                    that had been quietly failing.

No database. The ordering contract is expressed as a pure comparison over plain
rows, which is the same discipline `test_run_telemetry.py` uses on the rollup —
the parts most worth being sure about should not need four layers of transport
to assert. Ordering AS EXECUTED BY POSTGRES is verified against a live Core in
the developer CLI's dogfooding pass, because that is the only place it is real.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from navis.developer.run_schemas import RecordOutcomeRequest
from navis.developer.lifecycle import ALLOWED_RUN_TRANSITIONS, may_transition
from navis.developer.run_service import _TERMINAL_RUN_STATUSES, loses_to_stored_outcome
from navis.developer.sanitize import REDACTED, sanitize_metadata

_T0 = datetime(2026, 9, 18, 12, 0, 0, tzinfo=timezone.utc)


# ── Monotonic writes: compare-and-swap on the reality clock ─────────────────


def test_a_write_without_the_flag_is_never_refused():
    """DEFAULT OFF is the compatibility statement. An evaluator scoring a run a
    week later, or a backfill writing history, is not a stale writer — refusing
    it would break every caller that predates the flag."""
    assert not loses_to_stored_outcome(stored=_T0, incoming=_T0 - timedelta(hours=1), if_newer=False)


def test_an_older_observation_loses_to_a_newer_stored_one():
    """The failure this exists for: two writers on one run, ordered by arrival
    rather than by authorship. The older statement must not land last and take
    the append-only sections with it."""
    assert loses_to_stored_outcome(stored=_T0, incoming=_T0 - timedelta(seconds=1), if_newer=True)


def test_a_newer_observation_wins():
    """A session revising its OWN checkpoint forward is the normal case and the
    whole point of an upsert. The guard must not make progress impossible."""
    assert not loses_to_stored_outcome(stored=_T0, incoming=_T0 + timedelta(seconds=1), if_newer=True)


def test_an_identical_instant_is_admitted():
    """REGRESSION. Refusing an equal instant looked obviously right — arrival
    should not decide a tie — and it broke the normal case, because on one run
    there is only ever one writer.

    `agent_run_outcomes` is UNIQUE per run and a session owns exactly one run,
    so every write to a row is the same author revising its own statement; for
    one author, arrival order IS authorship order. With equality refused, an
    agent recording two decisions inside the same millisecond had the second
    silently dropped."""
    assert not loses_to_stored_outcome(stored=_T0, incoming=_T0, if_newer=True)


def test_a_repeated_identical_write_is_admitted_rather_than_refused():
    """The retry case, stated as its own guarantee. A duplicate delivery of the
    SAME request carries the same instant and the same content, so admitting it
    is idempotent — and refusing it would make an ordinary network retry look
    like a concurrency conflict to every caller that reads the status."""
    assert not loses_to_stored_outcome(stored=_T0, incoming=_T0, if_newer=True)


@pytest.mark.parametrize(
    "stored, incoming",
    [(None, _T0), (_T0, None), (None, None)],
    ids=["no stored clock", "no incoming clock", "neither"],
)
def test_a_write_is_never_refused_when_either_side_has_no_clock(stored, incoming):
    """A write with no `produced_at` makes no claim about when it was observed,
    so there is nothing to compare and nothing to refuse. Guessing would mean
    inventing an ordering between two statements that never asserted one."""
    assert not loses_to_stored_outcome(stored=stored, incoming=incoming, if_newer=True)


def test_a_naive_timestamp_is_compared_rather_than_crashing():
    """Python refuses to order an aware datetime against a naive one. A
    `TypeError` raised from inside a concurrency guard would turn a losing write
    into a 500 — an outage caused by the code that exists to prevent data loss."""
    naive_older = (_T0 - timedelta(hours=1)).replace(tzinfo=None)
    assert loses_to_stored_outcome(stored=_T0, incoming=naive_older, if_newer=True)

    naive_newer = (_T0 + timedelta(hours=1)).replace(tzinfo=None)
    assert not loses_to_stored_outcome(stored=_T0, incoming=naive_newer, if_newer=True)


def test_the_conditional_flag_is_off_unless_asked_for():
    """Pinned on the SCHEMA, not just the service: a default that flipped to
    true would start refusing writes from every existing client at once, and
    the symptom — checkpoints silently not landing — is the one this whole
    domain is least able to notice."""
    assert RecordOutcomeRequest(status="unknown").if_newer is False
    assert RecordOutcomeRequest(status="unknown", if_newer=True).if_newer is True


# ── The ordering contract of the exact lookup ───────────────────────────────
#
# `latest_task_outcome` orders `produced_at DESC NULLS LAST, updated_at DESC,
# run_id DESC`. That is not an implementation detail: it has to agree rung for
# rung with the client's own selection, because the two answer the same
# question and disagreeing would surface only for the tenants large enough to
# need the server-side path. Expressed here as the comparison it encodes.


def _key(produced_at: datetime | None, updated_at: datetime, run_id: str):
    """The sort key `ORDER BY produced_at DESC NULLS LAST, updated_at DESC,
    run_id DESC` induces, as a value Python can order ascending."""
    return (produced_at is None, -(produced_at or _T0).timestamp() if produced_at else 0,
            -updated_at.timestamp(), _invert(run_id))


def _invert(run_id: str) -> tuple:
    """Descending on a string, expressed as an ascending key."""
    return tuple(-ord(c) for c in run_id)


def test_reality_time_outranks_row_time():
    """A write that LANDS late having OBSERVED early must never present itself
    as the newest state. Ordering on row time would mean a long-running agent,
    a retried write or a laptop waking up rewrites the present."""
    observed_early_landed_late = _key(_T0, _T0 + timedelta(hours=5), "a")
    observed_late_landed_early = _key(_T0 + timedelta(hours=1), _T0, "b")
    assert observed_late_landed_early < observed_early_landed_late


def test_an_unstamped_outcome_never_outranks_a_stamped_one():
    """A row that cannot say when its content was observed is not discarded —
    it may be all there is — but it cannot outrank a row that can, whatever the
    row clock says. That is what NULLS LAST buys."""
    unstamped_written_now = _key(None, _T0 + timedelta(hours=9), "z")
    stamped_long_ago = _key(_T0 - timedelta(days=30), _T0 - timedelta(days=30), "a")
    assert stamped_long_ago < unstamped_written_now


def test_the_order_is_total_so_two_reads_agree():
    """Two outcomes stamped in the same microsecond must not swap between two
    calls. Without the id tiebreak the winner is whatever the planner visited
    first, and a coin flip decides which session a developer continues from."""
    same_instant_a = _key(_T0, _T0, "aaa")
    same_instant_b = _key(_T0, _T0, "bbb")
    assert same_instant_a != same_instant_b
    assert (same_instant_b < same_instant_a) is True


# ── Honest scrubbing: both halves of the guarantee ──────────────────────────


@pytest.mark.parametrize(
    "key",
    ["pin", "PIN", "pin_code", "pinCode", "userPin", "otp", "OTP", "otp_code"],
    ids=lambda k: f"credential:{k}",
)
def test_a_short_credential_key_is_still_destroyed(key: str):
    """Narrowing `pin` and `otp` to exact matches must not narrow the coverage
    they were carrying. A compound spelling normalizes onto the same stem."""
    assert sanitize_metadata({key: "9134"}, field="metadata")[key] == REDACTED


@pytest.mark.parametrize(
    "key",
    ["mapping", "shipping", "spinner", "pinned", "opinion", "pipeline", "endpoint"],
    ids=lambda k: f"ordinary:{k}",
)
def test_an_ordinary_key_containing_pin_survives(key: str):
    """The other half of the guarantee, and the half that was failing. Stripping
    separators destroys word boundaries, so `pin` as a SUBSTRING stem was
    redacting the value of every key with those three letters in it — `mapping`
    is an ordinary thing for an agent to report, and its value was being
    deleted with nothing to show that anything had happened."""
    assert sanitize_metadata({key: "src/handlers"}, field="metadata")[key] == "src/handlers"


def test_the_navis_work_state_keys_survive_scrubbing():
    """The keys this feature actually writes. A deny-list that eats its own
    caller's vocabulary corrupts the records it exists to protect — and a work
    key destroyed on arrival means the continuity lookup silently stops
    matching, which reads as "no work here" rather than as an error."""
    written = {
        "navisWorkKey": "nws1:b13ca19a50c6cda56372076618bbed61",
        "navisSchema": "navis-workstate-1",
        "navisObservedThrough": "2026-09-18T05:18:45.549000Z",
    }
    assert sanitize_metadata(dict(written), field="metadata") == written


def test_a_correlation_id_reported_as_session_id_is_still_redacted():
    """The deliberate FALSE POSITIVE, pinned so it stays deliberate.

    `sessionId` is a harness correlation id here and not a credential — but in
    a great many systems a session identifier IS the bearer credential, and
    this list is matched against keys written by agents reporting on arbitrary
    third-party tools. The value has a first-class unredacted home in
    `agent_runs.correlation_id`; a caller that needs it readable writes it
    there rather than asking the deny-list to get narrower."""
    out = sanitize_metadata({"sessionId": str(uuid.uuid4())}, field="metadata")
    assert out["sessionId"] == REDACTED


# ── Run lifecycle: finishing twice the same way is not a conflict ───────────
#
# WHY THIS MATTERS FOR CONTINUITY, AND WHERE IT STOPS MATTERING. A run's status
# is NOT what continuity depends on — the checkpoint lives on the outcome, and
# `record_outcome` never inspects the run's status, so a closed run can still
# receive one. Closing a run therefore cannot destroy work state, which is what
# makes closing it safe to do at all.
#
# What it does affect is whether the ledger can be reasoned about: a run that
# never closes has a NULL `completed_at`, so every duration and every
# reliability rate computed over `agent_runs` is wrong. On the live tenant 307
# of 321 runs were still `running`.


def test_finishing_a_run_twice_the_same_way_is_idempotent():
    """The caller that hits this cannot avoid it.

    A session closes its run as the process is torn down — exactly when a
    response is most likely to be lost — and the documented remedy for a lost
    response is to retry. Answering the retry with 409 tells a client its
    shutdown failed when the ledger already says what it was trying to say."""
    assert may_transition("running", "completed")
    # The transition table gives a terminal run no outgoing edges...
    assert not may_transition("completed", "completed")
    # ...so the idempotent repeat cannot be expressed there, and is handled in
    # `finish_run` ahead of the transition check. This pins the vocabulary that
    # decision reads, so a new terminal status cannot silently miss it.
    assert _TERMINAL_RUN_STATUSES == {"completed", "failed", "cancelled"}


def test_a_different_terminal_status_is_still_a_conflict():
    """`completed` becoming `failed` is a REWRITE of history, not a retry. An
    execution ledger that can be edited is not a ledger, so this stays refused
    — the idempotence above is deliberately narrow."""
    for other in ("failed", "cancelled"):
        assert not may_transition("completed", other)


def test_terminal_statuses_are_derived_from_the_transition_table():
    """Not written out a second time. A status added to the vocabulary with no
    outgoing edges becomes terminal everywhere at once, rather than in the one
    place somebody remembered."""
    for status in _TERMINAL_RUN_STATUSES:
        assert ALLOWED_RUN_TRANSITIONS[status] == frozenset()
