"""Observation event subscribers — the Execution -> Observation link.

Closes the loop: Execution publishes ACTION_EXECUTED / ACTION_FAILED (at-least-once,
with a stable event_id and full lineage); Observation captures each as an immutable
outcome, guarded by the inbox so duplicate deliveries collapse to one.

(ACTION_RETRIED has no dedicated event in Execution V2 — a retry re-emits
ACTION_EXECUTING then EXECUTED/FAILED, which we already capture. No redesign.)
"""
from __future__ import annotations

import uuid

from navis.observation.inbox import process_once
from navis.observation.models import OutcomeType
from navis.observation.service import ObservationService
from navis.platform.database import session_scope
from navis.platform.events import DomainEvent, EventType, subscribe
from navis.platform.monitoring import get_logger

logger = get_logger("observation.subscribers")

_CONSUMER = "observation"


async def _capture(event: DomainEvent, outcome_type: OutcomeType) -> None:
    if event.org_id is None:
        return
    source_event_id = (event.payload or {}).get("event_id")
    if not source_event_id:
        # Without a stable event id we cannot dedupe; skip rather than risk doubles.
        logger.warning("execution event without event_id; skipping outcome capture")
        return

    async with session_scope() as db:
        from navis.observation.repository import OutcomeRepository

        repo = OutcomeRepository(db)
        svc = ObservationService(db, repo=repo)

        async def handler() -> None:
            await svc.record_from_execution(event, outcome_type, source_event_id)

        await process_once(repo, _CONSUMER, str(source_event_id), handler)


@subscribe(EventType.ACTION_EXECUTED)
async def on_action_executed(event: DomainEvent) -> None:
    await _capture(event, OutcomeType.ACTION_EXECUTED)


@subscribe(EventType.ACTION_FAILED)
async def on_action_failed(event: DomainEvent) -> None:
    await _capture(event, OutcomeType.ACTION_FAILED)


# ── Human judgement ───────────────────────────────────────────────────
#
# Execution has always emitted these; nothing consumed them. A founder pressing
# Reject on a proposal Navis made is the most informative event in the system —
# it is a direct verdict on the proposal, arriving before any side effect — and
# it was written to the outbox, relayed, and dropped because no handler existed.
#
# They travel the SAME path as every machine outcome (inbox-deduped, lineage
# carried from the persisted row), so nothing new is introduced: this is the
# existing contract finally applied to the events that already flowed through it.


@subscribe(EventType.ACTION_REJECTED)
async def on_action_rejected(event: DomainEvent) -> None:
    await _capture(event, OutcomeType.ACTION_REJECTED)


@subscribe(EventType.ACTION_CANCELLED)
async def on_action_cancelled(event: DomainEvent) -> None:
    await _capture(event, OutcomeType.ACTION_CANCELLED)


# ── Human judgement on a DECISION ─────────────────────────────────────
#
# The same gap the two handlers above closed, one layer up. `/brief/decisions/
# {ref}/dismiss` has always persisted the dismissal and published
# DECISION_DISMISSED — its own docstring says "for Learning to consume as an
# explicit negative signal" — and no handler existed, so the claim was false and
# every dismissal was published and dropped.
#
# THIS IS THE ONLY VERDICT MOST DECISIONS CAN RECEIVE. `risk`, `opportunity` and
# `delegation` decisions map to no action type, so they never become an
# ExecutionRequest and can never be rejected or cancelled. Live, that is the
# category of every decision on the current brief — so before this handler, the
# entire brief was unmeasurable and the signals behind it could never be shown
# to be unreliable.
#
# SAME PATH, NO SHORTCUT: inbox-deduped on the event id, lineage read off the
# event, recorded by ObservationService, and learned by the existing
# OUTCOME_OBSERVED subscriber. Nothing here writes a learning event, and the
# router does not either — it publishes, and the chain does the rest.
#
# IDEMPOTENT BY CONSTRUCTION. `dismiss_decision` derives its dismissal id as
# uuid5(org | decision_ref | user) and publishes it as `event_id`, so a
# re-clicked dismiss produces the same event id, the inbox claim fails, and no
# second outcome is recorded.


@subscribe(EventType.DECISION_DISMISSED)
async def on_decision_dismissed(event: DomainEvent) -> None:
    await _capture(event, OutcomeType.DECISION_DISMISSED)


# BRIEF_OPENED IS DELIBERATELY NOT SUBSCRIBED.
#
# It is a page view, not a verdict. Every outcome recorded here becomes a
# learning event and increments the SIGNAL and DECISION aggregates for every
# decision on the brief, so subscribing it would add one NEUTRAL sample per
# signal per open — inflating `total` without ever moving `decisive`, and
# swamping the genuine judgement evidence with a count of how often a tab was
# focused. Opening a brief says nothing about whether its decisions were right,
# and Learning stores only facts that do.
