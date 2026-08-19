"""Execution worker — runs providers off the request path, relays the outbox.

Horizontally scalable (FOR UPDATE SKIP LOCKED, no global locks). Hardened:
  * claims APPROVED *and* retryable FAILED (attempts < max) requests;
  * gates each provider behind a per-provider rate limiter (skip = backpressure);
  * records telemetry; dead-letters on exhausted attempts / validation errors;
  * relays the transactional outbox (at-least-once; consumers dedupe via the inbox).

The HTTP API never runs a provider — it changes state + enqueues this worker.
"""
from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from time import monotonic
from typing import Any

from navis.execution.actions import get_spec
from navis.execution.config import MAX_EXECUTION_ATTEMPTS
from navis.execution.contract import FailureKind, ProviderReference, ProviderResult
from navis.execution.dead_letter import ExecutionDeadLetterService
from navis.execution.integrations import IntegrationUnavailable
from navis.execution.providers import get_provider
from navis.execution.providers.base import ProviderError, ProviderValidationError
from navis.execution.retry import execute_with_retry
from navis.execution.service import ExecutionService
from navis.execution.status import ExecutionStatus
from navis.execution.telemetry import provider_telemetry, request_durations
from navis.organization.contract import Capability
from navis.platform.composition import ensure_subscribers
from navis.platform.events import DomainEvent, EventType, event_bus
from navis.platform.monitoring import get_logger

logger = get_logger("execution.worker")


@dataclass(frozen=True, slots=True)
class _Lease:
    """A DETACHED snapshot of a claimed request.

    Phase A commits and closes its session, which expires the ORM row — so
    everything the provider call needs is copied out first. Touching the ORM
    object after that commit would either raise or silently re-open a session,
    which is the coupling this split exists to remove.
    """

    execution_id: uuid.UUID
    org_id: uuid.UUID
    decision_id: uuid.UUID
    action_type: str
    payload: dict[str, Any]


@asynccontextmanager
async def _borrowed(repo):
    """Run a phase inside the CALLER's transaction.

    Used when `execute_pass` is handed a repo directly (the unit tests, and the
    Postgres concurrency suite). The caller owns the commit, so the three phases
    collapse back into one transaction — the pre-existing behaviour — while
    running the exact same code path as production. That is deliberate: a
    production-only path would be the one path no test covers.
    """
    yield repo


def _unit_for(repo, unit):
    if unit is not None:
        return unit
    if repo is None:
        raise TypeError("execute_pass requires either `repo` or `unit`")
    return lambda: _borrowed(repo)


async def execute_pass(
    repo=None,
    *,
    unit=None,
    providers=None,
    policy=None,
    limit: int = 100,
    rate_limiter=None,
    max_attempts: int = MAX_EXECUTION_ATTEMPTS,
    telemetry=None,
    reconciler=None,
) -> dict:
    """Drain up to `limit` executable requests, ONE SHORT TRANSACTION PER PHASE.

    `unit` is a factory yielding a repo for one transaction (production passes
    `session_scope`). `repo` is the legacy/in-transaction form kept for callers
    that already own a session — see `_borrowed`.

    `reconciler` is injected so tests can assert the read-model write-back
    happened WITHOUT reaching Google. `None` means "resolve the production one
    lazily" — importing it at module scope would drag the Signals stack into
    every test that touches the worker.
    """
    unit = _unit_for(repo, unit)
    tel = telemetry or provider_telemetry
    executed = failed = skipped = 0
    # Every row this pass has already touched, whatever the outcome.
    #
    # ONE PASS GIVES A ROW AT MOST ONE ATTEMPT. A FAILED row with attempts left
    # is immediately re-claimable, so without this the single-row claim loop
    # would re-claim the row it just failed and burn the whole retry budget in
    # one pass, back to back, with no backoff between attempts — five provider
    # calls in a few milliseconds instead of five attempts spread across five
    # scheduled passes. Measured on the failure-injection harness before this
    # guard: a single retryable failure reached attempts=5 in one pass.
    #
    # Also covers the rate-limited case: a throttled row stays APPROVED and must
    # not be re-offered by the very next iteration.
    seen: set = set()

    for _ in range(limit):
        outcome = await _execute_one(
            unit,
            providers=providers,
            policy=policy,
            rate_limiter=rate_limiter,
            max_attempts=max_attempts,
            telemetry=tel,
            reconciler=reconciler,
            exclude=seen,
        )
        if outcome is None:
            break
        kind, eid = outcome
        seen.add(eid)
        if kind == "executed":
            executed += 1
        elif kind == "failed":
            failed += 1
        else:
            skipped += 1

    if executed or failed or skipped:
        logger.info("execute_pass executed=%d failed=%d skipped=%d", executed, failed, skipped)
    return {"executed": executed, "failed": failed, "skipped": skipped}


async def _execute_one(
    unit, *, providers, policy, rate_limiter, max_attempts, telemetry, reconciler, exclude
):
    """One request through three phases. Returns (kind, execution_id) or None.

    THE BOUNDARY THIS ENFORCES
    --------------------------
    Phase B calls an external system and MUST NOT hold a database transaction
    while it does. Previously the whole pass ran in one transaction: up to 100
    rows were held under FOR UPDATE for the duration of every Google call in the
    pass, and a failure anywhere rolled back the recorded state of side effects
    that had ALREADY happened — the database forgetting an email that was sent.

    A transaction cannot roll back an external side effect, so the split is the
    only correct shape: commit the lease, act, then commit the result. The
    unavoidable window between "provider succeeded" and "we recorded it" is
    bridged by PROVIDER idempotency (Gmail's Message-ID pre-check, Calendar's
    client-generated event id), never by the transaction.
    """
    # ── PHASE A — CLAIM (short transaction) ───────────────────────────────
    async with unit() as repo:
        rows = await repo.claim_executable(
            max_attempts=max_attempts, limit=1, exclude=exclude
        )
        if not rows:
            return None
        req = rows[0]
        svc = ExecutionService(repo=repo)
        dlq = ExecutionDeadLetterService(repo)
        provider = get_provider(req.action_type, providers)

        if provider is None:
            # APPROVED -> FAILED is NOT a legal edge (status.py), so failing
            # straight from the claimed state raised InvalidTransition out of
            # the pass — taking every OTHER claimed row down with it. Step
            # through EXECUTING, which is the same path every other terminal
            # outcome takes, so one unroutable row can only fail itself.
            await svc.begin_execution(req)
            await svc.fail_execution(
                req,
                f"no provider for {req.action_type}",
                kind=FailureKind.UNROUTABLE,
            )
            await _exhaust_attempts(repo, req, max_attempts)
            await dlq.record(req, provider=None, error="no provider")
            return "failed", req.id

        # Already-executed guard. A claimed row can carry a provider reference
        # when a PREVIOUS attempt reached the provider successfully but did not
        # get to record EXECUTED — the stale-EXECUTING recovery path pushes such
        # rows to FAILED, which makes them re-claimable here. Re-running the
        # provider would create a SECOND calendar event / send a SECOND email.
        # The reference is proof the side effect already happened, so close the
        # row out instead of repeating it.
        #
        # This is ALSO what closes CASE B of the crash matrix: a worker that
        # died between a successful provider call and Phase C.
        existing = _provider_reference_id(req)
        if existing:
            logger.warning(
                "execution %s already has provider reference %s; completing "
                "without re-calling %s", req.id, existing, provider.name,
            )
            await svc.begin_execution(req)
            await svc.complete_execution(
                req,
                ProviderResult(
                    ok=True,
                    reference=ProviderReference(
                        provider.name, existing, {"replayed": True}
                    ),
                ),
            )
            return "executed", req.id

        # Rate-limit gate: leave APPROVED/FAILED for the next pass (no storm).
        # Checked BEFORE the EXECUTING transition — transitioning first would
        # strand a throttled row in EXECUTING until stale recovery swept it.
        if rate_limiter is not None and not await rate_limiter.allow(provider.name):
            return "skipped", req.id

        # -> EXECUTING, attempts += 1. Committing THIS is what leases the row:
        # `claim_executable` only ever selects APPROVED or FAILED, so once the
        # transition is committed no other worker can claim it, with no lock
        # held. Stale-EXECUTING recovery is the lease expiry.
        await svc.begin_execution(req)

        # ── AUTHORIZATION RE-CHECK — the last gate before the outside world ──
        #
        # APPROVAL IS NOT AUTHORIZATION. Approval is a human statement of intent,
        # recorded once and never revisited. Authorization is a property of the
        # organization RIGHT NOW, and it can change after a human clicks Approve:
        # the org is suspended for non-payment, a plan downgrade revokes
        # EXECUTION, an admin revokes the connector capability, or `kill:gmail`
        # is thrown during an incident.
        #
        # Until this check existed, none of that stopped queued work. An approved
        # row sat in the queue, the worker claimed it on the next pass, and the
        # email went out for an org that was no longer allowed to send it — which
        # made the kill switch, the one lever for stopping side effects during an
        # incident, ineffective against exactly the actions already in flight.
        #
        # Placed AFTER `begin_execution` on purpose. Denying before the EXECUTING
        # transition would leave the row APPROVED and immediately re-claimable
        # with no attempt recorded — a hot loop that never terminates and never
        # dead-letters. Failing from EXECUTING spends one attempt, which bounds
        # the retries and lands the row in the DLQ with a stated reason once the
        # budget is gone.
        #
        # Deliberately RETRYABLE (attempts are not exhausted): revocation is
        # frequently temporary — an incident kill switch is lifted, a suspension
        # is resolved, a plan is re-provisioned — and leaving the retry budget
        # intact means restoring authorization resumes the work on the next pass
        # with no operator action. A permanent revocation still terminates, at
        # the DLQ, after the normal number of attempts.
        #
        # The approval record is untouched: `approved_by` / `approved_at` still
        # say who authorized the intent and when. This adds a second, current
        # question; it does not rewrite the answer to the first one.
        spec = get_spec(req.action_type)
        denial = await _authorization_denial(repo, req, spec)
        if denial is not None:
            # Structured, and deliberately identifier-only. This is the record an
            # operator reads during an incident ("why did 10k sends stop?") and
            # an auditor reads afterwards ("who was refused, and on what"), so it
            # carries the full lineage — org, execution, decision, action,
            # provider, the capability required and the ones actually denied,
            # plus the lifecycle status that distinguishes a suspension from a
            # kill switch. It carries no payload and no credential: every field
            # is an id, an enum value or a capability name.
            logger.warning(
                "execution authorization DENIED org=%s execution=%s decision=%s "
                "action=%s provider=%s capability=%s denied=%s lifecycle=%s "
                "reason=%s",
                req.organization_id, req.id, req.decision_id, req.action_type,
                spec.provider if spec else None,
                spec.capability.value if spec else None,
                ",".join(denial.denied) or "-", denial.lifecycle or "-",
                denial.reason,
            )
            await svc.fail_execution(
                req, denial.reason, kind=FailureKind.AUTHORIZATION_DENIED
            )
            await _maybe_dlq(dlq, req, provider, denial.reason, max_attempts)
            return "failed", req.id

        lease = _Lease(
            execution_id=req.id,
            org_id=req.organization_id,
            decision_id=req.decision_id,
            action_type=req.action_type,
            payload=dict(req.payload or {}),
        )
    # ── transaction committed. NO locks held from here. ───────────────────

    # ── PHASE B — EXTERNAL SIDE EFFECT (no transaction) ───────────────────
    # Thread execution context (org_id, execution_id) into the payload under a
    # reserved key so providers whose real senders need to resolve org-scoped
    # resources (e.g. CalendarProvider resolving the org's Google Calendar
    # Integration) can do it without extending the Provider.execute()
    # signature. Keys are stripped by providers that don't care.
    augmented = {
        **lease.payload,
        "__ctx__": {
            "org_id": str(lease.org_id),
            "execution_id": str(lease.execution_id),
            "decision_id": str(lease.decision_id),
        },
    }
    result = None
    permanent_error = None
    transient_error = None
    # The classification the worker ALREADY makes below, kept so it can be
    # published instead of discarded. See execution/contract.py::FailureKind.
    failure_kind: FailureKind | None = None
    t0 = monotonic()
    try:
        result = await execute_with_retry(provider, augmented, policy)
        telemetry.record(
            provider.name, ok=result.ok, retried=False, latency_s=monotonic() - t0
        )
    except ProviderValidationError as exc:
        telemetry.record(provider.name, ok=False, retried=False, latency_s=monotonic() - t0)
        permanent_error = f"validation: {exc}"
        failure_kind = FailureKind.VALIDATION_FAILED
    except IntegrationUnavailable as exc:
        # PERMANENT, like a validation error — not transient. The organization
        # has no usable credential for this connector: not connected, revoked,
        # or connected without the scope this action needs. None of that resolves
        # itself in the seconds between retries; it needs a person to reconnect
        # the account. Retrying spent the whole budget rediscovering the same
        # thing and delayed the dead letter that actually tells them what to do.
        #
        # Contrast with an AUTHORIZATION denial (the re-check above), which stays
        # retryable on purpose: a kill switch or a suspension flips back at
        # machine speed, so leaving the budget intact lets the work resume by
        # itself. Missing consent does not flip back by itself.
        #
        # The message is the connector's own and is safe to persist: it names the
        # provider and the remedy, never a token or header. It travels to the
        # dead letter, the outbox and Observation.
        telemetry.record(provider.name, ok=False, retried=False, latency_s=monotonic() - t0)
        permanent_error = f"integration: {exc}"
        failure_kind = FailureKind.CREDENTIAL_UNAVAILABLE
    except ProviderError as exc:
        # A CONNECTOR SAYING "THIS CAN NEVER SUCCEED" MUST BE BELIEVED.
        #
        # `ProviderError.retryable` is the one provider-neutral vocabulary a
        # connector has for that, and until this clause existed nothing read it
        # on this path. `execute_with_retry` consults it (tenacity retries only
        # the retryable ones), but a NON-retryable ProviderError propagates out
        # of it and landed in `except Exception` below — which classifies
        # everything as transient. So the row went FAILED with attempts left,
        # was re-claimed on the next pass, and the provider was called AGAIN.
        #
        # Reproduced before this clause existed: a provider raising
        # `ProviderError("the target record was deleted", retryable=False)` was
        # invoked FIVE times — five external calls, five passes, one dead letter
        # at the end — for a failure it had declared terminal on the first.
        #
        # `ProviderValidationError` is a subclass and is caught above; it keeps
        # its own `validation:` prefix because it means a malformed payload
        # rather than a rejected-but-well-formed one, and the two lead an
        # operator to different places.
        #
        # A RETRYABLE ProviderError reaching here has already exhausted
        # tenacity's budget inside `execute_with_retry`, so it stays transient
        # and the pass-level retry behaviour is unchanged.
        telemetry.record(
            provider.name, ok=False, retried=exc.retryable, latency_s=monotonic() - t0
        )
        if exc.retryable:
            transient_error = str(exc)
            failure_kind = FailureKind.TRANSIENT
        else:
            permanent_error = f"provider: {exc}"
            failure_kind = FailureKind.PROVIDER_REJECTED
    except Exception as exc:  # retries exhausted / unexpected
        telemetry.record(provider.name, ok=False, retried=True, latency_s=monotonic() - t0)
        transient_error = str(exc)
        failure_kind = FailureKind.TRANSIENT

    if result is not None and result.ok:
        # READ-AFTER-WRITE. Pull the mutated calendar event's CURRENT state into
        # signal_events BEFORE the row reaches EXECUTED, so the transition
        # carries a real guarantee:
        #
        #     status == EXECUTED => Navis's read model reflects the change.
        #
        # Non-fatal by design (see reconcile's docstring): a bookkeeping failure
        # must never flip a genuinely-successful cancellation to FAILED.
        await _reconcile_read_model(lease, result, reconciler)

    # ── PHASE C — PERSIST RESULT (short transaction) ──────────────────────
    async with unit() as repo:
        svc = ExecutionService(repo=repo)
        dlq = ExecutionDeadLetterService(repo)
        req = await repo.get(lease.execution_id)
        if req is None:  # pragma: no cover - row deleted mid-flight
            logger.error("execution %s vanished mid-flight", lease.execution_id)
            return "failed", lease.execution_id

        if ExecutionStatus(req.status) is not ExecutionStatus.EXECUTING:
            # The lease expired under us — stale recovery re-drove the row while
            # the provider call was in flight. Forcing a transition now would
            # violate the status machine, so record the RECEIPT instead: the
            # already-executed guard above turns it into EXECUTED on the next
            # pass without re-calling the provider.
            logger.error(
                "execution %s lost its lease (status=%s) while the provider ran; "
                "preserving the receipt for the next pass", req.id, req.status,
            )
            if result is not None and result.ok and result.reference is not None:
                req.provider_reference = result.reference.to_dict()
                await repo.flush()
            return "failed", req.id

        if permanent_error is not None:
            await svc.fail_execution(req, permanent_error, kind=failure_kind)
            await _exhaust_attempts(repo, req, max_attempts)
            await dlq.record(req, provider=provider.name, error=permanent_error)
            return "failed", req.id

        if transient_error is not None:
            await svc.fail_execution(
                req, transient_error, kind=_transient_kind(req, max_attempts)
            )
            await _maybe_dlq(dlq, req, provider, transient_error, max_attempts)
            return "failed", req.id

        if result is None or not result.ok:
            error = (result.error if result is not None else None) or "provider failed"
            await svc.fail_execution(
                req, error, kind=_transient_kind(req, max_attempts)
            )
            await _maybe_dlq(dlq, req, provider, error, max_attempts)
            return "failed", req.id

        await svc.complete_execution(req, result)
        logger.info("executed %s %s", req.id, request_durations(req))
        return "executed", req.id


@dataclass(frozen=True, slots=True)
class _Denial:
    """Why a runtime authorization check refused, in a shape that can be logged.

    `reason` is the human-readable string that also becomes `last_error` and the
    dead-letter message; `denied` and `lifecycle` are the structured facts behind
    it. Nothing here is derived from the payload, so none of it can carry
    credential material or user content.
    """

    reason: str
    denied: tuple[str, ...] = ()
    lifecycle: str | None = None


async def _authorization_denial(repo, req, spec) -> _Denial | None:
    """Why this org may NOT perform this action right now, or None if it may.

    Asks for BOTH capabilities in one round trip:

      * `Capability.EXECUTION` — may this org run actions at all;
      * `spec.capability`      — may it use THIS connector (gmail / calendar /
        slack). Checking only EXECUTION is what let a revoked SLACK entitlement,
        or `kill:gmail`, still produce a send.

    Lifecycle (suspended / churned) and the kill switch are covered for free:
    both are resolved inside `EntitlementService._decide`, so this asks one
    question and inherits all four gates.

    FAIL CLOSED. A missing spec, a repository that cannot answer, or an error
    while resolving all deny — an authorization check that cannot complete is
    not a pass.
    """
    if spec is None:
        # Unreachable via the worker (get_provider already refuses an unspecced
        # action), but stated rather than assumed: no spec means no declared
        # capability, and an action with no declared capability must not run.
        return _Denial("no ActionSpec for this action type; refusing to execute")

    authorize = getattr(repo, "authorize", None)
    if authorize is None:
        # A repository that cannot answer the authorization question must not be
        # taken as consent. Raising here would take the whole pass down; failing
        # this row alone is the same shape as every other refusal.
        return _Denial(
            "execution repository cannot resolve authorization; refusing to execute"
        )

    try:
        # ORGANIZATION SCOPE COMES FROM THE PERSISTED ROW. Never from the
        # payload, never from `__ctx__`, never from anything a Decision or a chat
        # turn can influence — those are user-reachable, and letting one name the
        # tenant would make authorization self-selected.
        allowed, denied, lifecycle = await authorize(
            req.organization_id, [Capability.EXECUTION, spec.capability]
        )
    except Exception as exc:  # noqa: BLE001 — see docstring: cannot check => deny
        logger.exception("authorization re-check failed for execution %s", req.id)
        return _Denial(f"authorization could not be verified ({exc}); refusing to execute")

    if allowed:
        return None
    return _Denial(
        reason=(
            "organization is no longer authorized for "
            f"{', '.join(denied)} at execution time (approval remains on record); "
            "refusing to perform the side effect"
        ),
        denied=tuple(denied),
        lifecycle=lifecycle,
    )


def _transient_kind(req, max_attempts: int) -> FailureKind:
    """TRANSIENT while attempts remain; RETRY_EXHAUSTED once they are spent.

    The distinction is the whole point of publishing a failure kind at all: "Gmail
    timed out and we will try again in thirty seconds" and "Gmail timed out five
    times and this action is now in the dead-letter queue" are the same error
    string and completely different facts about the organization. Read AFTER
    `fail_execution` has incremented nothing — `attempts` was already incremented
    by `begin_execution` for this attempt — so the comparison reflects the budget
    as it stands now.
    """
    return (
        FailureKind.RETRY_EXHAUSTED
        if (req.attempts or 0) >= max_attempts
        else FailureKind.TRANSIENT
    )


async def _exhaust_attempts(repo, req, max_attempts: int) -> None:
    """Spend the remaining retry budget on a PERMANENT failure.

    "No provider" and a validation error cannot succeed on a later attempt, but
    the row stayed FAILED with attempts remaining — so every pass re-claimed it,
    re-failed it and wrote ANOTHER dead letter. One unroutable action produced
    five identical dead letters and burned five passes rediscovering that it is
    unroutable.

    Exhausting the budget expresses "permanent" in the mechanism that already
    means it, rather than adding a status: the row is no longer claimable,
    `router.py::retry` already 409s with an explanation for an exhausted row,
    and `dead_letter.replay()` still resets attempts to 0 so an operator replay
    works unchanged. Retryable failures are untouched.
    """
    if (req.attempts or 0) < max_attempts:
        req.attempts = max_attempts
        await repo.flush()


async def _reconcile_read_model(lease: _Lease, result, reconciler) -> None:
    """Write the just-mutated external object back into Navis's read model.

    NO CONNECTOR KNOWLEDGE. This used to import `CALENDAR_MUTATION_ACTION_TYPES`
    from Signals and branch on it, which meant the generic worker knew what a
    calendar was — and that every new connector needing a write-back would add
    another branch here. The ActionSpec now says whether an action reconciles and
    supplies the adapter that knows how; the worker only asks.

    Takes the detached `_Lease`, not an ORM row: this runs in Phase B with no
    transaction open, so touching a live row here would re-open one.

    Swallows everything: the side effect already happened, and failing the
    request here would both lie about that and re-drive the provider on the next
    pass. The scheduled sync remains the backstop.
    """
    spec = get_spec(lease.action_type)
    if spec is None or spec.reconciler is None:
        return
    if result.reference is not None and result.reference.extra.get("dry_run"):
        # A dry run changed nothing anywhere, so there is nothing to read back.
        # Provider-agnostic: `dry_run` is part of the ProviderReference contract.
        return
    try:
        await spec.reconciler(
            lease.org_id,
            lease.action_type,
            lease.payload or {},
            result.reference,
            override=reconciler,
        )
    except Exception:  # noqa: BLE001 — see docstring
        logger.exception(
            "read-model reconcile failed for execution %s (%s) "
            "(side effect DID happen; sync will converge)",
            lease.execution_id, lease.action_type,
        )


def _provider_reference_id(req) -> str | None:
    """The external id a prior attempt recorded, if any. Dry-run references are
    ignored: they represent no real side effect, so re-running is safe (and is
    what local dev expects when a real sender is later wired in)."""
    ref = getattr(req, "provider_reference", None)
    if not isinstance(ref, dict) or ref.get("dry_run"):
        return None
    external_id = ref.get("external_id")
    return str(external_id) if external_id else None


async def _maybe_dlq(dlq, req, provider, error, max_attempts) -> None:
    if (req.attempts or 0) >= max_attempts:
        await dlq.record(req, provider=getattr(provider, "name", None), error=error)


async def relay_pass(repo, *, limit: int = 500) -> int:
    """Publish persisted outbox events, then mark them published.

    `published_at` MEANS "every subscriber that was supposed to run for this
    event ran without raising" — not merely "publish() was called".

    THE DEFECT THIS FIXES. This relay ignored the return value of `publish()` and
    marked every row published unconditionally. `Delivery` exists precisely so an
    outbox relay can tell the difference, and both sibling relays already used it
    (`signals/outbox.py`, `memory/outbox.py`); this one was missed.

    The consequence landed exactly on the Execution -> Observation boundary. If
    the Observation subscriber raised — a database timeout, a transient
    connection error — the ACTION_EXECUTED row was marked published anyway and
    the outcome was PERMANENTLY LOST, with only a best-effort Redis dead letter
    behind it. That outcome is the sole durable record that the action happened,
    for Learning and for audit; the execution row itself is unaffected, so
    nothing surfaced the loss.

    Leaving a row unpublished is what makes it retryable, and re-delivery is safe
    by construction: `outcome_id` is a deterministic uuid5 of (source_event_id,
    outcome_type) and the consumer additionally claims the event in the
    observation inbox, so a replayed event converges on ONE outcome.

    `ensure_subscribers()` first, for the same reason the other two relays call
    it: in-process delivery depends on the composition root having been imported
    by THIS process, and publishing into an empty bus reports success while
    dropping the event.
    """
    ensure_subscribers()
    events = await repo.pending_events(limit=limit)
    published_ids = []
    failed = 0
    for e in events:
        delivery = await event_bus.publish(
            DomainEvent(
                type=EventType(e.event_type),
                org_id=e.organization_id,
                source="execution",
                # Stable event id so idempotent consumers can dedupe.
                payload={**e.payload, "event_id": str(e.id)},
            )
        )
        if delivery.ok:
            published_ids.append(e.id)
        else:
            # Already logged and dead-lettered by the bus. Staying unpublished is
            # the retry.
            failed += 1
            logger.warning(
                "execution outbox row=%s retained for retry: %d/%d handler(s) "
                "failed on %s",
                e.id, delivery.failed, delivery.handlers, e.event_type,
            )
    await repo.mark_published(published_ids)
    if published_ids or failed:
        logger.info("relay published=%d retained=%d", len(published_ids), failed)
    return len(published_ids)


# ---- Celery tasks (each pass in its own transaction) ----

def _repo(db):
    from navis.execution.repository import ExecutionRepository

    return ExecutionRepository(db)


async def _execute() -> dict:
    from navis.execution.providers import production_providers
    from navis.execution.ratelimit import default_rate_limiter
    from navis.platform.database import session_scope

    @asynccontextmanager
    async def unit():
        """ONE session per phase, committed on exit.

        This is the whole point of the claim/execute/persist split: production
        opens a transaction for the claim, closes it, calls Google with NOTHING
        held, then opens a second transaction to record the result. Passing a
        single session here (as this did) is what put every Google call of the
        pass inside one transaction.
        """
        async with session_scope() as db:
            yield _repo(db)

    # Production path: real Google Calendar + Gmail senders wired in.
    # Tests that call execute_pass() directly still get the dry-run
    # defaults (see providers/__init__.py) so no test hits Google APIs.
    return await execute_pass(
        unit=unit,
        providers=production_providers(),
        rate_limiter=default_rate_limiter(),
    )


async def _relay() -> int:
    from navis.platform.database import session_scope

    async with session_scope() as db:
        return await relay_pass(_repo(db))


async def _recover() -> dict:
    from navis.execution.recovery import ExecutionRecoveryService
    from navis.platform.database import session_scope

    async with session_scope() as db:
        return await ExecutionRecoveryService(_repo(db)).recover()


def _register_tasks():
    from navis.platform.queue import celery_app, run_async

    @celery_app.task(name="navis.execution.worker.execute_approved_actions")
    def execute_approved_actions() -> dict:
        return run_async(_execute())

    @celery_app.task(name="navis.execution.worker.relay_execution_outbox")
    def relay_execution_outbox() -> int:
        return run_async(_relay())

    @celery_app.task(name="navis.execution.worker.recover_stale_executions")
    def recover_stale_executions() -> dict:
        return run_async(_recover())

    return execute_approved_actions, relay_execution_outbox, recover_stale_executions


execute_approved_actions, relay_execution_outbox, recover_stale_executions = _register_tasks()
