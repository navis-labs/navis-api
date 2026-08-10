"""Execution worker — runs providers off the request path, relays the outbox.

Horizontally scalable (FOR UPDATE SKIP LOCKED, no global locks). Hardened:
  * claims APPROVED *and* retryable FAILED (attempts < max) requests;
  * gates each provider behind a per-provider rate limiter (skip = backpressure);
  * records telemetry; dead-letters on exhausted attempts / validation errors;
  * relays the transactional outbox (at-least-once; consumers dedupe via the inbox).

The HTTP API never runs a provider — it changes state + enqueues this worker.
"""
from __future__ import annotations

from time import monotonic

from navis.execution.config import MAX_EXECUTION_ATTEMPTS
from navis.execution.contract import ProviderReference, ProviderResult
from navis.execution.dead_letter import ExecutionDeadLetterService
from navis.execution.providers import get_provider
from navis.execution.providers.base import ProviderValidationError
from navis.execution.retry import execute_with_retry
from navis.execution.service import ExecutionService
from navis.execution.telemetry import provider_telemetry, request_durations
from navis.platform.events import DomainEvent, EventType, event_bus
from navis.platform.monitoring import get_logger

logger = get_logger("execution.worker")


async def execute_pass(
    repo,
    *,
    providers=None,
    policy=None,
    limit: int = 100,
    rate_limiter=None,
    max_attempts: int = MAX_EXECUTION_ATTEMPTS,
    telemetry=None,
    reconciler=None,
) -> dict:
    """`reconciler` is injected so tests can assert the read-model write-back
    happened WITHOUT reaching Google. `None` means "resolve the production one
    lazily" — importing it at module scope would drag the Signals stack into
    every test that touches the worker."""
    svc = ExecutionService(repo=repo)
    dlq = ExecutionDeadLetterService(repo)
    tel = telemetry or provider_telemetry

    claimed = await repo.claim_executable(max_attempts=max_attempts, limit=limit)
    executed = failed = skipped = 0

    for req in claimed:
        provider = get_provider(req.action_type, providers)
        if provider is None:
            # APPROVED -> FAILED is NOT a legal edge (status.py), so failing
            # straight from the claimed state raised InvalidTransition out of
            # the pass — taking every OTHER claimed row down with it. Step
            # through EXECUTING, which is the same path every other terminal
            # outcome takes, so one unroutable row can only fail itself.
            await svc.begin_execution(req)
            await svc.fail_execution(req, f"no provider for {req.action_type}")
            await dlq.record(req, provider=None, error="no provider")  # unrecoverable
            failed += 1
            continue

        # Already-executed guard. A claimed row can carry a provider reference
        # when a PREVIOUS attempt reached the provider successfully but did not
        # get to record EXECUTED — the stale-EXECUTING recovery path pushes such
        # rows to FAILED, which makes them re-claimable here. Re-running the
        # provider would create a SECOND calendar event / send a SECOND email.
        # The reference is proof the side effect already happened, so close the
        # row out instead of repeating it.
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
            executed += 1
            continue

        # Rate-limit gate: leave APPROVED/FAILED for the next pass (no storm).
        if rate_limiter is not None and not await rate_limiter.allow(provider.name):
            skipped += 1
            continue

        await svc.begin_execution(req)  # -> EXECUTING, attempts += 1
        t0 = monotonic()
        try:
            # Thread execution context (org_id, execution_id) into the
            # payload under a reserved key so providers whose real senders
            # need to resolve org-scoped resources (e.g. CalendarProvider
            # resolving the org's Google Calendar Integration) can do it
            # without extending the Provider.execute() signature. Keys are
            # stripped by providers that don't care; unknown callers just
            # ignore the reserved namespace.
            augmented = {
                **(req.payload or {}),
                "__ctx__": {
                    "org_id": str(req.organization_id),
                    "execution_id": str(req.id),
                    "decision_id": str(req.decision_id),
                },
            }
            result = await execute_with_retry(provider, augmented, policy)
            tel.record(provider.name, ok=result.ok, retried=False, latency_s=monotonic() - t0)
            if result.ok:
                # READ-AFTER-WRITE. Pull the mutated calendar event's CURRENT
                # state into signal_events BEFORE the row reaches EXECUTED, so
                # the transition carries a real guarantee:
                #
                #     status == EXECUTED => Navis's read model reflects the change.
                #
                # Without this, Execution wrote to Google and nothing told
                # Navis — the dashboard card, the chat fallback and Memory all
                # kept serving the cancelled meeting until an unrelated Celery
                # sync happened to run. Non-fatal by design (see reconcile's
                # docstring): a bookkeeping failure must never flip a
                # genuinely-successful cancellation to FAILED.
                await _reconcile_read_model(req, result, reconciler)
                await svc.complete_execution(req, result)
                logger.info("executed %s %s", req.id, request_durations(req))
                executed += 1
            else:
                await svc.fail_execution(req, result.error or "provider failed")
                await _maybe_dlq(dlq, req, provider, result.error or "provider failed", max_attempts)
                failed += 1
        except ProviderValidationError as exc:
            tel.record(provider.name, ok=False, retried=False, latency_s=monotonic() - t0)
            await svc.fail_execution(req, f"validation: {exc}")
            await dlq.record(req, provider=provider.name, error=str(exc))  # never recovers
            failed += 1
        except Exception as exc:  # retries exhausted / unexpected
            tel.record(provider.name, ok=False, retried=True, latency_s=monotonic() - t0)
            await svc.fail_execution(req, str(exc))
            await _maybe_dlq(dlq, req, provider, str(exc), max_attempts)
            failed += 1

    if claimed:
        logger.info("execute_pass executed=%d failed=%d skipped=%d", executed, failed, skipped)
    return {"executed": executed, "failed": failed, "skipped": skipped}


async def _reconcile_read_model(req, result, reconciler) -> None:
    """Write the just-mutated calendar event back into Navis's read model.

    Swallows everything: the side effect on Google already happened, and
    failing the request here would both lie about that and re-drive the
    provider on the next pass. The scheduled sync remains the backstop.
    """
    from navis.signals.calendar.reconcile import (
        CALENDAR_MUTATION_ACTION_TYPES,
        event_id_for,
    )

    if req.action_type not in CALENDAR_MUTATION_ACTION_TYPES:
        return
    if result.reference is not None and result.reference.extra.get("dry_run"):
        # A dry-run provider changed nothing on Google, so there is nothing to
        # read back. Reconciling would spend a session and a Google call to
        # re-confirm a state we never altered.
        return
    external_id = event_id_for(
        req.action_type,
        req.payload or {},
        result.reference.external_id if result.reference else None,
    )
    if not external_id:
        logger.warning(
            "execution %s (%s) produced no calendar event id; read model not "
            "reconciled", req.id, req.action_type,
        )
        return
    if reconciler is None:
        from navis.signals.calendar.reconcile import reconcile_calendar_event

        reconciler = reconcile_calendar_event
    try:
        await reconciler(req.organization_id, external_id)
    except Exception:  # noqa: BLE001 — see docstring
        logger.exception(
            "read-model reconcile failed for execution %s event=%s "
            "(side effect DID happen; sync will converge)", req.id, external_id,
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
    """Publish persisted outbox events, then mark them published. A crash between
    publish and mark re-publishes next pass; consumers dedupe via the inbox."""
    events = await repo.pending_events(limit=limit)
    published_ids = []
    for e in events:
        await event_bus.publish(
            DomainEvent(
                type=EventType(e.event_type),
                org_id=e.organization_id,
                source="execution",
                # Stable event id so idempotent consumers can dedupe.
                payload={**e.payload, "event_id": str(e.id)},
            )
        )
        published_ids.append(e.id)
    await repo.mark_published(published_ids)
    if published_ids:
        logger.info("relay published=%d", len(published_ids))
    return len(published_ids)


# ---- Celery tasks (each pass in its own transaction) ----

def _repo(db):
    from navis.execution.repository import ExecutionRepository

    return ExecutionRepository(db)


async def _execute() -> dict:
    from navis.execution.providers import production_providers
    from navis.execution.ratelimit import default_rate_limiter
    from navis.platform.database import session_scope

    # Production path: real Google Calendar + Gmail senders wired in.
    # Tests that call execute_pass() directly still get the dry-run
    # defaults (see providers/__init__.py) so no test hits Google APIs.
    async with session_scope() as db:
        return await execute_pass(
            _repo(db),
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
