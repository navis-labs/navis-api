"""Execution worker — runs providers off the request path, relays the outbox.

Horizontally scalable (FOR UPDATE SKIP LOCKED, no global locks). Hardened:
  * claims APPROVED *and* retryable FAILED (attempts < max) requests;
  * gates each provider behind a per-provider rate limiter (skip = backpressure);
  * records telemetry; dead-letters on exhausted attempts / validation errors;
  * relays the transactional outbox (at-least-once; consumers dedupe via the inbox).

The HTTP API never runs a provider — it changes state + enqueues this worker.
"""
from __future__ import annotations

import asyncio
from time import monotonic

from navis.execution.config import MAX_EXECUTION_ATTEMPTS
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
) -> dict:
    svc = ExecutionService(repo=repo)
    dlq = ExecutionDeadLetterService(repo)
    tel = telemetry or provider_telemetry

    claimed = await repo.claim_executable(max_attempts=max_attempts, limit=limit)
    executed = failed = skipped = 0

    for req in claimed:
        provider = get_provider(req.action_type, providers)
        if provider is None:
            await svc.fail_execution(req, f"no provider for {req.action_type}")
            await dlq.record(req, provider=None, error="no provider")  # unrecoverable
            failed += 1
            continue

        # Rate-limit gate: leave APPROVED/FAILED for the next pass (no storm).
        if rate_limiter is not None and not await rate_limiter.allow(provider.name):
            skipped += 1
            continue

        await svc.begin_execution(req)  # -> EXECUTING, attempts += 1
        t0 = monotonic()
        try:
            result = await execute_with_retry(provider, req.payload, policy)
            tel.record(provider.name, ok=result.ok, retried=False, latency_s=monotonic() - t0)
            if result.ok:
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
    from navis.execution.ratelimit import default_rate_limiter
    from navis.platform.database import session_scope

    async with session_scope() as db:
        return await execute_pass(_repo(db), rate_limiter=default_rate_limiter())


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
    from navis.platform.queue import celery_app

    @celery_app.task(name="navis.execution.worker.execute_approved_actions")
    def execute_approved_actions() -> dict:
        return asyncio.run(_execute())

    @celery_app.task(name="navis.execution.worker.relay_execution_outbox")
    def relay_execution_outbox() -> int:
        return asyncio.run(_relay())

    @celery_app.task(name="navis.execution.worker.recover_stale_executions")
    def recover_stale_executions() -> dict:
        return asyncio.run(_recover())

    return execute_approved_actions, relay_execution_outbox, recover_stale_executions


execute_approved_actions, relay_execution_outbox, recover_stale_executions = _register_tasks()
