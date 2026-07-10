"""ExecutionRecoveryService — re-drive orphaned EXECUTING requests.

If a worker dies mid-execution, its request is stuck EXECUTING. This service claims
rows whose `updated_at` is older than the stale window (FOR UPDATE SKIP LOCKED, so
multiple recovery runners are safe) and transitions them EXECUTING -> FAILED. A
FAILED request with attempts remaining is re-claimable by the worker — i.e. FAILED
+ attempts<max IS the "retryable" state (no new status, no redesign). Idempotent:
once a row leaves EXECUTING it is no longer matched.
"""
from __future__ import annotations

from navis.execution.config import MAX_EXECUTION_ATTEMPTS, STALE_EXECUTING_SECONDS
from navis.execution.dead_letter import ExecutionDeadLetterService
from navis.execution.service import ExecutionService
from navis.platform.monitoring import get_logger

logger = get_logger("execution.recovery")


class ExecutionRecoveryService:
    def __init__(
        self,
        repo,
        *,
        stale_seconds: int = STALE_EXECUTING_SECONDS,
        max_attempts: int = MAX_EXECUTION_ATTEMPTS,
    ) -> None:
        self.repo = repo
        self.stale_seconds = stale_seconds
        self.max_attempts = max_attempts
        self.svc = ExecutionService(repo=repo)
        self.dlq = ExecutionDeadLetterService(repo)

    async def recover(self, *, limit: int = 100) -> dict:
        stale = await self.repo.claim_stale_executing(
            older_than_seconds=self.stale_seconds, limit=limit
        )
        recovered = 0
        for req in stale:
            error = "stale execution timeout (worker presumed dead)"
            await self.svc.fail_execution(req, error)
            recovered += 1
            # Attempts were counted at begin_execution, so a crash-looped request can
            # spend its FINAL attempt here. FAILED + attempts exhausted is no longer
            # claimable, and only the worker's failure paths dead-letter — without this,
            # the request would silently vanish from the replay surface, violating the
            # DLQ's "never lose a failed execution" contract.
            if (req.attempts or 0) >= self.max_attempts:
                await self.dlq.record(req, provider=None, error=f"{error}; attempts exhausted")
        if recovered:
            logger.warning("recovered %d stale EXECUTING requests", recovered)
        return {"recovered": recovered}
