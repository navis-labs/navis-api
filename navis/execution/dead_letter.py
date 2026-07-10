"""ExecutionDeadLetterService — never lose a failed execution; replay on demand.

When a request exhausts its attempts (or fails validation), the worker records it
here. Replay resets the request's attempts and leaves it FAILED so the worker
re-claims and re-drives it — the side effect is still idempotent (same execution_id),
so replaying a dead letter that actually succeeded won't double-fire.
"""
from __future__ import annotations

import uuid

from navis.execution.models import ExecutionRequest
from navis.execution.status import ExecutionStatus, is_terminal
from navis.platform.monitoring import get_logger

logger = get_logger("execution.dlq")


class ExecutionDeadLetterService:
    def __init__(self, repo) -> None:
        self.repo = repo

    async def record(self, req: ExecutionRequest, *, provider: str | None, error: str) -> None:
        await self.repo.add_dead_letter(req, provider=provider, error=error)
        logger.error("DLQ execution=%s provider=%s err=%s", req.id, provider, error[:200])

    async def replay(self, org_id: uuid.UUID | None = None, *, limit: int = 100) -> dict:
        rows = await self.repo.unresolved_dead_letters(org_id, limit=limit)
        replayed = resolved = 0
        for dl in rows:
            req = await self.repo.get(dl.execution_id)
            if req is None or is_terminal(ExecutionStatus(req.status)):
                # Never replayable: the request is gone, or reached a terminal state
                # (cancelled / rejected / eventually executed). Skipped-but-unresolved,
                # these rows would permanently occupy the front of the LIMIT window and
                # starve every replayable dead letter behind them (the poison-prefix
                # failure shape). Resolve them — WITHOUT claiming a replay.
                await self.repo.mark_dead_letter_resolved(dl.id)
                resolved += 1
                continue
            if req.status != ExecutionStatus.FAILED.value:
                continue  # in-flight (approved/executing) — genuinely pending, retry later
            req.attempts = 0  # reset retry budget; status stays FAILED -> re-claimable
            await self.repo.flush()
            await self.repo.mark_dead_letter_replayed(dl.id)
            replayed += 1
        logger.info("replayed %d dead letters, resolved %d unreplayable", replayed, resolved)
        return {"replayed": replayed, "resolved": resolved}
