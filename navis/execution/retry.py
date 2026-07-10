"""Execution retry — exponential backoff + jitter, bounded attempts, 429-aware.

Retries only on retryable provider errors (transient/429/timeouts). Validation
errors and non-retryable failures fail fast. No infinite retries.

Timeout ownership lives HERE (mirrors ml's ResilientProvider owning the model-call
timeout): a provider execute() gets a hard liveness bound. Without one, a hung real
sender (socket with no deadline) pins a worker slot forever and pushes the request
past the stale-recovery window — recovery then reaps it and a second worker re-runs
it while the first is still in flight. A timeout is NOT retried in-call (the side
effect may have fired — retrying risks a double send); the pass-level FAILED +
attempts budget remains the retry authority.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from navis.execution.contract import ProviderResult
from navis.execution.providers.base import Provider, ProviderError


@dataclass(frozen=True, slots=True)
class ExecutionRetryPolicy:
    max_attempts: int = 5
    initial_backoff: float = 0.5
    max_backoff: float = 30.0
    jitter: float = 0.5
    # Hard liveness bound on ONE provider execute() call. Keep well under
    # config.STALE_EXECUTING_SECONDS so an in-flight request can never look orphaned.
    execute_timeout_s: float = 60.0


DEFAULT_RETRY_POLICY = ExecutionRetryPolicy()


def _is_retryable(exc: BaseException) -> bool:
    return isinstance(exc, ProviderError) and exc.retryable


async def execute_with_retry(
    provider: Provider, payload: dict[str, Any], policy: ExecutionRetryPolicy | None = None
) -> ProviderResult:
    """Validate (no retry), then execute with bounded retry on retryable errors.

    Returns a ProviderResult. Raises ProviderValidationError on bad payload and
    re-raises the last ProviderError if retries are exhausted (caller -> FAILED).
    """
    await provider.validate(payload)  # ProviderValidationError is not retried

    policy = policy or DEFAULT_RETRY_POLICY
    retrying = AsyncRetrying(
        stop=stop_after_attempt(policy.max_attempts),
        wait=wait_exponential_jitter(
            initial=policy.initial_backoff, max=policy.max_backoff, jitter=policy.jitter
        ),
        retry=retry_if_exception(_is_retryable),
        reraise=True,
    )
    async for attempt in retrying:
        with attempt:
            try:
                result = await asyncio.wait_for(
                    provider.execute(payload), timeout=policy.execute_timeout_s
                )
            except TimeoutError:
                # Ambiguous outcome (the send may have fired) — never retried in-call.
                # The request goes FAILED; the pass-level attempts budget governs re-drive.
                raise ProviderError(
                    f"{provider.name} execute timed out after {policy.execute_timeout_s}s",
                    retryable=False,
                )
            if not result.ok and result.retryable:
                raise ProviderError(result.error or "retryable provider error", retryable=True)
            return result
    raise AssertionError("unreachable")  # reraise=True exits via exception
