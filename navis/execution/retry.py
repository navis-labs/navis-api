"""Execution retry — exponential backoff + jitter, bounded attempts, 429-aware.

Retries only on retryable provider errors (transient/429/timeouts). Validation
errors and non-retryable failures fail fast. No infinite retries.
"""
from __future__ import annotations

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
            result = await provider.execute(payload)
            if not result.ok and result.retryable:
                raise ProviderError(result.error or "retryable provider error", retryable=True)
            return result
    raise AssertionError("unreachable")  # reraise=True exits via exception
