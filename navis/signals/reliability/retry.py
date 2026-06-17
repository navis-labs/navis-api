"""RetryPolicy — exponential backoff with jitter, honoring server Retry-After.

Wraps tenacity (proven) rather than hand-rolling. Used by the Google transport
(per HTTP call) and by the sync engine (per integration pass).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_attempts: int = 5
    initial_backoff: float = 0.5
    max_backoff: float = 30.0
    jitter: float = 0.5


DEFAULT_RETRY_POLICY = RetryPolicy()


def _retry_after_aware_wait(policy: RetryPolicy):
    base = wait_exponential_jitter(
        initial=policy.initial_backoff, max=policy.max_backoff, jitter=policy.jitter
    )

    def _wait(retry_state) -> float:
        exc = retry_state.outcome.exception() if retry_state.outcome else None
        retry_after = getattr(exc, "retry_after", None)
        if retry_after is not None:
            try:
                return min(float(retry_after), policy.max_backoff)
            except (TypeError, ValueError):
                pass
        return base(retry_state)

    return _wait


def async_retrying(
    policy: RetryPolicy | None = None,
    *,
    retry_on: Callable[[BaseException], bool],
) -> AsyncRetrying:
    """Build an AsyncRetrying loop. Usage:

        async for attempt in async_retrying(policy, retry_on=is_retryable):
            with attempt:
                return await do_call()
    """
    policy = policy or DEFAULT_RETRY_POLICY
    return AsyncRetrying(
        stop=stop_after_attempt(policy.max_attempts),
        wait=_retry_after_aware_wait(policy),
        retry=retry_if_exception(retry_on),
        reraise=True,
    )
