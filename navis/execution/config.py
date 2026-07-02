"""Execution hardening constants (tunable, versioned with deploys)."""
from __future__ import annotations

# A request is retried until attempts reach this, then dead-lettered.
MAX_EXECUTION_ATTEMPTS = 5

# An EXECUTING request whose row hasn't changed in this long is assumed orphaned
# (worker died mid-execution) and is recovered to FAILED for re-drive.
STALE_EXECUTING_SECONDS = 300

# Per-provider rate limits (max executions per second across all workers).
PROVIDER_RATE_LIMITS: dict[str, int] = {
    "gmail": 50,
    "calendar": 25,
    "slack": 50,
    "tasks": 500,  # internal, cheap
}
