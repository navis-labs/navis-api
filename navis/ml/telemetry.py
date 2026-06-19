"""ML observability — per-org token/cost/latency accounting for every call.

Every generation records: prompt id+version, model, latency, tokens, estimated cost,
success. Aggregated per organization. Cost is estimated from a transparent price
table (USD per 1M tokens); update as vendor pricing changes.
"""
from __future__ import annotations

import uuid
from collections import defaultdict
from dataclasses import dataclass, field

from navis.ml.providers.base import GenResult
from navis.platform.monitoring import get_logger

logger = get_logger("ml.telemetry")

# USD per 1M tokens (input, output). Estimates — keep in sync with vendor pricing.
_PRICE: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.0),
    "claude-opus-4-8": (15.0, 75.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
    "gemini-1.5-pro": (1.25, 5.0),
    "navis-deterministic-1": (0.0, 0.0),
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    pin, pout = _PRICE.get(model, (0.0, 0.0))
    return round(input_tokens / 1_000_000 * pin + output_tokens / 1_000_000 * pout, 6)


@dataclass(slots=True)
class OrgUsage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    failures: int = 0


@dataclass
class MLTelemetry:
    _by_org: dict[uuid.UUID, OrgUsage] = field(default_factory=lambda: defaultdict(OrgUsage))

    def record(
        self,
        org_id: uuid.UUID,
        result: GenResult,
        *,
        prompt_id: str | None,
        prompt_version: int | None,
        success: bool = True,
    ) -> float:
        cost = estimate_cost(result.model, result.input_tokens, result.output_tokens)
        u = self._by_org[org_id]
        u.calls += 1
        u.input_tokens += result.input_tokens
        u.output_tokens += result.output_tokens
        u.cost_usd = round(u.cost_usd + cost, 6)
        if not success:
            u.failures += 1
        logger.info(
            "ml call org=%s prompt=%s v=%s model=%s in=%d out=%d latency_ms=%d cost_usd=%.6f ok=%s",
            org_id, prompt_id, prompt_version, result.model,
            result.input_tokens, result.output_tokens, result.latency_ms, cost, success,
        )
        return cost

    def usage(self, org_id: uuid.UUID) -> OrgUsage:
        return self._by_org.get(org_id, OrgUsage())


telemetry = MLTelemetry()
