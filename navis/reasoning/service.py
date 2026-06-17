"""Run all reasoning engines over a state snapshot."""
from __future__ import annotations

from navis.reasoning.base import ReasoningSignal
from navis.reasoning.engines import ENGINES
from navis.platform.monitoring import get_logger
from navis.state.aggregate import OrganizationalState

logger = get_logger("reasoning")


def run_reasoning(state: OrganizationalState) -> list[ReasoningSignal]:
    """Fan the state through every engine; return the union of signals.

    Pure and synchronous — engines do no I/O. An engine that raises is isolated
    so one bad engine never blanks the whole reasoning pass.
    """
    signals: list[ReasoningSignal] = []
    for engine in ENGINES:
        try:
            signals.extend(engine.analyze(state))
        except Exception:  # pragma: no cover - defensive
            logger.exception("engine %s failed", engine.__class__.__name__)
    logger.info("reasoning produced %d signals for org=%s", len(signals), state.org_id)
    return signals
