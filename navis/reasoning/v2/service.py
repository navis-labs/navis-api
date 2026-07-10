"""run_reasoning_v2 — the deterministic reasoning pass over State V2.

Pure: OrganizationalState -> ReasoningSignal[]. No clock, no randomness, no I/O,
no LLM. The state's `as_of` is the logical time stamped on every signal, so
replaying the same State snapshot reproduces byte-identical output (including
signal_id and created_at). Per-engine isolation: one failing engine never blanks
the pass.

Output pipeline: engines' signals are MERGED into one list, de-duplicated by the
content-addressed signal_id (the runner guarantees unique ids), then ordered
deterministically (severity desc, signal_id tiebreak). A future AI runner is an
ADDITIVE source merged at the same point — the contract and this signature never change.
"""
from __future__ import annotations

import uuid

from navis.platform.monitoring import get_logger
from navis.reasoning.v2.base import ReasoningContext
from navis.reasoning.v2.contract import (
    REASONING_VERSION,
    ReasoningSignal,
    SignalDraft,
    compute_signal_id,
)
from navis.reasoning.v2.engines import ENGINES
from navis.state.v2.models import OrganizationalState

logger = get_logger("reasoning.v2")


def _stamp(draft: SignalDraft, engine, ctx: ReasoningContext) -> ReasoningSignal:
    return ReasoningSignal(
        signal_id=compute_signal_id(
            ctx.org_id, ctx.reasoning_version, engine.name, engine.version,
            draft.signal_type, draft.entity_id, draft.dedup_key,
        ),
        org_id=ctx.org_id,
        signal_type=draft.signal_type,
        entity_type=draft.entity_type,
        entity_id=draft.entity_id,
        title=draft.title,
        why=draft.why,
        evidence=draft.evidence,
        confidence=draft.confidence,
        severity=draft.severity,
        engine_name=engine.name,
        engine_version=engine.version,
        state_as_of=ctx.as_of,
        state_watermark=ctx.watermark,
        reasoning_version=ctx.reasoning_version,
        created_at=ctx.as_of,  # logical time = state.as_of (deterministic)
    )


def run_reasoning_v2(
    state: OrganizationalState, *, prior_state: OrganizationalState | None = None
) -> list[ReasoningSignal]:
    wm = state.watermark.last_memory_id
    ctx = ReasoningContext(
        org_id=state.org_id,
        as_of=state.as_of,
        watermark=str(wm) if wm else None,
        reasoning_version=REASONING_VERSION,
        prior=prior_state,
    )

    signals: list[ReasoningSignal] = []
    for engine in ENGINES:
        try:
            for draft in engine.analyze(state, ctx):
                signals.append(_stamp(draft, engine, ctx))
        except Exception:  # pragma: no cover - one bad engine never blanks the pass
            logger.exception("reasoning engine %s failed", engine.name)

    # Collapse by signal_id, then order deterministically. A ReasoningSignal is content-addressed
    # on its identity, so the same signal_id IS the same signal; keep the first occurrence (engines
    # run in a fixed order, so the winner is deterministic). This is a no-op for today's engines
    # (each emits a unique id) — byte-identical output — but it makes the runner's contract
    # "output signal_ids are unique", the invariant any future signal source (e.g. an additive AI
    # runner merged in here) must satisfy without changing this signature.
    unique: dict[uuid.UUID, ReasoningSignal] = {}
    for s in signals:
        unique.setdefault(s.signal_id, s)
    ordered = sorted(unique.values(), key=lambda s: (-s.severity, str(s.signal_id)))
    logger.info("reasoning v2 produced %d signals org=%s", len(ordered), state.org_id)
    return ordered
