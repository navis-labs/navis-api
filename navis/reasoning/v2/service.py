"""run_reasoning_v2 — the deterministic reasoning pass over State V2.

Pure: OrganizationalState -> ReasoningSignal[]. No clock, no randomness, no I/O,
no LLM. The state's `as_of` is the logical time stamped on every signal, so
replaying the same State snapshot reproduces byte-identical output (including
signal_id and created_at). Per-engine isolation: one failing engine never blanks
the pass.
"""
from __future__ import annotations

from collections.abc import Mapping

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
    state: OrganizationalState,
    *,
    prior_state: OrganizationalState | None = None,
    evidence: Mapping[str, frozenset[str]] | None = None,
) -> list[ReasoningSignal]:
    """Pure: (State, prior, evidence) -> ReasoningSignal[].

    `evidence` maps entity_id -> observed `signal_importance.classification` set
    (see `memory.lineage.classifications_by_entity`). It is READ by the caller
    and handed in, never fetched here — Reasoning performs no I/O. Omitting it
    is safe: every engine treats an absent entity as UNKNOWN and keeps its
    prior behaviour.
    """
    wm = state.watermark.last_memory_id
    ctx = ReasoningContext(
        org_id=state.org_id,
        as_of=state.as_of,
        watermark=str(wm) if wm else None,
        reasoning_version=REASONING_VERSION,
        prior=prior_state,
        evidence=evidence or {},
    )

    signals: list[ReasoningSignal] = []
    for engine in ENGINES:
        try:
            for draft in engine.analyze(state, ctx):
                signals.append(_stamp(draft, engine, ctx))
        except Exception:  # pragma: no cover - one bad engine never blanks the pass
            logger.exception("reasoning engine %s failed", engine.name)

    # Deterministic order: severity desc, then signal_id for a stable tiebreak.
    signals.sort(key=lambda s: (-s.severity, str(s.signal_id)))
    logger.info("reasoning v2 produced %d signals org=%s", len(signals), state.org_id)
    return signals
