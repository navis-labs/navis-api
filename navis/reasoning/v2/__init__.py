"""Reasoning V2 — deterministic interpretation over State V2.

State answers "what is true right now?"; Reasoning answers "what does it mean?".
Reasoning V2 consumes ONLY `state.v2` OrganizationalState and emits ReasoningSignals.
It never queries Memory/Signals/DB/Redis/Google/Azure, and it computes everything
itself (no LLM) — it must run identically with Azure OpenAI completely offline.

Determinism law: pure function of (OrganizationalState). The state's `as_of` is the
ONLY clock; no datetime.now()/random()/uuid4(). Replaying the same State snapshot
yields byte-identical signals (including signal_id and created_at).
"""
from navis.reasoning.v2.contract import (
    REASONING_VERSION,
    ReasoningSignal,
    SignalType,
)
from navis.reasoning.v2.service import run_reasoning_v2

__all__ = ["REASONING_VERSION", "ReasoningSignal", "SignalType", "run_reasoning_v2"]
