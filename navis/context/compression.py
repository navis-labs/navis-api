"""Greedy, deterministic budget compression — the LAST deterministic stage before Reasoning.

Compression decides ONLY what survives a token budget. It MAY admit, drop, and preserve
order; it MUST NEVER summarize, rewrite, reorder, infer, prioritize semantically, or compute
importance/risk/urgency — those belong exclusively to Reasoning. It performs pure budget
ALLOCATION, nothing else.

Algorithm — greedy "skip-and-continue" (NOT stop-at-first-overflow):
  scan `ranked` in order; admit each entry that fits the remaining budget, accumulating cost;
  an entry that does NOT fit is dropped and scanning CONTINUES (so a smaller, lower-ranked
  entry can still fill the leftover budget). WHY this is correct:
    * It maximizes the entities Reasoning sees within the budget (better utilization than
      stopping at the first oversize entry, which would waste the remaining room).
    * It is deterministic and does NOT reorder — admitted entries stay in ranked order
      (appended during the in-order scan); it is a pure size-vs-budget decision, never a
      semantic re-prioritization.
  An unbounded budget admits everything (FULL_ORG parity) via a fast path that skips token
  serialization entirely.

Invariants (hold by construction; covered by tests):
  * used <= budget.max_tokens               (a cost is added only when it still fits)
  * admitted + dropped == ranked            (every entry lands in exactly one list)
  * admitted/dropped preserve ranked order  (subsequences of `ranked`)

Fail-safe: token estimation is wrapped — a malformed entity that cannot be sized is DROPPED
(fail-closed: never overflow, never crash). Pure, deterministic, replay-safe: same
(ranked, state, budget) ⇒ same output, forever (no clock, no randomness, no mutable global,
no cache). Observability (admitted/dropped counts) lives in the orchestrating service, not in
this pure function; an un-sizable entity is an upstream data-quality issue, observed at
State/Memory ingestion.
"""
from __future__ import annotations

from navis.context.budget import entity_tokens
from navis.context.contracts import TokenBudget
from navis.context.models import BudgetReport, ScopeEntry
from navis.state.v2.models import OrganizationalState


class GreedyCompressor:
    """The default Compressor: pure function of (ranked, state, budget)."""

    def compress(
        self,
        ranked: list[ScopeEntry],
        state: OrganizationalState,
        budget: TokenBudget,
    ) -> tuple[list[ScopeEntry], list[ScopeEntry], BudgetReport]:
        entities = state.entities

        # Fast path: an unbounded budget admits everything (FULL_ORG). Skip per-entity token
        # serialization entirely — it would gate nothing and is O(entities × size). Entries
        # whose entity is not backed by State are still separated out (never fabricated).
        if budget.max_tokens is None:
            present: list[ScopeEntry] = []
            missing: list[ScopeEntry] = []
            for entry in ranked:
                target = present if entities.get(entry.entity_type, {}).get(entry.entity_id) is not None else missing
                target.append(entry)
            report = BudgetReport(
                token_estimate=0, admitted=len(present), dropped=len(missing), max_tokens=None,
            )
            return present, missing, report

        admitted: list[ScopeEntry] = []
        dropped: list[ScopeEntry] = []
        used = 0
        for entry in ranked:
            raw = entities.get(entry.entity_type, {}).get(entry.entity_id)
            if raw is None:
                dropped.append(entry)  # not backed by State → drop (never fabricate)
                continue
            try:
                cost = entity_tokens(raw)
            except Exception:
                # Fail-closed: a malformed entity that cannot be sized is dropped rather than
                # risking a budget overflow or a crash. Compression never throws on bad data.
                dropped.append(entry)
                continue
            if used + cost > budget.max_tokens:
                dropped.append(entry)  # over budget → drop, keep scanning (skip-and-continue)
                continue
            admitted.append(entry)
            used += cost

        report = BudgetReport(
            token_estimate=used, admitted=len(admitted), dropped=len(dropped),
            max_tokens=budget.max_tokens,
        )
        return admitted, dropped, report
