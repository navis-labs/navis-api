"""Greedy, deterministic budget compression.

Admit ranked entries until the token budget is exhausted, then drop the tail (never
summarize away a citation — provenance is preserved for every admitted entity). An
unbounded budget admits everything (FULL_ORG parity). Order in == order out for admitted
entries, so the result is fully reproducible.
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
        # Fast path: an unbounded budget admits everything (FULL_ORG). Skip per-entity
        # token serialization entirely — it would gate nothing and is O(entities × size).
        if budget.max_tokens is None:
            present: list[ScopeEntry] = []
            missing: list[ScopeEntry] = []
            for e in ranked:
                target = present if state.entities.get(e.entity_type, {}).get(e.entity_id) is not None else missing
                target.append(e)
            report = BudgetReport(
                token_estimate=0, admitted=len(present), dropped=len(missing), max_tokens=None,
            )
            return present, missing, report

        admitted: list[ScopeEntry] = []
        dropped: list[ScopeEntry] = []
        used = 0
        for entry in ranked:
            raw = state.entities.get(entry.entity_type, {}).get(entry.entity_id)
            if raw is None:
                dropped.append(entry)
                continue
            cost = entity_tokens(raw)
            if used + cost > budget.max_tokens:
                dropped.append(entry)  # over budget → drop, keep scanning (deterministic)
                continue
            admitted.append(entry)
            used += cost

        report = BudgetReport(
            token_estimate=used, admitted=len(admitted), dropped=len(dropped),
            max_tokens=budget.max_tokens,
        )
        return admitted, dropped, report
