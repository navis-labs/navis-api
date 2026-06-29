"""Reasoning — Layer 4. Deterministic interpretation over State V2.

The production implementation lives in `navis.reasoning.v2`: engines read the
State V2 `OrganizationalState` and emit `ReasoningSignal`s. Import directly from
`navis.reasoning.v2` (and `navis.reasoning.v2.contract`); this package no longer
re-exports anything, so importing it pulls in no engine code.
"""
