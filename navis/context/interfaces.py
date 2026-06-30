"""Protocols — the seams Context depends on, not concrete classes (dependency inversion).

ContextService talks only to these; any implementation (the deterministic defaults today,
an injected variant tomorrow) plugs in without changing the service. This is the extension
point for future selection/ranking/retrieval WITHOUT coupling Context to ml/ or any
concrete class.

Contract that EVERY protocol here upholds (so the Context layer's guarantees survive a
decade of new implementations):

  * Pure & deterministic — the synchronous protocols (Selector, PermissionResolver, Ranker,
    Compressor, RetrievalProvider) are pure functions of their inputs: no datetime.now(),
    no uuid4, no randomness, no I/O. The logical clock is always the State's `as_of`. Same
    inputs ⇒ identical outputs, forever, in any region. This is why they are sync, not
    async: an async implementation would imply I/O, which would break replay.
  * Read-only inputs (MUTATION RULE) — implementations MUST NOT mutate any argument
    (`state`, `scope`, `ranked`, …). State is the authoritative, shared projection;
    mutating it would corrupt every other reader. Treat all inputs as immutable and return
    NEW objects.
  * Caller owns the result — returned objects belong to the caller; an implementation must
    not retain or later mutate them.
  * Thread/async safety — implementations are expected to be STATELESS (or hold only
    immutable config), so a single instance is safe to share across concurrently building
    bundles. The async I/O protocols (StateReader, ContextCache) carry no cross-call state.
"""
from __future__ import annotations

import uuid
from typing import Protocol, runtime_checkable

from navis.context.contracts import (
    ActorRef,
    ContextBundle,
    PermissionScope,
    TokenBudget,
)
from navis.context.models import BudgetReport, ContextScope, ScopeEntry
from navis.context.tasks import TaskSpec
from navis.state.v2.models import OrganizationalState


@runtime_checkable
class StateReader(Protocol):
    """The ONLY inbound dependency: read the authoritative current OrganizationalState.

    Async because it is the one I/O boundary (it may rebuild State from Memory). Returns a
    fresh, caller-owned projection. The Context layer reads State and NOTHING else — never
    Memory, Signals, or the database directly (CQRS read-model boundary)."""

    async def get_state(self, org_id: uuid.UUID) -> OrganizationalState: ...


@runtime_checkable
class Selector(Protocol):
    """Task spec -> candidate ContextScope over a State (pure, deterministic).

    Must not mutate `state`. Returns a new ContextScope (which self-canonicalizes its order)."""

    def select(self, state: OrganizationalState, spec: TaskSpec) -> ContextScope: ...


@runtime_checkable
class PermissionResolver(Protocol):
    """Filter a candidate scope down to what an actor may see (DEFAULT-DENY, pure).

    Runs BEFORE ranking/budgeting, so anything an actor may not see can never enter a
    bundle. Must not mutate `scope` or `state`; returns a NEW, narrowed ContextScope."""

    def resolve(
        self,
        actor: ActorRef,
        permissions: PermissionScope,
        scope: ContextScope,
        state: OrganizationalState,
    ) -> ContextScope: ...


@runtime_checkable
class Ranker(Protocol):
    """Order a scope's entries by deterministic STRUCTURAL signals only — never risk,
    priority, health, or any score (those are Reasoning's exclusively).

    Must produce a TOTAL, stable order (tie-broken to a unique result) so the bundle is
    reproducible. Must not mutate inputs; returns a new ordered list owned by the caller."""

    def rank(self, state: OrganizationalState, scope: ContextScope) -> list[ScopeEntry]: ...


@runtime_checkable
class Compressor(Protocol):
    """Admit ranked entries until a token budget is hit (greedy, deterministic).

    Returns a 3-tuple in a FIXED order: (admitted, dropped, report)
      * admitted: list[ScopeEntry] kept within budget, in the input (ranked) order;
      * dropped:  list[ScopeEntry] excluded (over budget or absent from State);
      * report:   BudgetReport accounting (admitted == len(admitted), etc.).
    An unbounded budget admits everything. Must not mutate inputs."""

    def compress(
        self,
        ranked: list[ScopeEntry],
        state: OrganizationalState,
        budget: TokenBudget,
    ) -> tuple[list[ScopeEntry], list[ScopeEntry], BudgetReport]: ...


@runtime_checkable
class RetrievalProvider(Protocol):
    """Optional extension point: restrict candidate entity_ids BEFORE permission filtering.

    Return semantics (a meaningful sentinel, not ambiguity):
      * None        -> NO restriction (the default; all candidates pass);
      * set[str]    -> restrict to exactly these entity_ids (an empty set admits nothing).

    DELIBERATELY SYNC AND PURE: a provider must be a deterministic function of
    (state, scope) — it may NOT perform network/disk I/O, because that would break Context's
    replay guarantee. An ML retriever plugs in here ONLY if its scoring is precomputed into
    State (read deterministically), never as a live vector/embedding call. It is injected,
    never referenced — Context imports no ml/ code."""

    def candidates(self, state: OrganizationalState, scope: ContextScope) -> set[str] | None: ...


@runtime_checkable
class ContextCache(Protocol):
    """Content-addressed bundle cache. DISPOSABLE — a miss simply rebuilds from State, so
    the cache is never a source of truth and may be dropped at any time.

    Async because it is an I/O boundary. `get` MUST fail safe: a corrupt or version-mismatched
    entry returns None (a miss), never raises. `put` is keyed by the bundle's content-addressed
    `context_id`, so writing the same bundle twice is idempotent."""

    async def get(self, context_id: str) -> ContextBundle | None: ...

    async def put(self, bundle: ContextBundle) -> None: ...
