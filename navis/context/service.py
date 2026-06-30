"""ContextService — the SOLE public entry point of the Context layer.

    bundle = await ContextService(db).build(request)

Reads ONLY the authoritative OrganizationalState (via StateService). Never Memory, never
Signals, never DB tables directly, never RAG/embeddings. Produces an immutable, content-
addressed ContextBundle that embeds an INDEPENDENT scoped OrganizationalState (never an
alias of the live projection), so Reasoning consumes it through its existing accessors with
no change and can never corrupt State. Pure + deterministic: identical (state, request) ⇒
byte-identical bundle.
"""
from __future__ import annotations

import copy
import time

from navis.context import metrics
from navis.context.cache import NullContextCache
from navis.context.compression import GreedyCompressor
from navis.context.contracts import (
    CONTEXT_SCHEMA_VERSION,
    CONTEXT_VERSION,
    ContextBundle,
    ContextRequest,
    PermissionScope,
    compute_context_id,
)
from navis.context.exceptions import UnknownContextTask
from navis.context.interfaces import (
    Compressor,
    ContextCache,
    PermissionResolver,
    Ranker,
    RetrievalProvider,
    Selector,
    StateReader,
)
from navis.context.models import ContextScope, ScopeEntry
from navis.context.permissions import (
    DefaultDenyPermissionFilter,
    RolePermissionResolver,
    intersect_scopes,
)
from navis.context.ranking import StructuralRanker
from navis.context.selectors import DeterministicSelector
from navis.context.tasks import TASK_SPECS
from navis.context.telemetry import log_audit, log_built
from navis.state.v2.models import OrganizationalState
from navis.state.v2.service import StateService


class ContextService:
    """Dependency-inverted: every collaborator is a protocol with a deterministic default,
    so an ML retriever/ranker can be injected later without touching this class."""

    def __init__(
        self,
        db=None,
        *,
        state_service: StateReader | None = None,
        selector: Selector | None = None,
        permissions: PermissionResolver | None = None,
        permission_resolver: RolePermissionResolver | None = None,
        ranker: Ranker | None = None,
        compressor: Compressor | None = None,
        cache: ContextCache | None = None,
        retrieval: RetrievalProvider | None = None,
    ) -> None:
        self.state_service: StateReader = state_service if state_service is not None else StateService(db)
        self.selector: Selector = selector or DeterministicSelector()
        self.permissions: PermissionResolver = permissions or DefaultDenyPermissionFilter()
        # The RBAC authority: resolves an actor's identity roles into the scope they are
        # ENTITLED to. The request can only NARROW within it — never widen (no escalation).
        self.permission_resolver: RolePermissionResolver = permission_resolver or RolePermissionResolver()
        self.ranker: Ranker = ranker or StructuralRanker()
        self.compressor: Compressor = compressor or GreedyCompressor()
        self.cache: ContextCache = cache or NullContextCache()
        self.retrieval: RetrievalProvider | None = retrieval

    async def build(self, request: ContextRequest) -> ContextBundle:
        if request.task not in TASK_SPECS:
            raise UnknownContextTask(f"no selection spec for task {request.task!r}")

        # Effective scope = what the actor's roles GRANT ∩ what the request asks for. A
        # non-system actor presenting allow_all is bounded to its role grant here (no
        # privilege escalation), all deterministically (replay-safe).
        effective = intersect_scopes(request.permissions, self.permission_resolver.resolve(request.actor))

        start = time.perf_counter()
        bundle, cache_hit = await self._build(request, effective)
        latency_ms = (time.perf_counter() - start) * 1000.0
        metrics.observe_ms("assembly_ms", latency_ms, labels={"task": request.task.value})
        log_audit(
            org_id=request.org_id,
            actor_fingerprint=request.actor.fingerprint(),
            actor_kind=request.actor.kind,
            task=request.task.value,
            context_id=bundle.context_id,
            permission_fingerprint=effective.fingerprint(),
            policy_fingerprint=request.policy_fingerprint,
            entity_count=bundle.budget_report.admitted,
            bundle_tokens=bundle.budget_report.token_estimate,
            state_as_of=bundle.state_as_of,
            latency_ms=latency_ms,
            cache_hit=cache_hit,
        )
        return bundle

    async def _build(
        self, request: ContextRequest, effective: PermissionScope
    ) -> tuple[ContextBundle, bool]:
        state = await self.state_service.get_state(request.org_id)
        context_id = compute_context_id(request, state)

        cached = await self.cache.get(context_id)
        if cached is not None:
            metrics.incr("cache_hit", labels={"task": request.task.value})
            log_built(org_id=request.org_id, task=request.task.value, context_id=context_id,
                      admitted=cached.budget_report.admitted, dropped=cached.budget_report.dropped,
                      cache_hit=True)
            return cached, True

        spec = TASK_SPECS[request.task]
        scope = self.selector.select(state, spec)

        # Optional retrieval restriction (extension point; default: none).
        if self.retrieval is not None:
            allowed = self.retrieval.candidates(state, scope)
            if allowed is not None:
                scope = ContextScope(
                    scope.entity_types,
                    tuple(e for e in scope.entries if e.entity_id in allowed),
                )

        before = len(scope.entries)
        scope = self.permissions.resolve(request.actor, effective, scope, state)
        if before and not scope.entries:
            metrics.incr("permission_denied", labels={"task": request.task.value})

        ranked = self.ranker.rank(state, scope)
        admitted, dropped, report = self.compressor.compress(ranked, state, request.budget)

        admitted_ids = {e.entity_id for e in admitted}
        scoped_state = _scope_state(state, admitted_ids)
        final_scope = ContextScope(
            scope.entity_types, tuple(sorted(admitted, key=lambda e: e.entity_id))
        )
        evidence = _evidence(state, admitted)

        bundle = ContextBundle(
            context_id=context_id,
            context_version=CONTEXT_VERSION,
            schema_version=CONTEXT_SCHEMA_VERSION,
            org_id=request.org_id,
            task=request.task,
            actor_fingerprint=request.actor.fingerprint(),
            permission_fingerprint=effective.fingerprint(),
            policy_fingerprint=request.policy_fingerprint,
            state_as_of=state.as_of,
            state_watermark=state.watermark.to_dict(),
            state=scoped_state,
            scope=final_scope,
            evidence=evidence,
            budget_report=report,
            created_at=state.as_of,  # logical clock, never now()
        )

        if _should_cache(state, admitted_ids):
            await self.cache.put(bundle)
        metrics.incr("assembled", labels={"task": request.task.value})
        if dropped:
            metrics.incr("dropped_by_budget", value=float(len(dropped)),
                         labels={"task": request.task.value})
        log_built(org_id=request.org_id, task=request.task.value, context_id=context_id,
                  admitted=len(admitted), dropped=len(dropped), cache_hit=False)
        return bundle, False


def _should_cache(state: OrganizationalState, admitted_ids: set[str]) -> bool:
    """A bundle is worth persisting only if it is a STRICT SUBSET of the org — a full-org
    bundle embeds a whole copy of State, adding no value over State itself while doubling
    storage at scale. Scoped/budgeted bundles (the expensive, repeated ones) are cached."""
    all_ids = {eid for bucket in state.entities.values() for eid in bucket}
    return bool(all_ids) and len(admitted_ids) < len(all_ids)


def _evidence(state: OrganizationalState, admitted: list[ScopeEntry]) -> dict[str, dict[str, list]]:
    """Provenance (memory_ids, signal_ids) for each admitted entity — a read-only view over
    State's existing provenance. Citations are never dropped or summarized away."""
    out: dict[str, dict[str, list]] = {}
    for entry in admitted:
        raw = state.entities.get(entry.entity_type, {}).get(entry.entity_id)
        if raw is None:
            continue
        prov = raw.get("provenance", {}) or {}
        out[entry.entity_id] = {
            "memory_ids": list(prov.get("memory_ids", [])),
            "signal_ids": list(prov.get("signal_ids", [])),
        }
    return out


def _scope_state(state: OrganizationalState, admitted_ids: set[str]) -> OrganizationalState:
    """The INDEPENDENT scoped OrganizationalState embedded in the bundle.

    Always a deep copy (never an alias of the live projection) with links pruned to admitted
    entities — so a bundle can never mutate State and the 'immutable bundle' contract holds
    for every task, including FULL_ORG. The State's clock/version are preserved (filtering
    does not change which projection point this is). Trade-off: a one-time O(scoped) copy
    instead of a shared mutable reference — correctness over a micro-optimization; the copy
    happens only on a cache miss."""
    entities: dict[str, dict[str, dict]] = {}
    for et, bucket in state.entities.items():
        kept = {eid: copy.deepcopy(raw) for eid, raw in bucket.items() if eid in admitted_ids}
        if kept:
            entities[et] = kept

    for bucket in entities.values():
        for raw in bucket.values():
            links = raw.get("links") or {}
            for key in list(links.keys()):
                pruned = [t for t in links[key] if t in admitted_ids]
                if pruned:
                    links[key] = pruned
                else:
                    del links[key]

    return OrganizationalState(
        org_id=state.org_id,
        as_of=state.as_of,
        watermark=state.watermark,
        schema_version=state.schema_version,
        projection_version=state.projection_version,
        entities=entities,
    )
