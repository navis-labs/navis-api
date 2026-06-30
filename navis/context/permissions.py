"""Default-deny RBAC/ABAC filtering over a scope.

Runs BEFORE ranking/budgeting so nothing an actor may not see can ever enter a bundle
(and therefore never reaches Reasoning or an LLM). Pure and deterministic; the permission
fingerprint is folded into context_id so a grant/revoke invalidates cached bundles.
"""
from __future__ import annotations

from navis.context.contracts import ActorRef, PermissionScope
from navis.context.exceptions import PermissionConfigurationError
from navis.context.models import ContextScope, ScopeEntry
from navis.memory.envelope import EntityType
from navis.state.v2.models import OrganizationalState

# --- RBAC policy: identity-derived role -> entity types it may read --------------------
# Declarative and extensible. Default-DENY: a role absent here grants nothing. Roles arrive
# on ActorRef from the authenticated Identity session (claims-based authz) — Context never
# re-authenticates, it maps trusted claims to a scope deterministically (replay-safe).
_ALL_TYPES = frozenset(e.value for e in EntityType)
ROLE_POLICY: dict[str, frozenset[str]] = {
    "owner": _ALL_TYPES,
    "admin": _ALL_TYPES,
    "member": _ALL_TYPES,
    "viewer": _ALL_TYPES,
    # Example of a genuinely restricted role (bounds even a caller that asks for more):
    "guest": frozenset({EntityType.PERSON.value, EntityType.COMPANY.value}),
}


class RolePermissionResolver:
    """Resolves an actor's identity roles into the PermissionScope they are ENTITLED to.

    This is the authority Context uses to bound a request — a caller's requested scope is
    intersected with what the actor's roles allow, so a caller can only ever NARROW, never
    widen, their access. The system actor is unrestricted (it runs trusted pipelines like
    the daily brief). Deterministic ⇒ replay-safe."""

    def __init__(self, policy: dict[str, frozenset[str]] | None = None) -> None:
        self.policy = policy if policy is not None else ROLE_POLICY

    def resolve(self, actor: ActorRef) -> PermissionScope:
        if actor.kind == "system":
            return PermissionScope.system()
        allowed: set[str] = set()
        for role in actor.roles:
            allowed |= self.policy.get(role, frozenset())
        return PermissionScope(allow_all=False, allowed_entity_types=tuple(sorted(allowed)))


def intersect_scopes(requested: PermissionScope, granted: PermissionScope) -> PermissionScope:
    """The EFFECTIVE permission scope: a request may only narrow what an actor's roles grant.

    - granted.allow_all (system): the caller's requested scope is honoured as-is.
    - otherwise: allowed types = requested ∩ granted; ABAC predicates from both apply
      (conjunctive), so neither side can relax the other's constraints."""
    if granted.allow_all:
        return requested
    if requested.allow_all:
        return granted  # caller asked for everything but is bounded by the role grant
    allowed = tuple(sorted(set(requested.allowed_entity_types) & set(granted.allowed_entity_types)))
    predicates = tuple(sorted(set(requested.attribute_predicates) | set(granted.attribute_predicates)))
    return PermissionScope(allow_all=False, allowed_entity_types=allowed, attribute_predicates=predicates)


class DefaultDenyPermissionFilter:
    """The default PermissionResolver. allow_all passes everything; otherwise an entity
    survives only if its type is allowed AND it satisfies every ABAC predicate."""

    def resolve(
        self,
        actor: ActorRef,
        permissions: PermissionScope,
        scope: ContextScope,
        state: OrganizationalState,
    ) -> ContextScope:
        if permissions.allow_all:
            if permissions.allowed_entity_types or permissions.attribute_predicates:
                raise PermissionConfigurationError(
                    "allow_all must not also constrain types/predicates"
                )
            return scope

        allowed_types = set(permissions.allowed_entity_types)
        predicates = permissions.attribute_predicates
        kept: list[ScopeEntry] = []
        for entry in scope.entries:
            if entry.entity_type not in allowed_types:
                continue  # default-deny by type
            if predicates and not self._matches(state, entry, predicates):
                continue  # default-deny by attribute
            kept.append(entry)

        return ContextScope(entity_types=scope.entity_types, entries=tuple(kept))

    @staticmethod
    def _matches(
        state: OrganizationalState, entry: ScopeEntry, predicates: tuple[tuple[str, str], ...]
    ) -> bool:
        raw = state.entities.get(entry.entity_type, {}).get(entry.entity_id)
        if raw is None:
            return False
        fields = raw.get("fields", {})
        return all(str(fields.get(k)) == v for k, v in predicates)
