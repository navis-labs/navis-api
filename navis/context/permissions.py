"""Authorization for the Context layer — RBAC + ABAC, default-deny, fail-closed.

SECURITY CONTRACTS (an authorization failure is a CRITICAL incident):
  * Default-deny  — an entity enters a bundle ONLY if explicitly permitted.
  * Least privilege — a request can only NARROW the actor's role grant, never widen.
  * Fail-closed   — a missing/unverifiable attribute DENIES; nothing falls through to allow.
  * Deterministic — same (actor, request, State) -> same decision, forever (replay-safe).

TENANT-BINDING INVARIANT (enforced upstream; relied on here):
  ActorRef.roles MUST be the actor's roles FOR `request.org_id`, resolved at the
  authenticated edge (claims-based authz). Context never re-authenticates and never reads
  Identity. Cross-tenant DATA leakage is nonetheless structurally impossible: State is
  loaded per-org (StateService.get_state(org_id)) and entity_ids encode the org, so a bundle
  can only ever contain the requested org's entities — regardless of the actor. The ONLY
  thing the caller must guarantee is that the actor is entitled to `request.org_id` at all.

POLICY / REPLAY:
  ROLE_POLICY is CODE (a default mapping). Any change to it changes assembly logic and MUST
  bump CONTEXT_VERSION, which invalidates cached bundles. For a dynamically INJECTED policy,
  fold `RolePermissionResolver.fingerprint()` into the cache key so a policy change
  invalidates stale bundles (the seam exists; wiring it is the caller's responsibility).
"""
from __future__ import annotations

import hashlib

from navis.context import metrics
from navis.context.contracts import ActorRef, PermissionScope
from navis.context.exceptions import PermissionConfigurationError
from navis.context.models import ContextScope, ScopeEntry
from navis.memory.envelope import EntityType
from navis.platform.monitoring import get_logger
from navis.state.v2.models import OrganizationalState

logger = get_logger("context.permissions")

_ALL_TYPES = frozenset(e.value for e in EntityType)

# DEFAULT RBAC policy (role -> readable entity types). This is a permissive PLACEHOLDER:
# owner/admin/member/viewer currently map to ALL types, which does NOT honour least
# privilege. Production MUST inject the authoritative per-org policy (RolePermissionResolver
# accepts one) before non-system actors are served. Unknown roles grant NOTHING.
ROLE_POLICY: dict[str, frozenset[str]] = {
    "owner": _ALL_TYPES,
    "admin": _ALL_TYPES,
    "member": _ALL_TYPES,
    "viewer": _ALL_TYPES,
    "guest": frozenset({EntityType.PERSON.value, EntityType.COMPANY.value}),
}


def _validate_policy(policy: dict[str, frozenset[str]]) -> None:
    """Fail-fast: every type a role grants must be a real EntityType. A typo would otherwise
    silently grant nothing (a confusing empty scope) — caught here at construction instead."""
    for role, types in policy.items():
        unknown = sorted(t for t in types if t not in _ALL_TYPES)
        if unknown:
            raise PermissionConfigurationError(
                f"ROLE_POLICY role {role!r} grants unknown entity type(s): {unknown}"
            )


_validate_policy(ROLE_POLICY)


class RolePermissionResolver:
    """Resolves an actor's identity roles into the PermissionScope they are ENTITLED to.

    Roles are ADDITIVE (RBAC union): a multi-role actor receives the union of their roles'
    grants — standard RBAC, and it can never exceed what Identity assigned. The result only
    GRANTS; the request then NARROWS it (see intersect_scopes), so union cannot escalate
    beyond assigned roles. The system actor is unrestricted (trusted pipelines). Unknown
    roles contribute nothing (default-deny). Deterministic ⇒ replay-safe.
    """

    def __init__(self, policy: dict[str, frozenset[str]] | None = None) -> None:
        # `is_bootstrap` is True when no policy was injected → the permissive in-code default
        # is in use. Production deployments MUST inject a least-privilege policy provider; the
        # default exists only to bootstrap dev/test and the system pipeline.
        self.is_bootstrap = policy is None
        self.policy = policy if policy is not None else ROLE_POLICY
        if policy is not None:
            _validate_policy(self.policy)

    def resolve(self, actor: ActorRef) -> PermissionScope:
        if actor.kind == "system":
            return PermissionScope.system()
        if self.is_bootstrap:
            # A REAL actor is being authorized by the permissive bootstrap policy — surface it
            # loudly (metric + warning) so production can never silently run without an
            # injected least-privilege policy. Does not change the decision (observability only).
            metrics.incr("permission.bootstrap_policy_used")
            logger.warning(
                "context.permissions: authorizing a non-system actor with the BOOTSTRAP "
                "ROLE_POLICY (kind=%s roles=%s) — inject a production policy provider",
                actor.kind, actor.roles,
            )
        allowed: set[str] = set()
        for role in actor.roles:
            allowed |= self.policy.get(role, frozenset())  # default-deny on unknown role
        return PermissionScope(allow_all=False, allowed_entity_types=tuple(sorted(allowed)))

    def fingerprint(self) -> str:
        """Deterministic digest of the active policy. Fold into the cache key when the policy
        can vary at runtime, so a policy change invalidates stale bundles (replay-safe)."""
        canonical = ";".join(
            f"{role}:{','.join(sorted(types))}" for role, types in sorted(self.policy.items())
        )
        return hashlib.blake2b(canonical.encode("utf-8"), digest_size=16).hexdigest()


def intersect_scopes(requested: PermissionScope, granted: PermissionScope) -> PermissionScope:
    """The EFFECTIVE scope: a request may only NARROW what the actor's roles grant.

    Monotonic narrowing (never widens):
      * granted.allow_all (system): honour the caller's requested scope as-is.
      * requested.allow_all but bounded grant: return the grant (caller bounded by role) — no
        privilege escalation.
      * else: allowed types = requested ∩ granted; ABAC predicates = requested ∪ granted
        (conjunctive in _matches, so MORE predicates = MORE restrictive). Both operations
        SHRINK the scope; neither side can relax the other's constraints.
    Sorted ⇒ deterministic ⇒ stable fingerprint ⇒ correct cache keying.
    """
    if granted.allow_all:
        return requested
    if requested.allow_all:
        return granted
    allowed = tuple(sorted(set(requested.allowed_entity_types) & set(granted.allowed_entity_types)))
    predicates = tuple(sorted(set(requested.attribute_predicates) | set(granted.attribute_predicates)))
    return PermissionScope(allow_all=False, allowed_entity_types=allowed, attribute_predicates=predicates)


class DefaultDenyPermissionFilter:
    """Applies an effective PermissionScope to a candidate scope BEFORE ranking/budgeting.
    Default-deny + fail-closed: an entry survives only if its type is allowed AND it
    satisfies every ABAC predicate; anything unverifiable is dropped."""

    def resolve(
        self,
        actor: ActorRef,
        permissions: PermissionScope,
        scope: ContextScope,
        state: OrganizationalState,
    ) -> ContextScope:
        if permissions.allow_all:
            # Only the system actor reaches here with allow_all (intersect_scopes bounds every
            # other actor). A constrained allow_all is contradictory → fail closed. (Redundant
            # with PermissionScope.__post_init__; kept as defense-in-depth for an authz layer.)
            if permissions.allowed_entity_types or permissions.attribute_predicates:
                metrics.incr("permission.config_error")
                raise PermissionConfigurationError(
                    "allow_all PermissionScope must not also constrain types/predicates"
                )
            return scope

        allowed_types = frozenset(permissions.allowed_entity_types)
        predicates = permissions.attribute_predicates
        entities = state.entities
        kept: list[ScopeEntry] = []
        rbac_denied = abac_denied = unbacked = 0
        for entry in scope.entries:
            if entry.entity_type not in allowed_types:
                rbac_denied += 1
                continue
            # TENANT/INTEGRITY GUARD (fail-closed): authorize ONLY entities backed by THIS
            # org's State. `state` is loaded per-org (entity_ids encode the org), so an id not
            # present here is foreign/injected and must never enter a bundle — cross-tenant
            # leakage is therefore impossible even if a scope were poisoned upstream.
            raw = entities.get(entry.entity_type, {}).get(entry.entity_id)
            if raw is None:
                unbacked += 1
                continue
            if predicates and not _matches_fields(raw, predicates):
                abac_denied += 1
                continue
            kept.append(entry)

        # Security observability (counts only — never entity content): denial rates surface
        # misconfiguration, probing, and integrity violations. Do not affect the returned scope.
        if rbac_denied:
            metrics.incr("permission.rbac_denied", value=float(rbac_denied))
        if abac_denied:
            metrics.incr("permission.abac_denied", value=float(abac_denied))
        if unbacked:
            metrics.incr("permission.unbacked_denied", value=float(unbacked))
        return ContextScope(entity_types=scope.entity_types, entries=tuple(kept))


def _matches_fields(raw: dict, predicates: tuple[tuple[str, str], ...]) -> bool:
    """ABAC predicate evaluation. Semantics are EXACT STRING EQUALITY against the entity's
    PRESENT field value — equality-only by design (NO operators, ranges, expressions, or
    runtime policy language). Edge behaviour, all FAIL-CLOSED:
      * missing field          -> deny (an absent attribute can never satisfy a predicate;
                                   the old `str(fields.get(k)) == v` let it match literal "None")
      * field value None       -> str(None) == "None": matches ONLY the literal predicate "None"
      * empty-string field/pred-> compared exactly ("" == "")
      * nested object/list     -> str(obj) compared; never equals a scalar predicate -> deny
      * type coercion          -> str(value) == required (e.g. True->"True", 1->"1"), documented
    `raw` is guaranteed non-None by the caller (it already verified State backing)."""
    fields = raw.get("fields", {})
    for key, required in predicates:
        if key not in fields:
            return False
        if str(fields[key]) != required:
            return False
    return True
