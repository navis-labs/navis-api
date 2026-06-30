"""Public contracts for the Context layer: the request in, the immutable bundle out.

Determinism law (inherited from State and extended): a bundle is a pure function of
(OrganizationalState @ watermark, request, CONTEXT_VERSION). No datetime.now(), no
uuid4, no randomness — the logical clock is the State's `as_of`. Identical inputs
produce a byte-identical bundle and an identical `context_id`, forever, in any region.
"""
from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from navis.context.models import BudgetReport, ContextScope
from navis.context.tasks import ContextTask
from navis.state.v2.models import OrganizationalState

# Bump CONTEXT_VERSION when assembly LOGIC changes (forces cache miss + re-assembly);
# CONTEXT_SCHEMA_VERSION tracks the serialized bundle shape.
#
# CONTEXT_VERSION migration 1 -> 2: `_digest` switched from delimiter-joined to
# length-prefixed (injective) canonicalization, which changes every fingerprint and
# context_id. Migration is non-destructive and automatic: context_id folds CONTEXT_VERSION,
# so v1 ids can never collide with v2 ids, and the durable cache's version guard
# (repository.get) treats any v1 row as a MISS and rebuilds deterministically. No
# production context_ids exist yet (the layer ships behind a default-off flag), so there is
# nothing to backfill.
CONTEXT_VERSION = 2
CONTEXT_SCHEMA_VERSION = 1

# Fixed sentinel for the platform/system actor (no real user). Never collides with a
# real user id, and is stable across replays.
SYSTEM_ACTOR_ID = uuid.UUID("00000000-0000-0000-0000-000000000000")

# Free-text identity tokens (e.g. policy_fingerprint) must be bounded, printable, and
# control-char-free so they cannot smuggle structure into a digest or a log line.
_MAX_TOKEN_LEN = 256
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")


def _digest(*parts: str) -> str:
    """Deterministic, UNAMBIGUOUS digest.

    Each part is length-prefixed (netstring-style: ``<len>:<part>``) before hashing, so the
    concatenation is INJECTIVE — no choice of delimiter can ever be confused with content
    (the old "\\x1f".join could alias ("a\\x1fb","c") with ("a","b\\x1fc")). blake2b/16B is
    stable across Python versions, builds, and platforms (unlike the salted built-in hash)."""
    canonical = "".join(f"{len(p)}:{p}" for p in parts)
    return hashlib.blake2b(canonical.encode("utf-8"), digest_size=16).hexdigest()


def _validate_token(name: str, value: str) -> None:
    """Validate a free-text identity token (deterministic; raises ValueError)."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    if len(value) > _MAX_TOKEN_LEN:
        raise ValueError(f"{name} exceeds {_MAX_TOKEN_LEN} characters")
    if _CONTROL_CHARS.search(value):
        raise ValueError(f"{name} must not contain control characters")


@dataclass(frozen=True, slots=True)
class ActorRef:
    """Who the context is for. An actor is a user, an autonomous agent, or the system.
    The fingerprint folds identity + roles + attributes so a permission-relevant change
    re-keys the bundle (replay-safe RBAC/ABAC)."""

    actor_id: uuid.UUID
    kind: Literal["user", "agent", "system"]
    roles: tuple[str, ...] = ()
    attributes: tuple[tuple[str, str], ...] = ()  # sorted (key, value) pairs

    def __post_init__(self) -> None:
        if not isinstance(self.actor_id, uuid.UUID):
            raise ValueError("ActorRef.actor_id must be a uuid.UUID")
        if self.kind not in ("user", "agent", "system"):
            raise ValueError(f"ActorRef.kind must be user|agent|system, got {self.kind!r}")
        if not isinstance(self.roles, tuple) or not all(isinstance(r, str) for r in self.roles):
            raise ValueError("ActorRef.roles must be a tuple[str, ...]")
        if not isinstance(self.attributes, tuple) or not all(
            isinstance(a, tuple) and len(a) == 2 and isinstance(a[0], str) and isinstance(a[1], str)
            for a in self.attributes
        ):
            raise ValueError("ActorRef.attributes must be a tuple[tuple[str, str], ...]")

    @classmethod
    def system(cls) -> "ActorRef":
        return cls(actor_id=SYSTEM_ACTOR_ID, kind="system")

    def fingerprint(self) -> str:
        return _digest(
            str(self.actor_id), self.kind,
            ",".join(sorted(self.roles)),
            ",".join(f"{k}={v}" for k, v in sorted(self.attributes)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "actor_id": str(self.actor_id), "kind": self.kind,
            "roles": list(self.roles), "attributes": [list(a) for a in self.attributes],
        }


@dataclass(frozen=True, slots=True)
class PermissionScope:
    """The resolved RBAC/ABAC envelope for a build. Default-deny: an empty scope that is
    not `allow_all` admits nothing. ABAC predicates are conjunctive field==value checks."""

    allow_all: bool = False
    allowed_entity_types: tuple[str, ...] = ()
    attribute_predicates: tuple[tuple[str, str], ...] = ()  # (field, required_value)

    def __post_init__(self) -> None:
        # Fail closed at construction: an allow_all scope that ALSO constrains is
        # contradictory and a likely misconfiguration.
        if self.allow_all and (self.allowed_entity_types or self.attribute_predicates):
            raise ValueError("allow_all PermissionScope must not also constrain types/predicates")
        if not all(isinstance(t, str) for t in self.allowed_entity_types):
            raise ValueError("PermissionScope.allowed_entity_types must be a tuple[str, ...]")
        if not all(
            isinstance(p, tuple) and len(p) == 2 and isinstance(p[0], str) and isinstance(p[1], str)
            for p in self.attribute_predicates
        ):
            raise ValueError("PermissionScope.attribute_predicates must be a tuple[tuple[str, str], ...]")

    @classmethod
    def system(cls) -> "PermissionScope":
        return cls(allow_all=True)

    def fingerprint(self) -> str:
        return _digest(
            "all" if self.allow_all else "scoped",
            ",".join(sorted(self.allowed_entity_types)),
            ",".join(f"{k}={v}" for k, v in sorted(self.attribute_predicates)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "allow_all": self.allow_all,
            "allowed_entity_types": list(self.allowed_entity_types),
            "attribute_predicates": [list(p) for p in self.attribute_predicates],
        }


@dataclass(frozen=True, slots=True)
class TokenBudget:
    """The maximum token cost of a bundle. None = unbounded (whole scope admitted)."""

    max_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.max_tokens is not None:
            if not isinstance(self.max_tokens, int) or isinstance(self.max_tokens, bool):
                raise ValueError("TokenBudget.max_tokens must be an int or None")
            if self.max_tokens <= 0:
                raise ValueError(f"TokenBudget.max_tokens must be > 0 or None, got {self.max_tokens}")

    @classmethod
    def unbounded(cls) -> "TokenBudget":
        return cls(max_tokens=None)


@dataclass(frozen=True, slots=True)
class ContextRequest:
    """The reasoning task's information request — the sole input to ContextService.build.

    Every field that affects the assembled bundle is folded into context_id (see
    compute_context_id), so two requests that differ in ANY of these never share a cache
    entry. No 'pinned_watermark'/'goal' fields exist: Context reads the CURRENT State
    (historical pinning would require State point-in-time reconstruction, which State does
    not provide), and there is no goal-driven selection — exposing either would be a fake
    capability."""

    org_id: uuid.UUID
    task: ContextTask
    actor: ActorRef
    permissions: PermissionScope
    policy_fingerprint: str = "none"
    budget: TokenBudget = field(default_factory=TokenBudget)

    def __post_init__(self) -> None:
        if not isinstance(self.org_id, uuid.UUID):
            raise ValueError("ContextRequest.org_id must be a uuid.UUID")
        if not isinstance(self.task, ContextTask):
            raise ValueError("ContextRequest.task must be a ContextTask")
        if not isinstance(self.actor, ActorRef):
            raise ValueError("ContextRequest.actor must be an ActorRef")
        if not isinstance(self.permissions, PermissionScope):
            raise ValueError("ContextRequest.permissions must be a PermissionScope")
        _validate_token("ContextRequest.policy_fingerprint", self.policy_fingerprint)

    @classmethod
    def full_org(cls, org_id: uuid.UUID, *, actor: ActorRef | None = None) -> "ContextRequest":
        """The whole-org, system-actor, unbounded request — exact parity with the
        pre-Context reasoning input."""
        return cls(
            org_id=org_id, task=ContextTask.FULL_ORG,
            actor=actor or ActorRef.system(), permissions=PermissionScope.system(),
        )


def compute_context_id(
    request: ContextRequest, state: OrganizationalState, *, context_version: int = CONTEXT_VERSION
) -> str:
    """Content-addressed bundle id. Folds the State version (watermark), the task, and the
    actor/permission/policy fingerprints, so any of them changing yields a new id — and an
    identical request over the identical State always reproduces the same id."""
    wm = state.watermark
    return _digest(
        str(request.org_id),
        wm.last_timestamp.isoformat() if wm.last_timestamp else "",
        str(wm.last_memory_id) if wm.last_memory_id else "",
        request.task.value,
        request.actor.fingerprint(),
        request.permissions.fingerprint(),
        request.policy_fingerprint,
        # Budget is part of the request semantics: a 10-token bundle and an unbounded
        # bundle for the same (org, state, task, actor) are DIFFERENT artefacts and must
        # never collide in the cache.
        str(request.budget.max_tokens) if request.budget.max_tokens is not None else "unbounded",
        str(context_version),
    )


@dataclass(frozen=True, slots=True)
class ContextBundle:
    """The immutable boundary object: a permission-filtered, budgeted, content-addressed
    view of State for one (task, actor). Embeds a SCOPED OrganizationalState so Reasoning
    consumes it unchanged via its existing accessors. Disposable and replayable."""

    context_id: str
    context_version: int
    schema_version: int
    org_id: uuid.UUID
    task: ContextTask
    actor_fingerprint: str
    permission_fingerprint: str
    policy_fingerprint: str
    state_as_of: datetime | None
    state_watermark: dict[str, Any]
    state: OrganizationalState        # scoped subset of State (same type)
    scope: ContextScope
    evidence: dict[str, dict[str, list]]   # entity_id -> {memory_ids, signal_ids}
    budget_report: BudgetReport
    created_at: datetime | None       # logical clock = state_as_of

    def __post_init__(self) -> None:
        # Validate on EVERY construction — including from_dict — so a structurally-invalid
        # cached row raises here and the cache's corruption guard (repository.get) rebuilds.
        self.validate()

    def validate(self) -> None:
        """Deterministic invariant check (no clock/IO). Run before serialization so an
        inconsistent bundle can never be cached or handed to Reasoning."""
        if not isinstance(self.context_id, str) or not self.context_id:
            raise ValueError("ContextBundle.context_id must be a non-empty string")
        if self.context_version < 1 or self.schema_version < 1:
            raise ValueError("ContextBundle versions must be >= 1")
        if not isinstance(self.org_id, uuid.UUID):
            raise ValueError("ContextBundle.org_id must be a uuid.UUID")
        if self.state.org_id != self.org_id:
            raise ValueError("ContextBundle.state is scoped to a different org than the bundle")
        if self.created_at != self.state_as_of:
            raise ValueError("ContextBundle.created_at must equal state_as_of (logical clock)")
        if self.budget_report.admitted != len(self.scope.entries):
            raise ValueError("budget_report.admitted must equal the number of scope entries")
        scope_ids = set(self.scope.entity_ids())
        if not set(self.evidence).issubset(scope_ids):
            raise ValueError("evidence references entities outside the scope")

    def to_dict(self) -> dict[str, Any]:
        # Deterministic: State sorts itself; scope/evidence already ordered. (Invariants are
        # already enforced at construction via __post_init__.)
        return {
            "context_id": self.context_id,
            "context_version": self.context_version,
            "schema_version": self.schema_version,
            "org_id": str(self.org_id),
            "task": self.task.value,
            "actor_fingerprint": self.actor_fingerprint,
            "permission_fingerprint": self.permission_fingerprint,
            "policy_fingerprint": self.policy_fingerprint,
            "state_as_of": self.state_as_of.isoformat() if self.state_as_of else None,
            "state_watermark": self.state_watermark,
            "state": self.state.to_dict(),
            "scope": self.scope.to_dict(),
            "evidence": {k: self.evidence[k] for k in sorted(self.evidence)},
            "budget_report": self.budget_report.to_dict(),
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ContextBundle":
        as_of = d.get("state_as_of")
        created = d.get("created_at")
        return cls(
            context_id=d["context_id"],
            context_version=int(d.get("context_version", CONTEXT_VERSION)),
            schema_version=int(d.get("schema_version", CONTEXT_SCHEMA_VERSION)),
            org_id=uuid.UUID(d["org_id"]),
            task=ContextTask(d["task"]),
            actor_fingerprint=d["actor_fingerprint"],
            permission_fingerprint=d["permission_fingerprint"],
            policy_fingerprint=d.get("policy_fingerprint", "none"),
            state_as_of=datetime.fromisoformat(as_of) if as_of else None,
            state_watermark=d.get("state_watermark", {}),
            state=OrganizationalState.from_dict(d["state"]),
            scope=ContextScope.from_dict(d.get("scope", {})),
            evidence=d.get("evidence", {}),
            budget_report=BudgetReport.from_dict(d["budget_report"]),
            created_at=datetime.fromisoformat(created) if created else None,
        )
