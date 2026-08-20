"""Delegated administration: invite / list / remove members, assign roles.

Every mutation is audited. Removing a member bulk-revokes their access tokens.
"""
from __future__ import annotations

import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from navis.identity.audit import AuditAction, AuditService
from navis.identity.authz import grants_for
from navis.identity.models import Membership, Role, User
from navis.identity.permissions import DEFAULT_ROLE_PERMISSIONS, effective_permissions
from navis.identity.tokens import TokenRevocationService


def normalize_email(email: str) -> str:
    """Identity keys must be canonical — `John@X.com` and `john@x.com` are ONE user.
    Normalizing on every lookup/create prevents duplicate-identity rows."""
    return email.strip().lower()


class MemberError(Exception):
    pass


class MemberForbidden(MemberError):
    """Actor tried to grant more privilege than they hold (escalation). -> 403."""


class MemberConflict(MemberError):
    """Operation would leave the org without a founder (governance). -> 409."""


class MemberService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db
        self.audit = AuditService(db)

    async def _count_founders_locked(self, org_id: uuid.UUID) -> int:
        """Lock the org's FOUNDER rows, then count — closes the last-founder TOCTOU race.

        Two concurrent removals of different founders would each read count=2 and both
        proceed, leaving zero founders. `SELECT ... FOR UPDATE` serializes them: the
        second transaction blocks until the first commits, then re-reads the smaller
        set and correctly trips the guard. (Postgres forbids FOR UPDATE with aggregate
        count(), so we lock the id column and len() it — founders per org are few.)
        """
        rows = (
            await self.db.scalars(
                select(Membership.id)
                .where(
                    Membership.organization_id == org_id, Membership.role == Role.FOUNDER
                )
                .with_for_update()
            )
        ).all()
        return len(rows)

    # ── Privilege comparison ────────────────────────────────────────────────
    #
    # THERE IS NO ROLE HIERARCHY IN THIS SYSTEM, and one must not be invented.
    # `Role` is an unordered enum (its declaration order puts ADMIN last), and the
    # permission sets form a PARTIAL order, not a chain: `admin` holds
    # MANAGE_ROLES / MANAGE_API_KEYS / MANAGE_SERVICE_ACCOUNTS / MANAGE_SSO that
    # `ceo` lacks, while `ceo` holds EXECUTE_ACTIONS / MANAGE_POLICIES that `admin`
    # lacks. Neither outranks the other.
    #
    # The one ordering the codebase already defines is permission-set containment,
    # which `_ensure_grantable` has always used for the role being GRANTED. The
    # whole of this task is applying that same rule to the member being ACTED ON:
    #
    #     an actor may act on a target only if the actor holds every
    #     permission the target holds
    #
    # Reflexive by construction, so peers can administer each other — unchanged
    # from the existing grant rule, and not an escalation: neither party gains a
    # permission they did not already have.

    async def _effective_permissions(self, org_id: uuid.UUID, role: Role | str | None) -> frozenset[str]:
        """A role's effective permissions IN THIS ORG — defaults plus org grants.

        Resolved through the SAME `grants_for` the Principal was built with, so an
        actor and a target are never compared across two different notions of
        "effective". Defaults alone would under-count a role that an org has
        granted extra permissions to, which is precisely the case where a
        comparison must be strictest.

        An unrecognised role DENIES rather than resolving to the empty set. A role
        string this build does not know (a value added by migration ahead of the
        enum, or hand-edited data) would otherwise look like zero privilege and be
        freely removable — a fail-open on exactly the rows least understood.
        """
        value = role.value if isinstance(role, Role) else (role or "")
        if value not in DEFAULT_ROLE_PERMISSIONS:
            raise MemberForbidden(
                "cannot evaluate the privilege of this membership; refusing the operation"
            )
        grants = await grants_for(self.db, org_id, value)
        return effective_permissions(value, grants)

    async def _ensure_grantable(
        self, org_id: uuid.UUID, role: Role, actor_permissions: frozenset[str]
    ) -> None:
        """A member may only grant a role whose permissions they themselves hold —
        prevents an admin (MANAGE_ROLES) from minting a FOUNDER (superuser).

        Now resolved with org grants rather than code defaults. With defaults only,
        an org that granted an extra permission to a weak role (a `role_permissions`
        row) let a lesser actor hand out that permission by assigning the role —
        escalation by proxy, since the actor never has to hold it themselves.
        """
        target = await self._effective_permissions(org_id, role)
        if not target <= actor_permissions:
            raise MemberForbidden(
                f"cannot grant role '{role.value}': it exceeds your own permissions"
            )

    async def _ensure_may_manage(
        self, org_id: uuid.UUID, membership: Membership, actor_permissions: frozenset[str]
    ) -> None:
        """Deny unless the actor holds every permission the TARGET holds.

        The gap this closes: authorization considered only the ACTOR's permission
        and, for role changes, the NEW role — never the target's CURRENT standing.
        So `admin` (MANAGE_MEMBERS + MANAGE_ROLES, deliberately no EXECUTE_ACTIONS)
        could remove a `ceo`, or demote one to `employee`, because
        `default_permissions_for_role("employee")` is a subset of admin's own.
        Delegated IT administration could dismantle business leadership.

        Target state is read from the PERSISTED membership row — never from the
        request — and the row was already selected under the actor's trusted
        `org_id`, so a cross-tenant target cannot reach this check at all.
        """
        target = await self._effective_permissions(org_id, membership.role)
        if not target <= actor_permissions:
            raise MemberForbidden(
                "cannot modify a member whose permissions exceed your own"
            )

    async def invite(
        self, *, org_id: uuid.UUID, email: str, role: Role, actor_id: uuid.UUID,
        actor_permissions: frozenset[str] = frozenset(),
    ) -> Membership:
        await self._ensure_grantable(org_id, role, actor_permissions)
        email = normalize_email(email)
        user = await self.db.scalar(select(User).where(User.email == email))
        if user is None:
            user = User(email=email, is_verified=False, is_active=True)  # set password on first login/SSO
            self.db.add(user)
            await self.db.flush()
        existing = await self.db.scalar(
            select(Membership).where(
                Membership.user_id == user.id, Membership.organization_id == org_id
            )
        )
        if existing is not None:
            raise MemberError("user is already a member")
        membership = Membership(user_id=user.id, organization_id=org_id, role=role)
        self.db.add(membership)
        await self.db.flush()
        await self.audit.record(
            AuditAction.MEMBER_INVITE, org_id=org_id, actor_id=actor_id,
            target_id=str(user.id), metadata={"email": email, "role": role.value},
        )
        return membership

    async def remove(
        self, *, org_id: uuid.UUID, user_id: uuid.UUID, actor_id: uuid.UUID,
        actor_permissions: frozenset[str] = frozenset(),
    ) -> None:
        """Remove a member. `actor_permissions` defaults to EMPTY, which denies any
        target holding a single permission — a caller that forgets to pass it fails
        CLOSED rather than inheriting the old unrestricted behaviour."""
        # Cross-org safety: the row is selected under the actor's TRUSTED org_id
        # (Principal.org_id), never an organization named by the request. A target
        # in another tenant simply does not match, and surfaces as "not a member".
        membership = await self.db.scalar(
            select(Membership).where(
                Membership.user_id == user_id, Membership.organization_id == org_id
            )
        )
        if membership is None:
            raise MemberError("not a member")
        # Target-aware authorization, BEFORE the governance guard: whether the org
        # would lose its last founder is a different question from whether this
        # actor may touch this member at all, and the second must be settled first.
        await self._ensure_may_manage(org_id, membership, actor_permissions)
        # Governance: never let an org lose its last founder (would be ungovernable).
        if membership.role == Role.FOUNDER and await self._count_founders_locked(org_id) <= 1:
            raise MemberConflict("cannot remove the last founder of the organization")
        await self.db.delete(membership)
        # Membership removal revokes the user's active access immediately.
        await TokenRevocationService(self.db).revoke_all_for_user(user_id)
        await self.audit.record(
            AuditAction.MEMBER_REMOVE, org_id=org_id, actor_id=actor_id, target_id=str(user_id),
        )

    async def assign_role(
        self, *, org_id: uuid.UUID, user_id: uuid.UUID, role: Role, actor_id: uuid.UUID,
        actor_permissions: frozenset[str] = frozenset(),
    ) -> Membership:
        # Escalation guard: cannot grant a role beyond your own permissions.
        await self._ensure_grantable(org_id, role, actor_permissions)
        # Cross-org safety: selected under the actor's trusted org_id (see `remove`).
        membership = await self.db.scalar(
            select(Membership).where(
                Membership.user_id == user_id, Membership.organization_id == org_id
            )
        )
        if membership is None:
            raise MemberError("not a member")
        # BOTH SIDES OF A ROLE CHANGE ARE PRIVILEGED OPERATIONS.
        #
        # Checking only the new role made DEMOTION a bypass: `employee` is a subset
        # of `admin`, so an admin could assign it to a `ceo` and strip a superior of
        # EXECUTE_ACTIONS and MANAGE_POLICIES. Granting is guarded above; being
        # allowed to act on THIS member is guarded here.
        await self._ensure_may_manage(org_id, membership, actor_permissions)
        # Governance: cannot demote the last founder away from FOUNDER.
        if (
            membership.role == Role.FOUNDER
            and role != Role.FOUNDER
            and await self._count_founders_locked(org_id) <= 1
        ):
            raise MemberConflict("cannot demote the last founder of the organization")
        old = membership.role.value if membership.role else None
        membership.role = role
        await self.db.flush()
        await self.audit.record(
            AuditAction.ROLE_CHANGE, org_id=org_id, actor_id=actor_id, target_id=str(user_id),
            metadata={"from": old, "to": role.value},
        )
        return membership

    async def list_members(
        self, org_id: uuid.UUID, *, limit: int = 50, offset: int = 0
    ) -> list[Membership]:
        return list(
            (
                await self.db.scalars(
                    select(Membership)
                    .where(Membership.organization_id == org_id)
                    .order_by(Membership.created_at.desc(), Membership.id)
                    .limit(limit)
                    .offset(offset)
                )
            ).all()
        )

    async def find_member_by_email(
        self, org_id: uuid.UUID, email: str
    ) -> tuple[Membership, User] | None:
        """Resolve a single member by (normalized) email — backs SCIM `userName eq`."""
        row = (
            await self.db.execute(
                select(Membership, User)
                .join(User, Membership.user_id == User.id)
                .where(
                    Membership.organization_id == org_id,
                    User.email == normalize_email(email),
                )
            )
        ).first()
        return (row[0], row[1]) if row else None

    async def count_members(self, org_id: uuid.UUID) -> int:
        return int(
            await self.db.scalar(
                select(func.count()).select_from(Membership).where(
                    Membership.organization_id == org_id
                )
            )
            or 0
        )

    async def list_members_with_users(
        self, org_id: uuid.UUID, *, limit: int = 50, offset: int = 0
    ) -> list[tuple[Membership, User]]:
        """One JOIN instead of N+1 — for SCIM /Users which needs the full user row."""
        rows = (
            await self.db.execute(
                select(Membership, User)
                .join(User, Membership.user_id == User.id)
                .where(Membership.organization_id == org_id)
                .order_by(Membership.created_at.desc(), Membership.id)
                .limit(limit)
                .offset(offset)
            )
        ).all()
        return [(m, u) for m, u in rows]
