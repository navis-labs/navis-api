"""Delegated administration: invite / list / remove members, assign roles.

Every mutation is audited. Removing a member bulk-revokes their access tokens.
"""
from __future__ import annotations

import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from navis.identity.audit import AuditAction, AuditService
from navis.identity.authz import grants_for_roles
from navis.identity.models import (
    DeveloperRole,
    Membership,
    Organization,
    OrganizationKind,
    Role,
    User,
)
from navis.identity.permissions import (
    DEFAULT_ROLE_PERMISSIONS,
    DEVELOPER_ROLE_PERMISSIONS,
    developer_grant_key,
    effective_permissions,
)
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
    """Operation would leave the org without a founder, or without its last
    Developer owner (governance). -> 409."""


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

    async def _count_developer_owners_locked(self, org_id: uuid.UUID) -> int:
        """The same last-founder governance rule, for the Developer surface (P1-6).

        WHY THE DEVELOPER SURFACE NEEDS ITS OWN. Assigning a Developer role is
        gated on MANAGE_ROLES and bounded by `_ensure_grantable`, so only
        somebody who already holds Developer authority can hand it out. An
        organization that lost its last Developer owner would therefore have
        nobody left who could grant Developer access to anyone — including
        themselves — and the surface would be permanently unreachable for that
        tenant with no in-product recovery. That is the same unrecoverable state
        `_count_founders_locked` exists to prevent, so it gets the same guard
        rather than a new kind of one.

        ONLY BINDS ONCE A TENANT HAS AN OWNER. An organization with no Developer
        owner at all is a legitimate state — an Application-only tenant — and
        this count simply reads zero there, so nothing is blocked.

        Same `FOR UPDATE` on the id column as the founder count, for the same
        TOCTOU reason and with the same Postgres constraint (no FOR UPDATE with
        an aggregate).
        """
        rows = (
            await self.db.scalars(
                select(Membership.id)
                .where(
                    Membership.organization_id == org_id,
                    Membership.developer_role == DeveloperRole.OWNER.value,
                )
                .with_for_update()
            )
        ).all()
        return len(rows)

    async def _actor_is_developer_owner(
        self, org_id: uuid.UUID, actor_id: uuid.UUID
    ) -> bool:
        """Does the ACTOR hold Developer owner in THIS organization?

        Read from the actor's persisted membership row rather than from the
        Principal's permission set, because owner and admin have identical
        permissions by design — the set cannot tell them apart, and that is
        exactly why succession needs the role value.

        Scoped by `org_id`, which is the actor's own trusted tenant, so a
        Developer owner seat in a DIFFERENT organization confers nothing here.
        Absent membership reads False, which fails closed.

        Selects the ROW and reads the column off it, rather than selecting the
        column — every other membership lookup in this service does the same,
        and a bare-column select is the one shape the in-memory session the
        identity tests run on does not model.
        """
        membership = await self.db.scalar(
            select(Membership).where(
                Membership.user_id == actor_id,
                Membership.organization_id == org_id,
            )
        )
        return (
            membership is not None
            and membership.developer_role == DeveloperRole.OWNER.value
        )

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

    async def _effective_permissions(
        self,
        org_id: uuid.UUID,
        role: Role | str | None,
        developer_role: DeveloperRole | str | None = None,
    ) -> frozenset[str]:
        """A membership's effective permissions IN THIS ORG — defaults plus org grants.

        Resolved through the SAME `grants_for_roles` and the SAME
        `effective_permissions` the Principal was built with, so an actor and a
        target are never compared across two different notions of "effective".
        Defaults alone would under-count a role that an org has granted extra
        permissions to, which is precisely the case where a comparison must be
        strictest.

        BOTH SURFACES, SINCE P1-6. A membership's privilege is its Application
        role UNION its Developer role, because that is the set
        `require_permission` will check. Comparing only the Application half
        would let an Application `admin` remove or re-role a Developer owner —
        the target's Developer authority would be invisible to the guard, which
        is the same class of bug as the demotion bypass `_ensure_may_manage` was
        written to close.

        An unrecognised APPLICATION role DENIES rather than resolving to the
        empty set. A role string this build does not know (a value added by
        migration ahead of the enum, or hand-edited data) would otherwise look
        like zero privilege and be freely removable — a fail-open on exactly the
        rows least understood.

        An unrecognised DEVELOPER role denies for the same reason. NULL is NOT
        unrecognised: it is the defined value for "no Developer access" and
        contributes the empty set, which is what an Application-only member is.
        """
        value = role.value if isinstance(role, Role) else (role or "")
        if value not in DEFAULT_ROLE_PERMISSIONS:
            raise MemberForbidden(
                "cannot evaluate the privilege of this membership; refusing the operation"
            )
        dev = (
            developer_role.value
            if isinstance(developer_role, DeveloperRole)
            else developer_role
        )
        if dev is not None and dev not in DEVELOPER_ROLE_PERMISSIONS:
            raise MemberForbidden(
                "cannot evaluate the Developer privilege of this membership; "
                "refusing the operation"
            )
        dev_key = developer_grant_key(dev)
        grants = await grants_for_roles(self.db, org_id, [value, dev_key])
        return effective_permissions(
            value,
            grants[value],
            developer_role=dev,
            developer_grants=grants[dev_key],
        )

    async def _ensure_grantable(
        self,
        org_id: uuid.UUID,
        role: Role,
        actor_permissions: frozenset[str],
        developer_role: DeveloperRole | str | None = None,
    ) -> None:
        """A member may only grant an assignment whose permissions they themselves
        hold — prevents an admin (MANAGE_ROLES) from minting a FOUNDER (superuser).

        Now resolved with org grants rather than code defaults. With defaults only,
        an org that granted an extra permission to a weak role (a `role_permissions`
        row) let a lesser actor hand out that permission by assigning the role —
        escalation by proxy, since the actor never has to hold it themselves.

        THIS IS ALSO THE WHOLE OF THE DEVELOPER-ACCESS ESCALATION GUARD (P1-6),
        and it needed no new rule. An Application `admin` holds MANAGE_ROLES but,
        since the surface split, holds no Developer permission at all — so the
        set for any Developer role is not a subset of theirs and this refuses.
        An Application administrator therefore cannot grant Developer access, to
        anyone else or to themselves, which is exactly the "do not silently
        inherit Application admin privileges" requirement enforced by the guard
        that was already here.
        """
        target = await self._effective_permissions(org_id, role, developer_role)
        if not target <= actor_permissions:
            named = f"role '{role.value}'"
            if developer_role is not None:
                dev = (
                    developer_role.value
                    if isinstance(developer_role, DeveloperRole)
                    else developer_role
                )
                named = f"Developer role '{dev}'"
            raise MemberForbidden(
                f"cannot grant {named}: it exceeds your own permissions"
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

        BOTH of the target's surfaces count (P1-6). `membership.developer_role`
        is read from the same persisted row, so an actor who holds no Developer
        authority cannot act on a member who does.
        """
        target = await self._effective_permissions(
            org_id, membership.role, membership.developer_role
        )
        if not target <= actor_permissions:
            raise MemberForbidden(
                "cannot modify a member whose permissions exceed your own"
            )

    async def _ensure_invitable(self, org_id: uuid.UUID) -> None:
        """A PERSONAL workspace holds exactly one member. Refuse to widen it.

        NOT AN AUTHORIZATION CHECK, and it must not be read as one. Authorization
        already happened in `_ensure_grantable` above, on the actor's role — the
        one input `effective_permissions` takes. This is a PRODUCT rule about
        what a personal workspace IS: a `kind` that says "one person's own
        space" while the row carries three members is simply a false statement
        about the tenant, and the console, the entitlement story and the
        Developer surface all read that classification.

        Refused rather than silently reclassified. Turning a personal workspace
        into a company on first invite would change what a tenant is as a side
        effect of an unrelated action; converting it is a deliberate operation
        that does not exist yet, and saying so is more honest than guessing.

        Fails OPEN on a missing row, deliberately: a `None` organization here
        means the row vanished under a caller whose `org_id` was already
        trusted, and turning that into a 400 on the invite path would report a
        storage problem as a product rule. Nothing is widened that was not
        already reachable — the actor's permission gate has already passed.
        """
        organization = await self.db.get(Organization, org_id)
        if organization is None:
            return
        if organization.kind == OrganizationKind.PERSONAL.value:
            raise MemberError(
                "a personal workspace cannot have additional members — "
                "create an organization to collaborate"
            )

    async def invite(
        self, *, org_id: uuid.UUID, email: str, role: Role, actor_id: uuid.UUID,
        actor_permissions: frozenset[str] = frozenset(),
        developer_role: DeveloperRole | None = None,
    ) -> Membership:
        """Invite a member, optionally with Developer access from the start.

        `developer_role=None` is the DEFAULT and means Application-only — the
        value every existing caller produces, so an invite that does not mention
        the Developer surface grants nothing on it. That is the P1-6 requirement
        stated as a default rather than as a check: Developer access is never
        acquired by omission.

        Both halves pass one `_ensure_grantable`, so the escalation bound is over
        the COMBINED assignment. An actor who may grant `employee` but holds no
        Developer authority cannot smuggle a Developer role in beside it.
        """
        await self._ensure_grantable(org_id, role, actor_permissions, developer_role)
        await self._ensure_invitable(org_id)
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
        membership = Membership(
            user_id=user.id,
            organization_id=org_id,
            role=role,
            developer_role=developer_role.value if developer_role else None,
        )
        self.db.add(membership)
        await self.db.flush()
        await self.audit.record(
            AuditAction.MEMBER_INVITE, org_id=org_id, actor_id=actor_id,
            target_id=str(user.id),
            metadata={
                "email": email,
                "role": role.value,
                # Recorded even when absent, so the audit trail distinguishes
                # "invited without Developer access" from a row written before
                # the field existed.
                "developer_role": developer_role.value if developer_role else None,
            },
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
        # The same rule for the Developer surface (P1-6): removing the last
        # Developer owner would leave nobody able to grant Developer access, so
        # the tenant could never reach that surface again. See
        # `_count_developer_owners_locked`.
        if (
            membership.developer_role == DeveloperRole.OWNER.value
            and await self._count_developer_owners_locked(org_id) <= 1
        ):
            raise MemberConflict(
                "cannot remove the last Developer owner of the organization — "
                "assign Developer owner to another member first"
            )
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

    async def assign_developer_role(
        self, *, org_id: uuid.UUID, user_id: uuid.UUID,
        developer_role: DeveloperRole | None, actor_id: uuid.UUID,
        actor_permissions: frozenset[str] = frozenset(),
    ) -> Membership:
        """Grant, change or REVOKE a member's Developer access (P1-6).

        A SEPARATE OPERATION FROM `assign_role`, not a second field on it. The
        two surfaces are independent by design, and folding them into one
        payload would mean every Developer-access change had to restate the
        Application role (or make it optional, which changes an existing
        contract). This route is additive; `POST /members/{id}/role` is
        untouched and keeps its exact meaning.

        `developer_role=None` REVOKES Developer access, and that is the whole
        revocation mechanism — NULL is the defined "no access to this surface"
        value, so there is nothing else to clear. It is still gated: revoking
        requires passing `_ensure_may_manage` against the target's CURRENT
        standing, so a lesser actor cannot strip a Developer owner.

        Escalation is bounded exactly as `assign_role` bounds it, by the guard
        that was already here. An actor must hold every permission the new
        assignment would confer (`_ensure_grantable`) AND every permission the
        target currently holds (`_ensure_may_manage`). Since P1-6 an Application
        administrator holds no Developer permission, so both halves refuse them.
        """
        # Cross-org safety: selected under the actor's trusted org_id (see `remove`).
        membership = await self.db.scalar(
            select(Membership).where(
                Membership.user_id == user_id, Membership.organization_id == org_id
            )
        )
        if membership is None:
            raise MemberError("not a member")
        # The GRANT bound is evaluated against the membership's own Application
        # role, not a placeholder: the target ends up holding
        # `role ∪ developer_role`, and that combined set is what the actor must
        # already hold.
        await self._ensure_grantable(
            org_id, membership.role, actor_permissions, developer_role
        )
        await self._ensure_may_manage(org_id, membership, actor_permissions)
        old = membership.developer_role
        new = developer_role.value if developer_role else None
        # ── Succession: only an OWNER may confer or remove the owner role ──
        #
        # REQUIRED BY P1-8, NOT AN EXTRA. Developer `admin` and `owner` hold
        # identical permission SETS — deliberately, so `_ensure_grantable`'s
        # subset comparison stays well defined — and P1-8 gave both
        # MANAGE_DEVELOPER_ACCESS so a Developer administrator can run their
        # own team. Those two facts together mean an admin reaching this route
        # would pass every existing guard while promoting themselves to owner:
        # the target set equals their own, so nothing refuses it. This is the
        # escalation guard for the capability P1-8 added.
        #
        # EXPRESSED AS GOVERNANCE RATHER THAN AS A PERMISSION for the same
        # reason the last-founder and last-Developer-owner rules are: it is a
        # rule about a SPECIFIC ROLE VALUE, and permissions cannot say "may
        # grant this one role" without the vocabulary gaining an ordering it
        # does not have.
        #
        # THIS IS ALSO WHAT MAKES OWNER MORE THAN A LABEL. Before it, owner and
        # admin differed only in that an organization must keep one owner —
        # a constraint on removal, not a power. Succession is a power: the
        # owner decides who else may hold the top of the Developer surface.
        #
        # An ACTOR holding MANAGE_ROLES (an Application founder/admin) is
        # unaffected: they never pass `_ensure_grantable` for a Developer role
        # unless they hold Developer authority themselves, and if they hold
        # Developer owner they satisfy this too.
        touches_owner = DeveloperRole.OWNER.value in (old, new)
        if touches_owner and not await self._actor_is_developer_owner(org_id, actor_id):
            raise MemberForbidden(
                "only a Developer owner can assign or remove the Developer "
                "owner role"
            )
        # Governance: the last Developer owner cannot be demoted away from owner,
        # mirroring the last-founder rule. Checked only when this call would
        # actually move them off `owner`.
        if (
            old == DeveloperRole.OWNER.value
            and new != DeveloperRole.OWNER.value
            and await self._count_developer_owners_locked(org_id) <= 1
        ):
            raise MemberConflict(
                "cannot remove Developer owner from the last Developer owner of "
                "the organization — assign Developer owner to another member first"
            )
        membership.developer_role = new
        await self.db.flush()
        # Losing Developer access must take effect NOW, not within the 30s
        # Principal cache TTL. `revoke_all_for_user` is the existing lever and it
        # evicts every cached Principal for the user as well as bumping the
        # bulk-revoke watermark, so the member re-authenticates with the new
        # assignment. Only on a NARROWING: a widening is safe to let the TTL
        # pick up, and signing someone out to give them MORE access would be a
        # gratuitous interruption.
        widened = new is not None and (
            old is None
            or DEVELOPER_ROLE_PERMISSIONS.get(old, frozenset())
            <= DEVELOPER_ROLE_PERMISSIONS.get(new, frozenset())
        )
        if not widened:
            await TokenRevocationService(self.db).revoke_all_for_user(user_id)
        await self.audit.record(
            # The EXISTING role-change action, with the surface named in metadata.
            # A second enum member would split "a role changed" across two values
            # that every audit query would then have to remember to union.
            AuditAction.ROLE_CHANGE, org_id=org_id, actor_id=actor_id,
            target_id=str(user_id),
            metadata={"surface": "developer", "from": old, "to": new},
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
