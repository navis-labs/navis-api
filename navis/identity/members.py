"""Delegated administration: invite / list / remove members, assign roles.

Every mutation is audited. Removing a member bulk-revokes their access tokens.
"""
from __future__ import annotations

import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from navis.identity.audit import AuditAction, AuditService
from navis.identity.models import Membership, Role, User
from navis.identity.permissions import default_permissions_for_role
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

    @staticmethod
    def _ensure_grantable(role: Role, actor_permissions: frozenset[str]) -> None:
        """A member may only grant a role whose permissions they themselves hold —
        prevents an admin (MANAGE_ROLES) from minting a FOUNDER (superuser)."""
        target = default_permissions_for_role(role.value)
        if not target <= actor_permissions:
            raise MemberForbidden(
                f"cannot grant role '{role.value}': it exceeds your own permissions"
            )

    async def invite(
        self, *, org_id: uuid.UUID, email: str, role: Role, actor_id: uuid.UUID,
        actor_permissions: frozenset[str] = frozenset(),
    ) -> Membership:
        self._ensure_grantable(role, actor_permissions)
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

    async def remove(self, *, org_id: uuid.UUID, user_id: uuid.UUID, actor_id: uuid.UUID) -> None:
        membership = await self.db.scalar(
            select(Membership).where(
                Membership.user_id == user_id, Membership.organization_id == org_id
            )
        )
        if membership is None:
            raise MemberError("not a member")
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
        self._ensure_grantable(role, actor_permissions)
        membership = await self.db.scalar(
            select(Membership).where(
                Membership.user_id == user_id, Membership.organization_id == org_id
            )
        )
        if membership is None:
            raise MemberError("not a member")
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
