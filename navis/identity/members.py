"""Delegated administration: invite / list / remove members, assign roles.

Every mutation is audited. Removing a member bulk-revokes their access tokens.
"""
from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from navis.identity.audit import AuditAction, AuditService
from navis.identity.models import Membership, Role, User
from navis.identity.tokens import TokenRevocationService


class MemberError(Exception):
    pass


class MemberService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db
        self.audit = AuditService(db)

    async def invite(self, *, org_id: uuid.UUID, email: str, role: Role, actor_id: uuid.UUID) -> Membership:
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
        await self.db.delete(membership)
        # Membership removal revokes the user's active access immediately.
        await TokenRevocationService(self.db).revoke_all_for_user(user_id)
        await self.audit.record(
            AuditAction.MEMBER_REMOVE, org_id=org_id, actor_id=actor_id, target_id=str(user_id),
        )

    async def assign_role(self, *, org_id: uuid.UUID, user_id: uuid.UUID, role: Role, actor_id: uuid.UUID) -> Membership:
        membership = await self.db.scalar(
            select(Membership).where(
                Membership.user_id == user_id, Membership.organization_id == org_id
            )
        )
        if membership is None:
            raise MemberError("not a member")
        old = membership.role.value if membership.role else None
        membership.role = role
        await self.db.flush()
        await self.audit.record(
            AuditAction.ROLE_CHANGE, org_id=org_id, actor_id=actor_id, target_id=str(user_id),
            metadata={"from": old, "to": role.value},
        )
        return membership

    async def list_members(self, org_id: uuid.UUID) -> list[Membership]:
        return list(
            (await self.db.scalars(select(Membership).where(Membership.organization_id == org_id))).all()
        )
