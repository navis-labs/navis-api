"""Append-only audit logging for identity-sensitive actions."""
from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from navis.identity.enterprise_models import AuditLog


class AuditAction:
    LOGIN = "login"
    LOGOUT = "logout"
    SIGNUP = "signup"
    TOKEN_REVOKE = "token_revoke"
    MEMBER_INVITE = "member_invite"
    MEMBER_REMOVE = "member_remove"
    ROLE_CHANGE = "role_change"
    PERMISSION_CHANGE = "permission_change"
    API_KEY_CREATE = "api_key_create"
    API_KEY_REVOKE = "api_key_revoke"
    SERVICE_ACCOUNT_CREATE = "service_account_create"
    SERVICE_ACCOUNT_ROTATE = "service_account_rotate"
    SERVICE_ACCOUNT_REVOKE = "service_account_revoke"
    USER_DEACTIVATE = "user_deactivate"


class AuditService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def record(
        self,
        action: str,
        *,
        org_id: uuid.UUID | None = None,
        actor_id: uuid.UUID | None = None,
        actor_kind: str = "user",
        target_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AuditLog:
        entry = AuditLog(
            organization_id=org_id,
            actor_id=actor_id,
            actor_kind=actor_kind,
            action=action,
            target_id=target_id,
            audit_metadata=metadata or {},
        )
        self.db.add(entry)
        await self.db.flush()
        return entry

    async def list_for_org(self, org_id: uuid.UUID, *, limit: int = 200) -> list[AuditLog]:
        return list(
            (
                await self.db.scalars(
                    select(AuditLog)
                    .where(AuditLog.organization_id == org_id)
                    .order_by(AuditLog.occurred_at.desc())
                    .limit(limit)
                )
            ).all()
        )
