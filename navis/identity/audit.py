"""Append-only audit logging for identity-sensitive actions."""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from navis.identity.enterprise_models import AuditLog

# Substrings that mark a metadata key as sensitive — never persist its value to the
# audit log (audit rows are widely readable; a leaked secret there is a real breach).
_SENSITIVE = ("password", "secret", "token", "authorization", "api_key", "apikey", "credential")
_REDACTED = "***REDACTED***"
# Audit metadata holds FACTS, not blobs. Cap it so a careless caller can't push a
# multi-MB payload into the audit table.
_MAX_METADATA_BYTES = 8 * 1024


def _sanitize(value: Any) -> Any:
    """Recursively redact sensitive keys before they hit the audit log."""
    if isinstance(value, dict):
        return {
            k: (_REDACTED if any(s in str(k).lower() for s in _SENSITIVE) else _sanitize(v))
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_sanitize(v) for v in value]
    return value


class AuditAction(StrEnum):
    """Audit vocabulary is a closed enum — callers using these constants get typo
    safety (`record(AuditAction.LOGIN)`, not `record("logn")`). StrEnum members ARE
    strings, so storage/compare is unchanged and free-form action strings still work
    for dynamically-named events (e.g. org lifecycle)."""

    LOGIN = "login"
    LOGOUT = "logout"
    SIGNUP = "signup"
    TOKEN_REVOKE = "token_revoke"
    REFRESH_REUSE_DETECTED = "refresh_reuse_detected"  # stolen/replayed refresh token
    MEMBER_INVITE = "member_invite"
    MEMBER_REMOVE = "member_remove"
    ROLE_CHANGE = "role_change"
    PERMISSION_CHANGE = "permission_change"
    API_KEY_CREATE = "api_key_create"
    API_KEY_REVOKE = "api_key_revoke"
    SERVICE_ACCOUNT_CREATE = "service_account_create"
    SERVICE_ACCOUNT_ROTATE = "service_account_rotate"
    SERVICE_ACCOUNT_REVOKE = "service_account_revoke"
    SSO_UPDATED = "sso_updated"
    SSO_LOGIN = "sso_login"            # successful SSO sign-in (google/oidc/saml)
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
        clean = _sanitize(metadata or {})
        if len(json.dumps(clean, default=str)) > _MAX_METADATA_BYTES:
            # Never let a giant payload bloat the audit table — record the fact, drop the blob.
            clean = {"_truncated": True}
        entry = AuditLog(
            organization_id=org_id,
            actor_id=actor_id,
            actor_kind=actor_kind,
            action=str(action),
            target_id=target_id,
            audit_metadata=clean,
        )
        self.db.add(entry)
        await self.db.flush()
        return entry

    async def list_for_org(
        self,
        org_id: uuid.UUID,
        *,
        limit: int = 100,
        offset: int = 0,
        since: datetime | None = None,
    ) -> list[AuditLog]:
        """Paginated, newest-first. `since` filters to events at/after a timestamp —
        the audit log is the fastest-growing identity table, so unbounded reads are
        forbidden; callers page with limit/offset (and may window with `since`)."""
        stmt = select(AuditLog).where(AuditLog.organization_id == org_id)
        if since is not None:
            stmt = stmt.where(AuditLog.occurred_at >= since)
        stmt = stmt.order_by(AuditLog.occurred_at.desc()).limit(limit).offset(offset)
        return list((await self.db.scalars(stmt)).all())
