"""Access-token revocation: a jti denylist + a per-user bulk-revoke watermark.

- Single-token revoke (logout): insert the jti into `revoked_tokens`.
- Bulk revoke (deactivation, membership removal): bump `users.tokens_valid_from`;
  any access token whose `iat` predates it is rejected.

Validation is consulted on every authenticated request (see authz.get_principal).
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from navis.identity.enterprise_models import RevokedToken
from navis.identity.models import User
from navis.platform.monitoring.metrics import metrics


class TokenRevocationService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def revoke_jti(self, jti: str, *, user_id: uuid.UUID | None, expires_at: datetime | None) -> None:
        if await self.db.get(RevokedToken, jti):
            return
        self.db.add(RevokedToken(jti=jti, user_id=user_id, expires_at=expires_at))
        await self.db.flush()

    async def is_jti_revoked(self, jti: str | None) -> bool:
        if not jti:
            return False
        revoked = await self.db.get(RevokedToken, jti) is not None
        if revoked:
            # A revoked token was actively presented — a stale/replay attempt. Cheap
            # SOC signal (no DB write); anomalies in this counter are worth alerting on.
            metrics.incr("auth.revoked_token_presented")
        return revoked

    async def revoke_all_for_user(self, user_id: uuid.UUID, *, at: datetime | None = None) -> None:
        user = await self.db.get(User, user_id)
        if user is not None:
            user.tokens_valid_from = at or datetime.now(timezone.utc)
            await self.db.flush()

    async def purge_expired(self, *, now: datetime | None = None) -> int:
        # Single bulk DELETE (served by ix_rt_expires) — no N-row ORM load/delete loop.
        now = now or datetime.now(timezone.utc)
        result = await self.db.execute(
            delete(RevokedToken).where(
                RevokedToken.expires_at.is_not(None), RevokedToken.expires_at < now
            )
        )
        return result.rowcount or 0


def token_predates_bulk_revoke(payload: dict, user: User) -> bool:
    """True if this access token was issued before the user's bulk-revoke watermark.

    Sub-second boundary: `iat` is integer seconds, so we floor it to the second — which
    makes `issued` EARLIER than the real issuance and thus biases toward `True` (reject).
    That is deliberate and fail-safe: a token issued in the same second as a bulk-revoke
    may be re-rejected (the user re-logs in), but a token issued just before the revoke
    can NEVER slip through. Truncating the watermark instead would risk under-rejection.
    """
    if user.tokens_valid_from is None:
        return False
    iat = payload.get("iat")
    if iat is None:
        return True  # no iat -> cannot prove it's recent -> reject
    issued = datetime.fromtimestamp(int(iat), tz=timezone.utc)
    return issued < user.tokens_valid_from
