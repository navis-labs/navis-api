"""API keys: hashed-at-rest, scoped, revocable credentials.

Format: navis_<prefix>_<secret>. Only the SHA-256 of the full string is stored,
plus the public prefix for O(1) lookup. The plaintext is returned exactly once.
"""
from __future__ import annotations

import enum
import hashlib
import secrets
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from navis.identity.enterprise_models import ApiKey

_PREFIX_BYTES = 8   # -> 16 hex chars (fits String(16)); ~1.8e19 space, collision-free at scale
_SECRET_BYTES = 32
# Only refresh last_used_at when it's this stale — avoids a write on EVERY auth
# request (write-amplification) while keeping "last used" accurate to the minute.
_LAST_USED_REFRESH = timedelta(minutes=15)



class CredentialRefusal(str, enum.Enum):
    """Why a presented credential was not accepted.

    NEVER SENT VERBATIM TO AN UNAUTHENTICATED CALLER except `EXPIRED` — see
    `authz._principal_from_api_key`. `MALFORMED` and `UNKNOWN` are what a
    guessing client produces and must stay indistinguishable from each other;
    `REVOKED` and `EXPIRED` can only be produced by somebody who holds a real
    credential, so telling them which one they hold reveals nothing they did not
    already have.
    """

    MALFORMED = "malformed"
    UNKNOWN = "unknown"
    REVOKED = "revoked"
    EXPIRED = "expired"


@dataclass(frozen=True, slots=True)
class CredentialCheck:
    """The row when a credential may authenticate, otherwise the reason it may not.

    Exactly one side is populated: `key` is None whenever `refusal` is set, and
    vice versa. Frozen so an authorization answer cannot be edited after the
    fact by the code that received it.
    """

    key: ApiKey | None
    refusal: CredentialRefusal | None


def _hash(plaintext: str) -> str:
    return hashlib.sha256(plaintext.encode()).hexdigest()


def _is_expired(expires_at: datetime | None, now: datetime) -> bool:
    """NULL means "does not expire" — the state of every key minted before 0052.

    A naive `expires_at` cannot come from the database (the column is
    timestamptz) but can come from a test or a fixture, and comparing naive to
    aware raises `TypeError`. An exception on the authentication path would be a
    500 where a decision belongs, so it is coerced to UTC rather than trusted to
    be aware.
    """
    if expires_at is None:
        return False
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    return expires_at <= now


def _mint() -> tuple[str, str]:
    prefix = secrets.token_hex(_PREFIX_BYTES)
    secret = secrets.token_urlsafe(_SECRET_BYTES)
    return prefix, f"navis_{prefix}_{secret}"


class ApiKeyService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def create(
        self,
        *,
        org_id: uuid.UUID,
        name: str,
        scopes: list[str],
        created_by: uuid.UUID | None = None,
        service_account_id: uuid.UUID | None = None,
        #: When this credential stops working. None — the default — means it
        #: does not expire, which keeps every existing caller producing exactly
        #: the key it produced before expiry existed.
        expires_at: datetime | None = None,
    ) -> tuple[ApiKey, str]:
        if expires_at is not None:
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=timezone.utc)
            # A credential born expired authenticates nothing, so accepting one
            # is accepting a support ticket. Refused here rather than at one
            # edge, because three routes mint keys and this is the floor under
            # all of them.
            if expires_at <= datetime.now(timezone.utc):
                raise ValueError("expires_at must be in the future")
        prefix, plaintext = _mint()
        row = ApiKey(
            organization_id=org_id,
            name=name,
            prefix=prefix,
            hashed_secret=_hash(plaintext),
            scopes=list(scopes),
            created_by=created_by,
            service_account_id=service_account_id,
            expires_at=expires_at,
        )
        self.db.add(row)
        await self.db.flush()
        return row, plaintext  # plaintext shown once, never stored

    async def verify(self, plaintext: str) -> ApiKey | None:
        """Resolve a presented key to its USABLE row, updating last_used_at.

        Unchanged contract: the row when the credential may authenticate, `None`
        otherwise. Delegates to `check`, so there is ONE implementation of what
        "usable" means — two would eventually disagree, and the disagreement
        would be an authentication bypass.
        """
        return (await self.check(plaintext)).key

    async def check(self, plaintext: str) -> CredentialCheck:
        """The same decision as `verify`, plus WHY a credential was refused.

        The reason exists for two readers and neither is the attacker. An
        operator needs `expired` and `revoked` apart in metrics to tell a
        rotation everybody forgot from a key somebody cut off; a developer whose
        key aged out needs to be told that rather than "invalid", which is the
        difference between a one-minute fix and an afternoon. See
        `authz._principal_from_api_key` for what reaches the wire.
        """
        # The secret segment may itself contain "_" (base64url), so split at most twice.
        parts = plaintext.split("_", 2)
        if len(parts) != 3 or parts[0] != "navis":
            return CredentialCheck(None, CredentialRefusal.MALFORMED)
        prefix = parts[1]
        # prefix is NOT unique (8 hex) — iterate all matches and constant-time
        # compare, so a prefix collision can never reject a genuinely valid key.
        #
        # REVOKED ROWS ARE FETCHED, NOT FILTERED OUT IN SQL. They used to be
        # excluded by the WHERE clause, which made a revoked key and an unknown
        # one indistinguishable here — and the same would now be true of an
        # expired one. The refusal is identical either way; being able to SAY
        # which happened is the whole point of this method, and it costs nothing
        # because `prefix` is indexed and matches a handful of rows.
        digest = _hash(plaintext)
        rows = (await self.db.scalars(select(ApiKey).where(ApiKey.prefix == prefix))).all()
        now = datetime.now(timezone.utc)
        for row in rows:
            if not secrets.compare_digest(row.hashed_secret, digest):
                continue
            # ── The credential is GENUINE. Now: is it still usable? ──────
            if row.revoked_at is not None:
                # Checked before expiry: revocation is a decision somebody made
                # and is terminal, so it is the honest reason even when an
                # expiry has also passed. Matches `credential_status_of`.
                return CredentialCheck(None, CredentialRefusal.REVOKED)
            if _is_expired(row.expires_at, now):
                return CredentialCheck(None, CredentialRefusal.EXPIRED)
            # Throttle the write: only bump last_used_at when it's stale, so a key
            # doing 100 req/s doesn't generate 100 writes/s for no business value.
            if row.last_used_at is None or (now - row.last_used_at) > _LAST_USED_REFRESH:
                row.last_used_at = now
            return CredentialCheck(row, None)
        return CredentialCheck(None, CredentialRefusal.UNKNOWN)

    async def revoke(self, key_id: uuid.UUID, *, org_id: uuid.UUID) -> ApiKey | None:
        """Org-scoped: a key may only be revoked by its OWNING org (tenant isolation)."""
        row = await self.db.get(ApiKey, key_id)
        if row is None or row.organization_id != org_id:
            return None  # not found OR belongs to another tenant -> 404 at the edge
        if row.revoked_at is None:
            row.revoked_at = datetime.now(timezone.utc)
            await self.db.flush()
        return row

    async def revoke_for_service_account(self, service_account_id: uuid.UUID) -> None:
        # Single bulk UPDATE — a service account may have many keys; no row-load loop.
        await self.db.execute(
            update(ApiKey)
            .where(
                ApiKey.service_account_id == service_account_id, ApiKey.revoked_at.is_(None)
            )
            .values(revoked_at=datetime.now(timezone.utc))
        )

    async def list_for_org(
        self, org_id: uuid.UUID, *, limit: int = 50, offset: int = 0
    ) -> list[ApiKey]:
        return list(
            (
                await self.db.scalars(
                    select(ApiKey)
                    .where(ApiKey.organization_id == org_id)
                    .order_by(ApiKey.created_at.desc(), ApiKey.id)
                    .limit(limit)
                    .offset(offset)
                )
            ).all()
        )
