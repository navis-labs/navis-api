"""Bring legacy-plaintext integration tokens under encryption at rest.

WHY THIS EXISTS
---------------
`TokenCipher` has always been backward compatible: an untagged value is treated
as legacy plaintext and returned as-is (signals/crypto.py::decrypt), and its
docstring promises "re-encryption happens lazily on next write". That promise
holds for `access_token`, which every sync pass and every sender rewrites
(orchestration/sync.py, execution/providers/_senders.py). It does NOT hold for
`refresh_token`: that column is only rewritten when Google ROTATES the refresh
token, which is rare, so a row written while the cipher was disabled keeps a
plaintext refresh token indefinitely. Lazy re-encryption cannot reach it.

The consequence is the durable credential — the one that survives access-token
expiry — sitting in plaintext in the database and therefore in every backup and
replica, silently, because `decrypt()` accepts it.

WHAT IT DOES
------------
Selects only rows whose token column is actually untagged, and rewrites that one
column with `token_cipher.encrypt(...)`. Nothing else on the row is touched.

`decrypt()` IS NEVER CALLED. The stored value already IS the plaintext, so the
migration encrypts it directly. Three properties follow:

  * it works with `signals_reject_legacy_plaintext` already enabled (that switch
    lives in `decrypt`), so the sunset flag and this job are order-independent;
  * it emits no `signals.token.legacy_plaintext_seen` metric noise;
  * there is no decrypted-plaintext local variable anywhere in this module.

SAFETY PROPERTIES
-----------------
  * IDEMPOTENT — selection is `NOT LIKE 'enc:1:%'`, so encrypting a row removes
    it from the next query's result set. A second run rewrites zero rows.
  * RESUMABLE — no cursor; an interrupted run just continues where it stopped.
  * BOUNDED — paged, and each page is one short transaction.
  * CONCURRENCY-SAFE — each row is re-read `FOR UPDATE` and re-checked inside its
    own transaction, so a sync pass writing the same row in parallel either loses
    the race harmlessly (it writes ciphertext too) or is serialized behind us.
  * FAIL-CLOSED — refuses to run when the cipher is disabled, because
    `encrypt()` is a passthrough with no keys and would "succeed" while changing
    nothing, reporting a migration that never happened.
  * PER-ROW ISOLATION — a row that fails to encrypt is left EXACTLY as it was
    (still readable through the legacy path) and is retried on the next run. It
    is never nulled, never marked migrated, never deactivated.

Never logs, returns, or records a token value — only ids, providers and counts.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import or_, select

from navis.identity.models import Integration
from navis.platform.database import session_scope
from navis.platform.monitoring import get_logger
from navis.platform.monitoring.metrics import metrics
from navis.platform.queue import celery_app, run_async
from navis.signals.crypto import token_cipher

logger = get_logger("signals.token.backfill")

PAGE_SIZE = 100

# The two credential columns TokenCipher protects. Both are gated by the same
# sunset switch, so both must be encrypted before it can be enabled.
_TOKEN_COLUMNS = ("access_token", "refresh_token")


class TokenBackfillUnavailable(RuntimeError):
    """The cipher has no keys, so encryption would be a silent no-op."""


@dataclass(slots=True)
class BackfillReport:
    """Counts only. Never carries token material."""

    scanned: int = 0
    encrypted: dict[str, int] = field(default_factory=dict)
    failed: list[dict[str, str]] = field(default_factory=list)

    @property
    def rewritten_rows(self) -> int:
        return self.scanned - len(self.failed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scanned": self.scanned,
            "encrypted": dict(self.encrypted),
            "failed": len(self.failed),
            "failures": self.failed,
        }


def _plaintext_predicate():
    """Rows holding at least one untagged (legacy plaintext) token column.

    `NOT LIKE 'enc:1:%'` mirrors `TokenCipher.is_encrypted` in SQL so the page
    query and the in-Python re-check agree. NULL is excluded by `is_not(None)`:
    a row with no token is not a migration target.
    """
    return or_(
        *[
            (getattr(Integration, col).is_not(None))
            & (getattr(Integration, col).not_like("enc:1:%"))
            for col in _TOKEN_COLUMNS
        ]
    )


async def _pending_ids(db, *, org_id: uuid.UUID | None, limit: int) -> list[uuid.UUID]:
    stmt = (
        select(Integration.id)
        .where(_plaintext_predicate())
        .order_by(Integration.id)
        .limit(limit)
    )
    if org_id is not None:
        stmt = stmt.where(Integration.organization_id == org_id)
    return list((await db.scalars(stmt)).all())


async def _encrypt_one(db, integration_id: uuid.UUID, report: BackfillReport) -> None:
    """Re-read the row under a lock, re-check, rewrite only untagged columns."""
    row = await db.get(Integration, integration_id, with_for_update=True)
    if row is None:  # deleted between the page query and now
        return
    changed: list[str] = []
    for col in _TOKEN_COLUMNS:
        value = getattr(row, col)
        # TOCTOU re-check: a concurrent sync may have encrypted it already.
        if value is None or token_cipher.is_encrypted(value):
            continue
        ciphertext = token_cipher.encrypt(value)
        if ciphertext is None or not token_cipher.is_encrypted(ciphertext):
            # encrypt() degraded to passthrough — refuse to "migrate" the row.
            raise TokenBackfillUnavailable(
                f"encrypt() did not produce ciphertext for {col}"
            )
        setattr(row, col, ciphertext)
        changed.append(col)
    if not changed:
        return
    report.scanned += 1
    for col in changed:
        report.encrypted[col] = report.encrypted.get(col, 0) + 1
    metrics.incr("signals.token.backfill_encrypted", value=len(changed))
    logger.info(
        "re-encrypted integration=%s provider=%s org=%s columns=%s",
        row.id, row.provider.value if row.provider else None,
        row.organization_id, ",".join(changed),
    )


async def reencrypt_legacy_tokens(
    *, org_id: uuid.UUID | None = None, page_size: int = PAGE_SIZE,
    max_rows: int | None = None,
) -> BackfillReport:
    """Encrypt every untagged integration token. Idempotent; safe to re-run.

    `org_id` narrows to one tenant (used by the test suite to prove isolation).
    `max_rows` bounds a single invocation.

    Each row commits in its OWN transaction so one failure cannot roll back the
    rows already migrated, and a failed row stays legacy-readable for the retry.
    """
    if not token_cipher.enabled:
        # Fail closed. With no keys, encrypt() returns its input unchanged and
        # every row would be reported "migrated" while staying plaintext.
        raise TokenBackfillUnavailable(
            "SIGNALS_TOKEN_KEYS is not configured; refusing to run a migration "
            "that would silently leave tokens unencrypted"
        )

    report = BackfillReport()
    processed = 0
    while max_rows is None or processed < max_rows:
        limit = page_size if max_rows is None else min(page_size, max_rows - processed)
        async with session_scope() as db:
            ids = await _pending_ids(db, org_id=org_id, limit=limit)
        if not ids:
            break
        progressed = False
        for integration_id in ids:
            try:
                async with session_scope() as db:
                    await _encrypt_one(db, integration_id, report)
                progressed = True
            except Exception as exc:  # noqa: BLE001 — isolate, record, continue
                # Redacted by construction: id/provider/category, never a token.
                report.failed.append(
                    {
                        "integration_id": str(integration_id),
                        "failure": type(exc).__name__,
                    }
                )
                metrics.incr("signals.token.backfill_failed")
                logger.error(
                    "re-encryption FAILED integration=%s kind=%s "
                    "(row left unchanged and still readable; will retry)",
                    integration_id, type(exc).__name__,
                )
            processed += 1
        if not progressed:
            # Every row in this page failed. Stop rather than spin: the same page
            # would be selected again forever.
            logger.error("token backfill made no progress; stopping after %d row(s)", processed)
            break
    logger.info("token backfill complete %s", report.to_dict())
    return report


@celery_app.task(name="navis.signals.token_backfill.reencrypt_legacy_tokens")
def reencrypt_legacy_tokens_task(
    org_id: str | None = None, max_rows: int | None = None
) -> dict[str, Any]:
    """Operator-invoked, one-shot. Deliberately NOT on the beat schedule: a
    completed migration is a no-op, and a repeating job would hide the fact that
    something is re-introducing plaintext rather than surfacing it.

    Re-runnable, which is how Phase-7 failures are retried: a row that failed is
    still legacy-readable and is selected again on the next invocation.
    """
    oid = uuid.UUID(org_id) if org_id else None
    return run_async(reencrypt_legacy_tokens(org_id=oid, max_rows=max_rows)).to_dict()


async def count_legacy_tokens() -> dict[str, int]:
    """Read-only census. Used by the pre-flight and post-migration checks."""
    from sqlalchemy import func

    async with session_scope() as db:
        out: dict[str, int] = {}
        for col in _TOKEN_COLUMNS:
            column = getattr(Integration, col)
            out[f"{col}_total"] = int(
                await db.scalar(select(func.count()).select_from(Integration)) or 0
            )
            out[f"{col}_null"] = int(
                await db.scalar(
                    select(func.count()).select_from(Integration).where(column.is_(None))
                ) or 0
            )
            out[f"{col}_encrypted"] = int(
                await db.scalar(
                    select(func.count()).select_from(Integration).where(
                        column.is_not(None), column.like("enc:1:%")
                    )
                ) or 0
            )
            out[f"{col}_plaintext"] = int(
                await db.scalar(
                    select(func.count()).select_from(Integration).where(
                        column.is_not(None), column.not_like("enc:1:%")
                    )
                ) or 0
            )
        return out
