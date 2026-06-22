"""GmailConnector — production capture for Gmail.

Architecture (Google's recommended pattern):

    watch() -> Pub/Sub notification -> historyId -> history.list() -> fetch -> SignalEvents

  * full_sync: bounded backfill at onboarding; records the starting historyId.
  * incremental_sync: history.list paginated from the stored historyId; advances
    the cursor; on 404 (historyId too old / expired) requests a full resync.
  * create_watch: registers the Pub/Sub channel and returns watch state to persist.

Pure capture — no DB writes, no business logic. The sync engine persists cursors
and ingests the returned events.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

from navis.platform.config import settings
from navis.platform.monitoring import get_logger
from navis.platform.monitoring.metrics import metrics
from navis.platform.security import field_cipher
from navis.signals.base import (
    SignalContext,
    SignalEvent,
    SignalEventType,
    SignalSource,
    SyncResult,
    WorkspaceConnector,
)
from navis.signals.gmail.client import GmailClient
from navis.signals.gmail.parser import parse_message
from navis.signals.gmail.watch import build_watch_state
from navis.signals.google.transport import GoogleAPIError

logger = get_logger("signals.gmail")

# Sensitive free-text content encrypted at rest in the signal payload (Fernet via the shared
# field_cipher — passthrough in dev, prod-mandatory keys). The append-only log, its backups,
# and any analytics replica therefore hold ciphertext, not raw email content. Only the Memory
# extraction seam (SignalViewFactory.from_record) decrypts, transparently, on read. Routing/
# identity fields (from/to/cc, thread_id, labels) stay plaintext — the people graph needs them.
_ENCRYPTED_CONTENT_FIELDS = ("body_text", "snippet", "bcc")


def _encrypt_content(parsed: dict[str, Any]) -> None:
    """Encrypt sensitive content fields in place and self-describe which scalar fields were
    sealed, so the reader decrypts generically without importing this module's field list."""
    sealed = [f for f in _ENCRYPTED_CONTENT_FIELDS if parsed.get(f) is not None]
    for f in sealed:
        parsed[f] = field_cipher.encrypt(parsed[f])
    # Listed regardless of cipher state: decrypt() is a no-op on untagged dev-passthrough
    # plaintext, so the reader runs one uniform path in every environment.
    parsed["_content_fields"] = list(sealed)
    # Hidden-recipient list + attachment filenames ("Layoff_Plan_Q4.xlsx" is itself sensitive
    # org intelligence). No current consumer reads them, so they are sealed at rest but NOT in
    # _content_fields — each value carries the fenc: tag, and decrypt() is a no-op on plaintext
    # for any future reader that needs them.
    if parsed.get("bcc_emails"):
        parsed["bcc_emails"] = [field_cipher.encrypt(e) for e in parsed["bcc_emails"]]
    for att in parsed.get("attachments") or []:
        if att.get("filename") is not None:
            att["filename"] = field_cipher.encrypt(att["filename"])


class GmailSyncIncomplete(Exception):
    """A history drain hit the page-safety cap before exhausting nextPageToken. Raised
    (not swallowed) so the sync fails loud + retryable instead of advancing the cursor
    past unprocessed history pages (which would skip those messages forever)."""


class GmailConnector(WorkspaceConnector):
    source = SignalSource.GMAIL

    async def full_sync(self, ctx: SignalContext, client: GmailClient) -> SyncResult:
        # Seed the cursor FIRST so nothing added during backfill is missed go-forward.
        profile = await client.get_profile()
        start_history_id = str(profile.get("historyId"))

        cap = max(1, settings.gmail_onboard_message_cap)
        message_ids: list[str] = []
        page_token: str | None = None
        while len(message_ids) < cap:
            listing = await client.list_messages(page_token=page_token, max_results=100)
            message_ids.extend(m["id"] for m in listing.get("messages", []))
            page_token = listing.get("nextPageToken")
            if not page_token:
                break
        capped = len(message_ids) >= cap and bool(page_token)
        message_ids = message_ids[:cap]
        if capped:
            # NOT silent: backfill is bounded; older mail beyond the cap is not captured.
            # Operators see this and can run a wider backfill if a full sweep is required.
            metrics.incr("gmail.full_sync.capped")
            logger.warning(
                "gmail full_sync capped at %d messages org=%s; older mail not backfilled",
                cap, ctx.org_id,
            )

        events = await self._fetch_and_normalize(ctx, client, message_ids)
        logger.info("gmail full_sync org=%s captured=%d capped=%s", ctx.org_id, len(events), capped)
        return SyncResult(events=events, cursor={"history_id": start_history_id})

    async def incremental_sync(
        self, ctx: SignalContext, client: GmailClient, cursor: dict[str, Any]
    ) -> SyncResult:
        history_id = cursor.get("history_id")
        if not history_id:
            return SyncResult(full_resync_required=True)

        # startHistoryId stays CONSTANT across the drain (Gmail paginates via pageToken);
        # the new cursor is the response's historyId, adopted ONLY after a complete drain.
        start = str(history_id)
        message_ids: list[str] = []
        next_history_id = start
        page_token: str | None = None
        pages = 0
        max_pages = max(1, settings.gmail_max_history_pages)
        try:
            while True:
                history = await client.list_history(start, page_token=page_token)
                for h in history.get("history", []):
                    for added in h.get("messagesAdded", []):
                        msg = added.get("message", {})
                        if msg.get("id"):
                            message_ids.append(msg["id"])
                if history.get("historyId"):
                    next_history_id = str(history["historyId"])
                pages += 1
                page_token = history.get("nextPageToken")
                if not page_token:
                    break  # drain complete — next_history_id is a valid cursor
                if pages >= max_pages:
                    metrics.incr("gmail.sync.truncated")
                    logger.error(
                        "gmail history drain exceeded %d pages org=%s; not advancing cursor",
                        max_pages, ctx.org_id,
                    )
                    raise GmailSyncIncomplete(
                        f"gmail history exceeded {max_pages} pages without draining"
                    )
        except GoogleAPIError as exc:
            if exc.is_not_found:
                # historyId expired — Google requires a full resync.
                logger.warning("gmail history 404 org=%s; full resync required", ctx.org_id)
                return SyncResult(full_resync_required=True)
            raise

        # De-dup ids within the batch before fetching.
        unique_ids = list(dict.fromkeys(message_ids))
        events = await self._fetch_and_normalize(ctx, client, unique_ids)
        metrics.incr("gmail.sync.events", value=len(events))
        logger.info("gmail incremental org=%s pages=%d captured=%d", ctx.org_id, pages, len(events))
        return SyncResult(events=events, cursor={"history_id": next_history_id})

    async def create_watch(self, ctx: SignalContext, client: GmailClient) -> dict[str, Any]:
        topic = os.environ.get("GOOGLE_PUBSUB_TOPIC")
        if not topic:
            logger.warning("gmail watch skipped org=%s: no Pub/Sub topic configured", ctx.org_id)
            return {}
        resp = await client.watch(topic)
        return build_watch_state(resp, topic=topic)

    async def _fetch_and_normalize(
        self, ctx: SignalContext, client: GmailClient, message_ids: list[str]
    ) -> list[SignalEvent]:
        if not message_ids:
            return []
        raw_messages = await client.get_messages(message_ids)
        events: list[SignalEvent] = []
        for raw in raw_messages:
            parsed = parse_message(raw)
            if not parsed.get("message_id"):
                continue
            events.append(self._to_event(ctx, parsed))
        return events

    @staticmethod
    def _to_event(ctx: SignalContext, parsed: dict[str, Any]) -> SignalEvent:
        when = datetime.now(timezone.utc)
        if parsed.get("internal_date"):
            try:
                when = datetime.fromtimestamp(int(parsed["internal_date"]) / 1000, tz=timezone.utc)
            except (TypeError, ValueError):
                pass
        elif parsed.get("date"):
            try:
                when = parsedate_to_datetime(parsed["date"])
            except (TypeError, ValueError):
                pass
        # A received email is a single IMMUTABLE fact, so the version is constant. Versioning
        # by the message's historyId was non-deterministic: historyId advances on later label
        # changes, so the SAME email re-fetched on a full-resync (404 recovery / re-onboard)
        # yielded a DIFFERENT external_version -> a DIFFERENT event_id -> a duplicate
        # EMAIL_RECEIVED signal. A constant version makes (org, gmail, message_id) the stable
        # dedupe identity: exactly one signal per email, idempotent across any number of syncs.
        version = "1"
        # Seal sensitive content at rest BEFORE it enters the payload (and thus the DB,
        # backups, and replay stream). event_id excludes payload, so this never affects
        # determinism/idempotency.
        _encrypt_content(parsed)
        return SignalEvent.create(
            ctx,
            source=SignalSource.GMAIL,
            event_type=SignalEventType.EMAIL_RECEIVED,
            external_id=parsed["message_id"],
            external_version=version,
            timestamp=when,
            payload=parsed,
            metadata={
                "thread_id": parsed.get("thread_id"),
                "label_ids": parsed.get("label_ids", []),
                # Security boundary: an email body is authored by an EXTERNAL party, so it is
                # attacker-controllable ("Ignore all previous instructions..."). We stamp
                # provenance + the one hard gate at capture, where origin is known with
                # certainty. content may be REASONED over (that is the product — extracting
                # commitments/decisions), but its embedded text must NEVER be EXECUTED as a
                # command by any future agent. So we forbid execution, not reasoning.
                "source_type": "external_email",
                "content_trust_level": "untrusted",
                "trusted_for_execution": False,
            },
        )
