"""Slack onboarding backfill — REST history into the SAME signal pipeline as real-time.

Events API only delivers post-subscription, so onboarding needs a one-shot REST sweep:
channel snapshots + user snapshots + per-channel message history. It does NOT fit the
poll-cursor WorkspaceConnector (Slack history is per-channel), so it is a bounded, RESUMABLE
job whose per-channel progress lives in Integration.sync_state["slack_backfill"].

Decisions enforced:
  * #2 NO reaction backfill — history carries only aggregate reaction counts (no per-event
    timestamp); fabricating REACTION_ADDED events would invent reality. The mapper ignores
    embedded reactions; we simply never synthesize them.
  * #3 every backfilled message is tagged capture_origin="backfill" (+ may_reflect_edits) —
    history shows current (possibly edited) text; real-time remains authoritative.
  * CR-4 the caller runs this under the existing integration advisory lock (single-flight) so
    the sync_state cursor read-modify-write can't be corrupted by a concurrent run.

Reuses: SlackClient, the events.py mapper + snapshot builders, SignalSyncService.ingest_events
(batch+fallback+DLQ), token_cipher, Integration. No new persistence/queue/dedupe.
"""
from __future__ import annotations

from typing import Any

from navis.platform.config import settings
from navis.platform.monitoring import get_logger
from navis.platform.monitoring.metrics import metrics
from navis.signals.base import SignalContext, SignalEvent, SignalSource
from navis.signals.slack.client import SlackAPIError, SlackClient
from navis.signals.slack.events import (
    channel_snapshot_event,
    event_from_slack,
    user_snapshot_event,
)

logger = get_logger("signals.slack.backfill")

# Bot can only read channels it belongs to; these errors mean "skip this channel", not fail.
_SKIP_CHANNEL_ERRORS = {"not_in_channel", "channel_not_found", "is_archived"}


def history_message_to_event(
    ctx: SignalContext, team_id: str, channel_id: str, msg: dict[str, Any], *,
    our_bot_id: str | None = None, our_bot_user_id: str | None = None,
) -> SignalEvent | None:
    """Pure: one conversations.history message -> one backfill SignalEvent (or None). History
    rows lack `type`/`channel`, so we synthesize the event shape the shared mapper expects —
    guaranteeing identical identity to the real-time path (minus reactions/edit-history)."""
    synthesized = {**msg, "type": "message", "channel": channel_id}
    return event_from_slack(
        ctx, team_id, synthesized, our_bot_id=our_bot_id, our_bot_user_id=our_bot_user_id,
        capture_origin="backfill",
    )


async def _ingest(db, ctx: SignalContext, events: list[SignalEvent]) -> int:
    if not events:
        return 0
    from navis.signals.sync import SignalSyncService

    outcome = await SignalSyncService(db).ingest_events(ctx, SignalSource.SLACK, events)
    return outcome.ingested


async def run_backfill(db, integration, client: SlackClient) -> dict[str, int]:
    """Run (or resume) the backfill for one Slack install. MUST be called under the
    integration advisory lock (CR-4). Returns capture counts."""
    ctx = SignalContext(integration.organization_id, integration.user_id, integration.id)
    team_id = integration.external_account
    slack_state = (integration.sync_state or {}).get("slack") or {}
    our_bot_id = slack_state.get("bot_id")
    our_bot_user_id = slack_state.get("bot_user_id")
    progress: dict[str, Any] = dict((integration.sync_state or {}).get("slack_backfill") or {})
    done_channels = set(progress.get("channels_done") or [])

    counts = {"channels": 0, "users": 0, "messages": 0, "skipped_channels": 0}

    # ---- channel snapshots (also the channel work-list) ----
    channels: list[dict[str, Any]] = []
    cursor = None
    while True:
        resp = await client.conversations_list(cursor=cursor)
        for ch in resp.get("channels", []):
            channels.append(ch)
        counts["channels"] += await _ingest(
            db, ctx, [channel_snapshot_event(ctx, team_id, ch) for ch in resp.get("channels", [])]
        )
        cursor = (resp.get("response_metadata") or {}).get("next_cursor") or None
        if not cursor:
            break

    # ---- user snapshots (CR-2 fail-closed handled in the builder) ----
    cursor = None
    while True:
        resp = await client.users_list(cursor=cursor)
        counts["users"] += await _ingest(db, ctx, [
            user_snapshot_event(ctx, team_id, u, our_team_id=team_id) for u in resp.get("members", [])
        ])
        cursor = (resp.get("response_metadata") or {}).get("next_cursor") or None
        if not cursor:
            break

    # ---- per-channel message history (bounded + resumable) ----
    max_pages = max(1, settings.slack_backfill_max_pages)
    cap = max(1, settings.slack_backfill_message_cap)
    for ch in channels:
        cid = ch.get("id")
        if not cid or cid in done_channels or ch.get("is_archived"):
            continue
        captured, pages, cursor = 0, 0, progress.get(f"cursor:{cid}")
        try:
            while captured < cap and pages < max_pages:
                resp = await client.conversations_history(cid, cursor=cursor)
                events = []
                for msg in resp.get("messages", []):
                    ev = history_message_to_event(ctx, team_id, cid, msg,
                                                  our_bot_id=our_bot_id, our_bot_user_id=our_bot_user_id)
                    if ev is not None:
                        events.append(ev)
                captured += await _ingest(db, ctx, events)
                pages += 1
                cursor = (resp.get("response_metadata") or {}).get("next_cursor") or None
                if not cursor:
                    break
        except SlackAPIError as exc:
            if exc.error in _SKIP_CHANNEL_ERRORS:
                counts["skipped_channels"] += 1
                logger.info("slack backfill skip channel=%s (%s)", cid, exc.error)
            else:
                raise  # genuine failure -> task retries; per-channel cursor already persisted
        counts["messages"] += captured
        done_channels.add(cid)
        progress.pop(f"cursor:{cid}", None)  # channel finished -> drop its resume cursor
        # Persist progress after each channel so a crash resumes here, not from zero.
        progress["channels_done"] = sorted(done_channels)
        integration.sync_state = {**(integration.sync_state or {}), "slack_backfill": progress}

    active = [c.get("id") for c in channels if not c.get("is_archived") and c.get("id")]
    if active and all(cid in done_channels for cid in active):
        progress["completed"] = True
        integration.sync_state = {**(integration.sync_state or {}), "slack_backfill": progress}

    metrics.incr("slack.backfill.messages", value=counts["messages"])
    logger.info("slack backfill org=%s %s", integration.organization_id, counts)
    return counts
