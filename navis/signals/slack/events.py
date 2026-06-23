"""Map Slack's wire format to the shared SignalEvent contract — reality capture only.

This is the ONE mapper used by BOTH ingestion paths (real-time Events API webhook and
onboarding REST backfill), so a message captured live and the same message captured by
backfill produce an IDENTICAL event_id (content-derived identity, never Slack's delivery id).

Laws enforced here:
  * No interpretation. We record the verb that happened, never a summary/classification of
    its content. `contains_pii` / `contains_sensitive_content` are STRUCTURAL flags derived
    from event type + channel visibility — never from reading the text.
  * Prompt-injection boundary: every signal is external_chat / untrusted / not-for-execution.
  * Privacy: message text and channel/user names are ENCRYPTED at rest (reused field_cipher,
    self-described via `_content_fields` so Memory's read seam decrypts transparently). Author
    identity is a `slack_user_id` reference, resolved out-of-band via SLACK_USER_SNAPSHOT —
    never denormalized as email/real-name into every message row.
  * Echo-loop: bot-authored messages are tagged so Memory can refuse to feed Navis's own
    output back in as organizational reality.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from navis.platform.monitoring.metrics import metrics
from navis.platform.security import field_cipher
from navis.signals.base import (
    SignalContext,
    SignalEvent,
    SignalEventType,
    SignalSource,
)


def _file_meta(f: dict[str, Any]) -> dict[str, Any]:
    """Minimal, replay-safe file reality: id + type/size, with the title ENCRYPTED (a filename
    leaks topic, like Drive/Gmail attachments). NEVER bytes, NEVER download/private URLs."""
    return {
        "file_id": f.get("id"),
        "title": field_cipher.encrypt(f.get("title") or f.get("name")),
        "mimetype": f.get("mimetype"),
        "filetype": f.get("filetype"),
        "size": f.get("size"),
    }

# Slack channel id prefixes when channel_type is absent: C=public, G=private(group), D=im.
_VIS_BY_PREFIX = {"C": "public", "G": "private", "D": "im"}
_VIS_BY_TYPE = {"channel": "public", "group": "private", "private_channel": "private",
                "im": "im", "mpim": "mpim", "public_channel": "public"}
_SENSITIVE_VIS = {"private", "im", "mpim", "unknown"}


def _visibility(channel_id: str | None, channel_type: str | None) -> str:
    if channel_type and channel_type in _VIS_BY_TYPE:
        return _VIS_BY_TYPE[channel_type]
    if channel_id:
        return _VIS_BY_PREFIX.get(channel_id[0], "unknown")
    return "unknown"


def _origin(event: dict[str, Any], our_bot_id: str | None, our_bot_user_id: str | None) -> str:
    inner = event.get("message") or {}
    bot_id = event.get("bot_id") or inner.get("bot_id")
    author = event.get("user") or inner.get("user")
    # CR-1: a Slack message carries `bot_id` (B…), but our install id is `bot_user_id` (U…).
    # Match on EITHER so Navis's own posts are caught — a miss here re-feeds our output into
    # Memory as reality (echo loop). Fail toward "navis_bot" when either identifier matches.
    if (our_bot_id and bot_id == our_bot_id) or (our_bot_user_id and author == our_bot_user_id):
        return "navis_bot"
    if not bot_id:
        return "human"
    return "bot"  # third-party bot — still organizational reality


def _ts_now(ts: str | None) -> datetime:
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc) if ts else datetime.now(timezone.utc)
    except (TypeError, ValueError):
        return datetime.now(timezone.utc)


def _seal(payload: dict[str, Any], *fields: str) -> dict[str, Any]:
    """Encrypt the named content fields in place + self-describe for the Memory decrypt seam."""
    sealed = [f for f in fields if payload.get(f) is not None]
    for f in sealed:
        payload[f] = field_cipher.encrypt(payload[f])
    if sealed:
        payload["_content_fields"] = sealed
    return payload


def _meta(
    *, team_id: str, channel_id: str | None, visibility: str, origin: str,
    retention_class: str, contains_pii: bool, capture_origin: str = "realtime",
) -> dict[str, Any]:
    meta = {
        # Prompt-injection boundary — mandatory on every Slack signal.
        "source_type": "external_chat",
        "content_trust_level": "untrusted",
        "trusted_for_execution": False,
        # Echo-loop + reality provenance.
        "source_origin": origin,
        # Capture provenance: realtime is authoritative; backfill may reflect later edits.
        "capture_origin": capture_origin,
        # Visibility / ACL — never dropped; Memory enforces channel ACLs from these.
        "workspace_id": team_id,
        "channel_id": channel_id,
        "channel_visibility": visibility,
        "visibility_scope": visibility,
        # Governance — STRUCTURAL (event type + visibility), not content analysis.
        "retention_class": retention_class,
        "contains_pii": contains_pii,
        "contains_sensitive_content": visibility in _SENSITIVE_VIS,
    }
    if capture_origin == "backfill":
        meta["may_reflect_edits"] = True  # decision #3: history shows current, not original, text
    return meta


def event_from_slack(
    ctx: SignalContext, team_id: str, event: dict[str, Any], *,
    event_ts: str | None = None, our_bot_id: str | None = None,
    our_bot_user_id: str | None = None, capture_origin: str = "realtime",
    api_app_id: str | None = None,
) -> SignalEvent | None:
    """Map ONE inner Slack event dict to ONE SignalEvent, or None if it's not a captured verb.
    Shared by webhook (envelope['event']) and backfill (synthesized event dicts)."""
    etype = event.get("type")
    subtype = event.get("subtype")
    ev_ts = event_ts or event.get("event_ts") or event.get("ts")
    origin = _origin(event, our_bot_id, our_bot_user_id)

    def build(signal_type, external_id, version, *, channel_id, payload, retention, pii, ts):
        vis = _visibility(channel_id, event.get("channel_type"))
        return SignalEvent.create(
            ctx,
            source=SignalSource.SLACK,
            event_type=signal_type,
            external_id=external_id,
            external_version=version,
            timestamp=_ts_now(ts),
            payload=payload,
            metadata=_meta(team_id=team_id, channel_id=channel_id, visibility=vis,
                           origin=origin, retention_class=retention, contains_pii=pii,
                           capture_origin=capture_origin),
        )

    # ---- messages ----
    if etype in ("message", "app_mention"):
        if subtype == "message_deleted":
            ch = event.get("channel")
            dts = event.get("deleted_ts")
            return build(SignalEventType.MESSAGE_DELETED, f"{ch}:{dts}", f"del:{ev_ts}",
                         channel_id=ch, payload={"channel_id": ch, "ts": dts}, retention="slack_message",
                         pii=False, ts=ev_ts)
        if subtype == "message_changed":
            msg = event.get("message") or {}
            ch = event.get("channel")
            ts = msg.get("ts")
            ver = (msg.get("edited") or {}).get("ts") or ev_ts
            payload = _seal({
                "channel_id": ch, "slack_user_id": msg.get("user"), "ts": ts,
                "thread_root_ts": msg.get("thread_ts"), "parent_message_ts": msg.get("thread_ts"),
                "text": msg.get("text"), "subtype": "message_changed",
            }, "text")
            return build(SignalEventType.MESSAGE_EDITED, f"{ch}:{ts}", str(ver),
                         channel_id=ch, payload=payload, retention="slack_message",
                         pii=_visibility(ch, None) in _SENSITIVE_VIS, ts=ts)
        if subtype == "channel_join":
            ch, u = event.get("channel"), event.get("user")
            return build(SignalEventType.USER_JOINED_CHANNEL, f"{ch}:{u}", str(ev_ts),
                         channel_id=ch, payload={"channel_id": ch, "slack_user_id": u},
                         retention="slack_channel_event", pii=False, ts=ev_ts)
        if subtype == "channel_leave":
            ch, u = event.get("channel"), event.get("user")
            return build(SignalEventType.USER_LEFT_CHANNEL, f"{ch}:{u}", str(ev_ts),
                         channel_id=ch, payload={"channel_id": ch, "slack_user_id": u},
                         retention="slack_channel_event", pii=False, ts=ev_ts)
        # plain message / app_mention / bot_message / file_share-on-message
        ch, ts = event.get("channel"), event.get("ts")
        thread_ts = event.get("thread_ts")
        is_reply = bool(thread_ts and thread_ts != ts)
        sig = (SignalEventType.APP_MENTION if etype == "app_mention"
               else SignalEventType.THREAD_REPLY if is_reply else SignalEventType.MESSAGE_SENT)
        files = [_file_meta(f) for f in event.get("files", []) if f.get("id")]
        data: dict[str, Any] = {
            "channel_id": ch, "slack_user_id": event.get("user"), "ts": ts,
            "thread_root_ts": thread_ts, "parent_message_ts": thread_ts,
            "text": event.get("text"), "files": files or None, "subtype": subtype,
        }
        if etype == "app_mention":
            # Who was addressed (our bot) + which app — so Memory/agents needn't reparse text.
            data["mentioned_bot_user_id"] = our_bot_user_id
            data["app_id"] = api_app_id
        payload = _seal(data, "text")
        return build(sig, f"{ch}:{ts}", "1", channel_id=ch, payload=payload,
                     retention="slack_message", pii=_visibility(ch, None) in _SENSITIVE_VIS, ts=ts)

    # ---- reactions ----
    if etype in ("reaction_added", "reaction_removed"):
        item = event.get("item") or {}
        ch, ts = item.get("channel"), item.get("ts")
        reaction, u = event.get("reaction"), event.get("user")
        added = etype == "reaction_added"
        return build(
            SignalEventType.REACTION_ADDED if added else SignalEventType.REACTION_REMOVED,
            f"{ch}:{ts}:{reaction}:{u}", "add" if added else f"rm:{ev_ts}",
            channel_id=ch, payload={"channel_id": ch, "ts": ts, "reaction": reaction, "slack_user_id": u},
            retention="slack_reaction", pii=False, ts=ev_ts,
        )

    # ---- channel lifecycle ----
    if etype == "channel_created":
        ch = event.get("channel") or {}
        cid = ch.get("id")
        payload = _seal({"channel_id": cid, "name": ch.get("name"), "created": ch.get("created")}, "name")
        return build(SignalEventType.CHANNEL_CREATED, cid, str(ev_ts), channel_id=cid,
                     payload=payload, retention="slack_channel_event", pii=False, ts=ev_ts)
    if etype in ("channel_archive", "channel_unarchive"):
        # Both lifecycle transitions captured as distinct facts (different event_type, and a
        # distinct event_ts -> distinct identity), so archive->unarchive replays correctly.
        cid = event.get("channel")
        archived = etype == "channel_archive"
        return build(
            SignalEventType.CHANNEL_ARCHIVED if archived else SignalEventType.CHANNEL_UNARCHIVED,
            cid, str(ev_ts), channel_id=cid,
            payload={"channel_id": cid}, retention="slack_channel_event", pii=False, ts=ev_ts,
        )
    if etype == "channel_rename":
        ch = event.get("channel") or {}
        cid = ch.get("id")
        payload = _seal({"channel_id": cid, "name": ch.get("name")}, "name")
        return build(SignalEventType.CHANNEL_RENAMED, cid, str(ev_ts), channel_id=cid,
                     payload=payload, retention="slack_channel_event", pii=False, ts=ev_ts)

    # ---- membership ----
    if etype in ("member_joined_channel", "member_left_channel"):
        ch, u = event.get("channel"), event.get("user")
        joined = etype == "member_joined_channel"
        return build(
            SignalEventType.USER_JOINED_CHANNEL if joined else SignalEventType.USER_LEFT_CHANNEL,
            f"{ch}:{u}", str(ev_ts), channel_id=ch,
            payload={"channel_id": ch, "slack_user_id": u},
            retention="slack_channel_event", pii=False, ts=ev_ts,
        )

    # ---- file shared ----
    if etype == "file_shared":
        ch = event.get("channel_id") or event.get("channel")
        f = dict(event.get("file") or {})
        f.setdefault("id", event.get("file_id"))
        meta = _file_meta(f)
        fid = meta["file_id"]
        return build(SignalEventType.FILE_SHARED, f"{ch}:{fid}", "1", channel_id=ch,
                     payload={"channel_id": ch, **meta}, retention="slack_file", pii=False, ts=ev_ts)

    # Unhandled verb: metrics only (no persistence) so missing coverage / Slack API drift is
    # visible instead of a silent drop. Reality capture stays bounded to the declared set.
    metrics.incr("slack.mapper.unhandled_event", labels={"type": str(etype), "subtype": str(subtype)})
    return None


def events_from_envelope(
    ctx: SignalContext, envelope: dict[str, Any], *,
    our_bot_id: str | None = None, our_bot_user_id: str | None = None,
) -> list[SignalEvent]:
    """Webhook entry: an Events API `event_callback` envelope -> SignalEvent(s)."""
    event = envelope.get("event") or {}
    ev = event_from_slack(
        ctx, str(envelope.get("team_id")), event,
        event_ts=envelope.get("event_time") and str(envelope.get("event_time")),
        our_bot_id=our_bot_id, our_bot_user_id=our_bot_user_id,
        api_app_id=envelope.get("api_app_id"),
    )
    return [ev] if ev is not None else []


def _is_internal(user: dict[str, Any], our_team_id: str | None) -> bool:
    """CR-2: FAIL CLOSED. Internal only when provably so — same workspace and not a guest /
    single-channel guest / stranger / bot. Everyone else (incl. Slack Connect) is external and
    their email/name is NOT stored: holding third-party PII has no lawful basis."""
    if not our_team_id or user.get("team_id") != our_team_id:
        return False
    if user.get("is_restricted") or user.get("is_ultra_restricted") or user.get("is_stranger"):
        return False
    return True


def user_snapshot_event(
    ctx: SignalContext, team_id: str, user: dict[str, Any], *, our_team_id: str | None = None
) -> SignalEvent:
    """A periodic, replayable identity fact: slack_user_id -> (encrypted) email/name for INTERNAL
    users only. The one place identity PII lives, so messages stay PII-minimized."""
    internal = _is_internal(user, our_team_id or team_id)
    profile = user.get("profile") or {}
    payload: dict[str, Any] = {
        "slack_user_id": user.get("id"),
        "is_bot": bool(user.get("is_bot")),
        "deleted": bool(user.get("deleted")),
        "is_internal": internal,
        "identity_confidence": "internal" if internal else "external",
    }
    if internal:  # external/guest/Connect: store ONLY the opaque reference, drop all PII
        payload["email"] = profile.get("email")
        payload["display_name"] = profile.get("display_name") or profile.get("real_name")
        payload = _seal(payload, "email", "display_name")
    return SignalEvent.create(
        ctx, source=SignalSource.SLACK, event_type=SignalEventType.SLACK_USER_SNAPSHOT,
        external_id=str(user.get("id")), external_version=str(user.get("updated") or "1"),
        payload=payload,
        metadata=_meta(team_id=team_id, channel_id=None, visibility="n/a", origin="human",
                       retention_class="slack_identity", contains_pii=internal),
    )


def channel_snapshot_event(ctx: SignalContext, team_id: str, channel: dict[str, Any]) -> SignalEvent:
    """A periodic, replayable channel-existence fact (id, encrypted name, visibility)."""
    cid = channel.get("id")
    vis = ("private" if channel.get("is_private") else
           "im" if channel.get("is_im") else "mpim" if channel.get("is_mpim") else "public")
    payload = _seal({
        "channel_id": cid, "name": channel.get("name"), "visibility": vis,
        "is_archived": bool(channel.get("is_archived")),
    }, "name")
    return SignalEvent.create(
        ctx, source=SignalSource.SLACK, event_type=SignalEventType.SLACK_CHANNEL_SNAPSHOT,
        external_id=str(cid), external_version=str(channel.get("updated") or "1"),
        payload=payload,
        metadata=_meta(team_id=team_id, channel_id=cid, visibility=vis, origin="human",
                       retention_class="slack_identity", contains_pii=False),
    )
