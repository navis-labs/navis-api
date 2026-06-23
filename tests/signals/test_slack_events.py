"""Slack mapper: identity, determinism (webhook==backfill), trust boundary, echo-loop,
visibility/ACL metadata, governance flags, and PII-minimized + encrypted content."""
from __future__ import annotations

from navis.signals.base import SignalEventType, SignalSource
from navis.signals.slack.events import (
    channel_snapshot_event,
    event_from_slack,
    events_from_envelope,
    user_snapshot_event,
)
from tests.signals.fakes import make_ctx

TEAM = "T123"


def _msg(**over):
    e = {"type": "message", "channel": "C999", "user": "U1", "ts": "1700000000.001",
         "text": "ship it friday"}
    e.update(over)
    return e


def test_message_maps_to_signal_with_identity_and_trust():
    ctx = make_ctx()
    ev = event_from_slack(ctx, TEAM, _msg())
    assert ev is not None
    assert ev.source == SignalSource.SLACK and ev.event_type == SignalEventType.MESSAGE_SENT
    assert ev.external_id == "C999:1700000000.001" and ev.external_version == "1"
    # mandatory security + visibility metadata
    m = ev.metadata
    assert m["source_type"] == "external_chat"
    assert m["content_trust_level"] == "untrusted"
    assert m["trusted_for_execution"] is False
    assert m["workspace_id"] == TEAM and m["channel_id"] == "C999"
    assert m["channel_visibility"] == "public" and m["source_origin"] == "human"
    assert m["retention_class"] == "slack_message"


def test_text_is_encrypted_and_no_pii_denormalized():
    ctx = make_ctx()
    ev = event_from_slack(ctx, TEAM, _msg())
    assert ev is not None
    assert ev.payload["slack_user_id"] == "U1"          # author is a REFERENCE, not a name
    assert ev.payload["text"] != "ship it friday"        # sealed at rest
    assert ev.payload["_content_fields"] == ["text"]     # memory decrypt seam
    assert "ship it friday" not in str(ev.payload)
    assert "email" not in ev.payload and "real_name" not in ev.payload


def test_webhook_and_backfill_produce_identical_event_id():
    # Same logical message via the envelope (webhook) and a synthesized event (backfill) ->
    # identical event_id, so overlapping ingestion dedupes to one fact.
    ctx = make_ctx()
    envelope = {"type": "event_callback", "team_id": TEAM, "event_id": "Ev1", "event": _msg()}
    from_webhook = events_from_envelope(ctx, envelope)[0]
    from_backfill = event_from_slack(ctx, TEAM, _msg())
    assert from_backfill is not None
    assert from_webhook.event_id == from_backfill.event_id


def test_thread_reply_classified_and_carries_thread_lineage():
    ctx = make_ctx()
    ev = event_from_slack(ctx, TEAM, _msg(thread_ts="1699999999.000"))
    assert ev is not None and ev.event_type == SignalEventType.THREAD_REPLY
    assert ev.payload["thread_root_ts"] == "1699999999.000"


def test_private_channel_visibility_and_sensitivity():
    ctx = make_ctx()
    ev = event_from_slack(ctx, TEAM, _msg(channel="G777"))  # G prefix = private
    assert ev is not None
    assert ev.metadata["channel_visibility"] == "private"
    assert ev.metadata["contains_sensitive_content"] is True


def test_bot_message_tagged_for_echo_loop():
    ctx = make_ctx()
    ev = event_from_slack(ctx, TEAM, _msg(bot_id="B1", user=None))
    assert ev is not None and ev.metadata["source_origin"] == "bot"
    # CR-1: match on bot_id (B…) OR bot_user_id (U…) — a Slack message carries bot_id, but our
    # install knows bot_user_id; missing either re-feeds Navis output into Memory.
    by_bot_id = event_from_slack(ctx, TEAM, _msg(bot_id="BNAVIS"), our_bot_id="BNAVIS")
    assert by_bot_id is not None and by_bot_id.metadata["source_origin"] == "navis_bot"
    by_user_id = event_from_slack(ctx, TEAM, _msg(user="UNAVIS", bot_id="B9"), our_bot_user_id="UNAVIS")
    assert by_user_id is not None and by_user_id.metadata["source_origin"] == "navis_bot"


def test_reaction_maps_with_composite_identity():
    ctx = make_ctx()
    ev = event_from_slack(ctx, TEAM, {
        "type": "reaction_added", "user": "U2", "reaction": "tada",
        "item": {"channel": "C999", "ts": "1700000000.001"},
    })
    assert ev is not None and ev.event_type == SignalEventType.REACTION_ADDED
    assert ev.external_id == "C999:1700000000.001:tada:U2"


def test_unmapped_event_is_skipped_and_metered(monkeypatch):
    from navis.platform.monitoring.metrics import metrics

    seen = []
    monkeypatch.setattr(metrics, "incr", lambda name, **k: seen.append((name, k.get("labels"))))
    assert event_from_slack(make_ctx(), TEAM, {"type": "dnd_updated"}) is None
    assert ("slack.mapper.unhandled_event", {"type": "dnd_updated", "subtype": "None"}) in seen


def test_channel_archive_and_unarchive_both_captured():
    ctx = make_ctx()
    arch = event_from_slack(ctx, TEAM, {"type": "channel_archive", "channel": "C1", "event_ts": "100"})
    unarch = event_from_slack(ctx, TEAM, {"type": "channel_unarchive", "channel": "C1", "event_ts": "200"})
    assert arch is not None and arch.event_type == SignalEventType.CHANNEL_ARCHIVED
    assert unarch is not None and unarch.event_type == SignalEventType.CHANNEL_UNARCHIVED
    assert arch.event_id != unarch.event_id  # distinct facts -> archive->unarchive replays


def test_file_shared_captures_metadata_no_urls_title_sealed():
    ctx = make_ctx()
    ev = event_from_slack(ctx, TEAM, {
        "type": "file_shared", "channel_id": "C1", "event_ts": "100",
        "file": {"id": "F1", "title": "Layoffs.pdf", "mimetype": "application/pdf",
                 "filetype": "pdf", "size": 1024, "url_private": "https://secret"},
    })
    assert ev is not None and ev.event_type == SignalEventType.FILE_SHARED
    p = ev.payload
    assert p["file_id"] == "F1" and p["mimetype"] == "application/pdf" and p["size"] == 1024
    assert "Layoffs.pdf" not in str(p)          # title sealed
    assert "url_private" not in p and "https://secret" not in str(p)  # never store URLs


def test_app_mention_captures_addressee_and_app():
    ctx = make_ctx()
    envelope = {
        "type": "event_callback", "team_id": TEAM, "event_id": "Ev1", "api_app_id": "A99",
        "event": {"type": "app_mention", "channel": "C1", "user": "U1", "ts": "1.1", "text": "<@B> hi"},
    }
    ev = events_from_envelope(ctx, envelope, our_bot_user_id="UBOT")[0]
    assert ev.event_type == SignalEventType.APP_MENTION
    assert ev.payload["mentioned_bot_user_id"] == "UBOT" and ev.payload["app_id"] == "A99"


def test_identity_confidence_on_snapshots():
    ctx = make_ctx()
    internal = user_snapshot_event(ctx, TEAM, {"id": "U1", "team_id": TEAM, "profile": {}}, our_team_id=TEAM)
    external = user_snapshot_event(ctx, TEAM, {"id": "U2", "team_id": "T_X", "profile": {}}, our_team_id=TEAM)
    assert internal.payload["identity_confidence"] == "internal"
    assert external.payload["identity_confidence"] == "external"


def test_internal_user_snapshot_seals_pii():
    ctx = make_ctx()
    ev = user_snapshot_event(ctx, TEAM, {
        "id": "U1", "updated": 1700, "team_id": TEAM,
        "profile": {"email": "a@x.com", "display_name": "Alice"},
    }, our_team_id=TEAM)
    assert ev.event_type == SignalEventType.SLACK_USER_SNAPSHOT
    assert ev.payload["is_internal"] is True and ev.metadata["contains_pii"] is True
    assert "a@x.com" not in str(ev.payload) and "Alice" not in str(ev.payload)  # sealed
    assert set(ev.payload["_content_fields"]) == {"email", "display_name"}


def test_external_user_snapshot_drops_pii():
    # CR-2 fail-closed: Slack Connect / guest / other-workspace -> store ONLY the reference.
    ctx = make_ctx()
    for user in (
        {"id": "U2", "team_id": "T_OTHER", "profile": {"email": "ext@other.com"}},   # other workspace
        {"id": "U3", "team_id": TEAM, "is_restricted": True, "profile": {"email": "guest@x.com"}},  # guest
    ):
        ev = user_snapshot_event(ctx, TEAM, user, our_team_id=TEAM)
        assert ev.payload["is_internal"] is False
        assert ev.metadata["contains_pii"] is False
        assert "email" not in ev.payload and "_content_fields" not in ev.payload
        assert "other.com" not in str(ev.payload) and "guest@x.com" not in str(ev.payload)


def test_backfill_message_tagged_and_id_matches_realtime():
    from navis.signals.slack.backfill import history_message_to_event

    ctx = make_ctx()
    history_msg = {"ts": "1700000000.001", "user": "U1", "text": "hi", "channel_unused": True}
    bf = history_message_to_event(ctx, TEAM, "C999", history_msg)
    rt = event_from_slack(ctx, TEAM, _msg(text="hi"))
    assert bf is not None and rt is not None
    assert bf.event_id == rt.event_id                       # identical identity across paths
    assert bf.metadata["capture_origin"] == "backfill"
    assert bf.metadata["may_reflect_edits"] is True
    assert rt.metadata["capture_origin"] == "realtime"


def test_channel_snapshot_seals_name():
    ctx = make_ctx()
    ev = channel_snapshot_event(ctx, TEAM, {"id": "C1", "name": "project-zeus", "is_private": True})
    assert ev.event_type == SignalEventType.SLACK_CHANNEL_SNAPSHOT
    assert ev.metadata["channel_visibility"] == "private"
    assert "project-zeus" not in str(ev.payload)
