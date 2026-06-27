"""Deterministic extractors over Slack SignalEvents.

Slack is an organizational event stream: messages, threads, channels, people, and the
relationships between them. Every extractor is pure (same signal → same drafts) and reads
ONLY the captured payload — message text is already decrypted by the Memory read seam
(SignalViewFactory). Commitment/Decision detection reuses the exact email regex so the rule
is identical across chat + mail. No interpretation beyond deterministic pattern capture.
"""
from __future__ import annotations

from navis.memory.envelope import EntityType
from navis.memory.extractors.base import Extractor, MemoryDraft, SignalView
from navis.memory.extractors.email import _COMMITMENT_RE, _DECISION_RE, _fact_key, _sentences
from navis.memory.schemas import current_version
from navis.memory.confidence import HEURISTIC, HIGH, LOW, STRUCTURAL
from navis.signals.base import SignalEventType, SignalSource

_SLACK = {SignalSource.SLACK}
_MESSAGE_TYPES = {
    SignalEventType.MESSAGE_SENT,
    SignalEventType.THREAD_REPLY,
    SignalEventType.APP_MENTION,
}


class SlackPersonExtractor(Extractor):
    name = "slack_person"
    version = "1.0.0"
    sources = _SLACK

    def extract(self, signal: SignalView) -> list[MemoryDraft]:
        p = signal.payload
        ver = current_version(EntityType.PERSON)
        # Identity snapshot: the authoritative slack_user_id -> name/email fact (internal only;
        # external users carry no PII per the signal-layer fail-closed rule).
        if signal.event_type == SignalEventType.SLACK_USER_SNAPSHOT:
            uid = p.get("slack_user_id")
            if not uid:
                return []
            return [MemoryDraft(
                entity_type=EntityType.PERSON, entity_natural_key=str(uid),
                dedup_key=f"slack_user:{uid}",
                payload={"slack_user_id": uid, "email": p.get("email"),
                         "name": p.get("display_name"), "role": "slack_member",
                         "source": "slack", "is_internal": p.get("is_internal")},
                confidence=HIGH, schema_version=ver,
            )]
        # Authorship: a message author is observed as a person (keyed by slack_user_id).
        if signal.event_type in _MESSAGE_TYPES:
            uid = p.get("slack_user_id")
            if not uid:
                return []
            return [MemoryDraft(
                entity_type=EntityType.PERSON, entity_natural_key=str(uid),
                dedup_key=f"slack_author:{uid}",
                payload={"slack_user_id": uid, "email": None, "name": None,
                         "role": "slack_member", "source": "slack"},
                confidence=HIGH, schema_version=ver,
            )]
        return []


class SlackChannelExtractor(Extractor):
    name = "slack_channel"
    version = "1.0.0"
    sources = _SLACK

    def extract(self, signal: SignalView) -> list[MemoryDraft]:
        p = signal.payload
        cid = p.get("channel_id")
        if not cid:
            return []
        ver = current_version(EntityType.CHANNEL)
        # Snapshot/lifecycle events carry name + visibility in the payload; messages carry only
        # the id (channel_visibility lives in signal metadata, which extractors don't receive).
        return [MemoryDraft(
            entity_type=EntityType.CHANNEL, entity_natural_key=str(cid),
            dedup_key=f"channel:{cid}:{signal.external_version}",
            payload={"channel_id": cid, "name": p.get("name"),
                     "visibility": p.get("visibility"), "workspace_id": p.get("workspace_id")},
            confidence=STRUCTURAL, schema_version=ver,
        )]


class SlackThreadExtractor(Extractor):
    name = "slack_thread"
    version = "1.0.0"
    sources = _SLACK

    def extract(self, signal: SignalView) -> list[MemoryDraft]:
        if signal.event_type not in _MESSAGE_TYPES:
            return []
        p = signal.payload
        root = p.get("thread_root_ts")
        cid = p.get("channel_id")
        if not root or not cid:
            return []
        return [MemoryDraft(
            entity_type=EntityType.THREAD, entity_natural_key=f"{cid}:{root}",
            dedup_key=f"slack_thread:{cid}:{root}",
            payload={"thread_id": f"{cid}:{root}", "channel_id": cid, "subject": None,
                     "message_id": signal.external_id},
            confidence=STRUCTURAL, schema_version=current_version(EntityType.THREAD),
        )]


class SlackRelationshipExtractor(Extractor):
    name = "slack_relationship"
    version = "1.0.0"
    sources = _SLACK

    def extract(self, signal: SignalView) -> list[MemoryDraft]:
        if signal.event_type not in _MESSAGE_TYPES:
            return []
        p = signal.payload
        uid, cid = p.get("slack_user_id"), p.get("channel_id")
        if not uid or not cid:
            return []
        # An OBSERVED interaction (person active in channel) — no health/score.
        return [MemoryDraft(
            entity_type=EntityType.RELATIONSHIP, entity_natural_key=str(uid),
            dedup_key=f"slack:{uid}:{cid}",
            payload={"slack_user_id": uid, "channel_id": cid, "person_email": None,
                     "direction": "active_in", "via": "slack"},
            confidence=HIGH, schema_version=current_version(EntityType.RELATIONSHIP),
        )]


class SlackCommitmentExtractor(Extractor):
    name = "slack_commitment"
    version = "1.0.0"
    sources = _SLACK

    def extract(self, signal: SignalView) -> list[MemoryDraft]:
        if signal.event_type not in _MESSAGE_TYPES:
            return []
        body = signal.payload.get("text") or ""
        if not body:
            return []
        ver = current_version(EntityType.COMMITMENT)
        drafts = []
        for idx, sentence in enumerate(_sentences(body)):
            if 8 <= len(sentence) <= 400 and _COMMITMENT_RE.search(sentence):
                drafts.append(MemoryDraft(
                    entity_type=EntityType.COMMITMENT,
                    entity_natural_key=f"{signal.external_id}:{_fact_key(sentence)}",
                    dedup_key=_fact_key(sentence),
                    payload={"text": sentence, "speaker_slack_user_id": signal.payload.get("slack_user_id"),
                             "speaker_email": None, "channel_id": signal.payload.get("channel_id"),
                             "thread_id": signal.payload.get("thread_root_ts"), "via": "slack"},
                    confidence=HEURISTIC, schema_version=ver,
                ))
                if len(drafts) >= 5:
                    break
        return drafts


class SlackDecisionExtractor(Extractor):
    name = "slack_decision"
    version = "1.0.0"
    sources = _SLACK

    def extract(self, signal: SignalView) -> list[MemoryDraft]:
        if signal.event_type not in _MESSAGE_TYPES:
            return []
        body = signal.payload.get("text") or ""
        if not body:
            return []
        ver = current_version(EntityType.DECISION)
        drafts = []
        for idx, sentence in enumerate(_sentences(body)):
            if 8 <= len(sentence) <= 400 and _DECISION_RE.search(sentence):
                drafts.append(MemoryDraft(
                    entity_type=EntityType.DECISION,
                    entity_natural_key=f"{signal.external_id}:{_fact_key(sentence)}",
                    dedup_key=f"d:{_fact_key(sentence)}",
                    payload={"text": sentence, "channel_id": signal.payload.get("channel_id"),
                             "subject": None, "via": "slack"},
                    confidence=LOW, schema_version=ver,
                ))
                if len(drafts) >= 5:
                    break
        return drafts
