"""Per-entity payload schema versions.

Each entity_type's `payload` has a `schema_version`. When a payload shape changes,
bump the version here and register an upcaster (see `memory.upcasting`) — old
records keep their stored version and are upcast on read, so v1/v2/v3 coexist and
replay never breaks.

The payload field documentation below is the contract State will read against.
"""
from __future__ import annotations

from navis.memory.envelope import EntityType

# The version a NEWLY extracted record of each type is written with.
CURRENT_SCHEMA_VERSION: dict[EntityType, int] = {
    EntityType.PERSON: 1,
    EntityType.COMPANY: 1,
    EntityType.CUSTOMER: 1,
    EntityType.PROJECT: 1,
    EntityType.MEETING: 1,
    EntityType.THREAD: 1,
    EntityType.RELATIONSHIP: 1,
    EntityType.COMMITMENT: 1,
    EntityType.DECISION: 1,
    EntityType.CHANNEL: 1,
    EntityType.DOCUMENT: 1,
}

# Human-readable payload contracts. Fields beyond the original v1 set are ADDITIVE optional
# keys (backward compatible — old records simply lack them; State reads via .get()), so they
# do NOT bump the schema version. Breaking shape changes still bump + register an upcaster.
PAYLOAD_FIELDS_V1: dict[EntityType, tuple[str, ...]] = {
    # `is_automated` / `category` are ADDITIVE (no version bump) and SENDER-SCOPED:
    # they are written ONLY on the fact whose `role` is "from", because everything
    # the classifier reads (List-Id, Precedence, Auto-Submitted, the Gmail tab
    # label, a no-reply local-part) describes how that address MAILS and says
    # nothing about who it was addressed to. Absent means "never observed as a
    # sender" — which is not the same as "human", and readers must not treat it as
    # a claim either way. Same key names and same value domain as THREAD below, so
    # one classifier serves both. See extractors/email.py:sender_classification.
    EntityType.PERSON: ("email", "name", "role", "source", "slack_user_id",
                        "is_automated", "category"),
    # COMPANY deliberately carries NEITHER: it is keyed by DOMAIN, and one domain
    # legitimately hosts both human and automated senders, so a latest-wins flag
    # there would track whichever mailed last rather than any fact about the
    # company. See the note on CompanyExtractor.
    EntityType.COMPANY: ("domain", "name"),
    EntityType.CUSTOMER: ("external_ref", "name"),  # populated from a CRM signal source
    EntityType.PROJECT: ("folder_id", "name", "owners", "shared", "mime_type", "drive_id"),
    EntityType.MEETING: ("title", "starts_at", "ends_at", "attendees", "status",
                         "has_conference", "organizer", "is_recurring", "conference_id"),
    # participants / snippet / category / is_automated are ADDITIVE (see the note
    # above — no version bump). They exist so a thread fact has something
    # semantically meaningful to embed; before them the only embeddable field was
    # the opaque thread_id. `category` is one of: personal | promotional | social |
    # forum | update | automated | spam.
    # `direction` / `from_email` / `from_name` are ADDITIVE (no version bump).
    # They are what makes "is someone waiting on a reply?" answerable WITHOUT an
    # LLM: State folds thread records in ascending time order with latest-wins,
    # so the surviving `direction` is simply the direction of the newest message
    # in the thread. inbound => the other party spoke last => nobody has replied.
    EntityType.THREAD: ("thread_id", "subject", "message_id",
                        "rfc_message_id", "in_reply_to", "references", "channel_id",
                        "participants", "snippet", "category", "is_automated",
                        "direction", "from_email", "from_name",
                        # STICKY thread markers (see extractors/email.py):
                        # somebody claimed delivery / recorded a decision at
                        # some point in this thread's history. Null is never
                        # written, so State's latest-non-null fold makes them
                        # persist once set — which is what lets a follow-up be
                        # suppressed by a LATER message.
                        "fulfilment_at", "fulfilment_from",
                        "resolution_at", "resolution_from"),
    # `is_automated` / `category` are ADDITIVE and SENDER-SCOPED — identical
    # contract to PERSON above. Written only when this address was the `from` of
    # the message (i.e. on the "inbound" fact), because that is the only party the
    # classification describes. This is what lets a relationship be told apart
    # from a newsletter blast without a second classifier; Slack-derived
    # relationships have no equivalent notion and simply omit both keys.
    EntityType.RELATIONSHIP: ("person_email", "direction", "via", "slack_user_id", "channel_id",
                              "is_automated", "category"),
    # due_at / due_text / promised_at are ADDITIVE. `due_at` is present ONLY
    # when the sentence stated a deadline — a null means "no date given", never
    # "we could not parse one", so nothing downstream may invent it.
    EntityType.COMMITMENT: ("text", "speaker_email", "speaker_name",
                            "speaker_slack_user_id", "subject",
                            "thread_id", "channel_id", "via",
                            "due_at", "due_text", "promised_at"),
    # `kind` discriminates a decision RECORDED ("we decided to raise") from one
    # REQUESTED of the founder ("can you approve the budget?"). One entity type,
    # one history per decision — not a second decision store.
    EntityType.DECISION: ("text", "subject", "channel_id", "via",
                          "kind", "asked_by", "asked_by_name", "asked_at",
                          "thread_id", "rfc_message_id"),
    EntityType.CHANNEL: ("channel_id", "name", "visibility", "workspace_id"),
    # `status` is ADDITIVE (no version bump, per the note above): "active" | "trashed".
    # Always written, never null — State's fold skips nulls, so an omitted status
    # would leave a stale "trashed" as the surviving value.
    EntityType.DOCUMENT: ("document_id", "title", "owners", "parents", "drive_id",
                          "source", "status"),
}


def current_version(entity_type: EntityType) -> int:
    return CURRENT_SCHEMA_VERSION[entity_type]
