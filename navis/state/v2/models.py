"""State V2 read models — pure, deterministic, serializable.

The canonical internal form is `OrganizationalState.entities`: a nested dict
    {entity_type: {entity_id: <raw current-state dict>}}
which round-trips losslessly to/from JSON (so snapshots and incremental rehydration
are exact). Typed accessors (PersonState, CommitmentState, ...) are thin views over
that raw form for downstream consumers.

NOTHING here computes intelligence. No now(), no uuid4, no randomness.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, cast

from navis.memory.envelope import EntityType

# Bump when the projection LOGIC changes (forces rebuild); schema_version tracks
# the serialized shape. Both are stored on every snapshot — no destructive migrations.
PROJECTION_VERSION = 1
STATE_SCHEMA_VERSION = 1


# --- Ordering / watermark ----------------------------------------------------

def order_key(event_timestamp: datetime, memory_id: uuid.UUID) -> tuple[datetime, int]:
    """Total deterministic order: event_timestamp, then memory_id as tiebreak."""
    return (event_timestamp, memory_id.int)


@dataclass(frozen=True, slots=True)
class Watermark:
    last_timestamp: datetime | None = None
    last_memory_id: uuid.UUID | None = None

    def passed(self, event_timestamp: datetime, memory_id: uuid.UUID) -> bool:
        """True if (ts, memory_id) is strictly AFTER this watermark.

        The two fields are a PAIR — both set, or both None; the snapshot table
        enforces exactly that with `ck_state_snapshots_watermark_paired`.
        Testing both is what makes the pairing legible to a type checker, and
        it fails SAFE: a half-set cursor cannot say what has already been
        folded, so every record counts as unseen and the fold replays rather
        than silently skipping history. Testing only `last_timestamp` here
        would reach `None.int` inside `order_key` and raise AttributeError.
        """
        if self.last_timestamp is None or self.last_memory_id is None:
            return True
        return order_key(event_timestamp, memory_id) > order_key(
            self.last_timestamp, self.last_memory_id
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "last_timestamp": self.last_timestamp.isoformat() if self.last_timestamp else None,
            "last_memory_id": str(self.last_memory_id) if self.last_memory_id else None,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "Watermark":
        if not d:
            return cls()
        ts = d.get("last_timestamp")
        mid = d.get("last_memory_id")
        if not ts or not mid:
            # Half-set. `ck_state_snapshots_watermark_paired` guards the two
            # dedicated COLUMNS, but this parses the snapshot's `full_projection`
            # JSONB, which no constraint validates — so a legacy or hand-edited
            # blob can carry one half without the other. An incomplete cursor
            # cannot be trusted to say what has been folded: drop it and let the
            # projection rebuild from scratch.
            return cls()
        return cls(
            last_timestamp=datetime.fromisoformat(ts),
            last_memory_id=uuid.UUID(mid),
        )


# --- Typed entity views (read-only projections of the raw current state) ------

@dataclass(frozen=True, slots=True)
class _BaseView:
    entity_id: str
    first_seen: str | None
    last_seen: str | None
    fields: dict[str, Any]
    lists: dict[str, list]
    links: dict[str, list]
    provenance: dict[str, list]  # memory_ids, signal_ids


# Both accessors below are deliberately TRI-STATE (None = never observed as the
# sender of a message), unlike ThreadState's boolean coercion. A thread always
# has a newest message to classify; an address may only ever have appeared on
# `to`/`cc`, or predate the field entirely. Collapsing that "unknown" into False
# would assert "this is a human" on no evidence — the exact class of overclaim
# these fields exist to remove. Callers that want the permissive reading can
# still write `if x.is_automated:` (None is falsy, so an unknown surfaces rather
# than being silently suppressed); callers that need the distinction test
# `is None`.
class _SenderClassifiedView(_BaseView):
    """Mixin for entities that carry the sender classification (see
    memory/extractors/email.py:sender_classification). Written only from the
    `from` address, so it describes how THIS address mails — never who mailed
    it, and never whether it is a real party."""

    @property
    def is_automated(self):
        """True | False | None — None means "never observed as a sender"."""
        value = self.fields.get("is_automated")
        return None if value is None else bool(value)

    @property
    def category(self):
        """personal | promotional | social | forum | update | automated | spam,
        or None when never observed as a sender. `category != "personal"` is
        equivalent to `is_automated` by construction of `classify_email`; it is
        kept because it says WHICH kind of automation, which is what
        `engines/waiting.py` already gates on."""
        return self.fields.get("category")


class PersonState(_SenderClassifiedView):
    @property
    def email(self): return self.fields.get("email")
    @property
    def name(self): return self.fields.get("name")
    @property
    def role(self): return self.fields.get("role")
    @property
    def relationship_ids(self): return self.links.get("relationships", [])
    @property
    def channel_ids(self): return self.links.get("channels", [])
    @property
    def document_ids(self): return self.links.get("documents", [])


class CompanyState(_BaseView):
    @property
    def domain(self): return self.fields.get("domain")
    @property
    def person_ids(self): return self.links.get("people", [])


class CustomerState(_BaseView):
    @property
    def external_ref(self): return self.fields.get("external_ref")


class ProjectState(_BaseView):
    @property
    def folder_id(self): return self.fields.get("folder_id")
    @property
    def document_names(self): return self.lists.get("document_names", [])
    @property
    def document_ids(self): return self.links.get("documents", [])
    @property
    def commitment_ids(self): return self.links.get("commitments", [])
    @property
    def meeting_ids(self): return self.links.get("meetings", [])


class MeetingState(_BaseView):
    @property
    def title(self): return self.fields.get("title")
    @property
    def status(self): return self.fields.get("status")
    @property
    def participants(self): return self.fields.get("attendees", [])  # latest-wins, current
    @property
    def commitment_ids(self): return self.links.get("commitments", [])
    @property
    def decision_ids(self): return self.links.get("decisions", [])


class ThreadState(_BaseView):
    @property
    def subject(self): return self.fields.get("subject")
    @property
    def message_ids(self): return self.lists.get("message_ids", [])
    @property
    def participants(self): return self.lists.get("participants", [])
    # Records fold in ascending (event_timestamp, memory_id) order with
    # latest-non-null-wins, so these describe the NEWEST message in the thread —
    # not the thread as a whole. `direction == "inbound"` therefore means "the
    # other party spoke last", i.e. nobody has replied yet, and `last_seen` is
    # when they spoke. That is the entire mechanism behind WAITING_ON_USER_REPLY,
    # and it needs no extra state, no new table and no LLM.
    @property
    def direction(self): return self.fields.get("direction")
    @property
    def last_message_from(self): return self.fields.get("from_email")
    @property
    def last_message_from_name(self): return self.fields.get("from_name")
    @property
    def rfc_message_id(self): return self.fields.get("rfc_message_id")
    @property
    def category(self): return self.fields.get("category")
    @property
    def is_automated(self): return bool(self.fields.get("is_automated"))
    @property
    def thread_id(self): return self.fields.get("thread_id")
    # STICKY history markers — set once and never cleared, because the fold
    # skips nulls. "Someone said they delivered" / "someone recorded a
    # decision" are facts about the whole thread, not about its newest message.
    @property
    def fulfilment_at(self): return self.fields.get("fulfilment_at")
    @property
    def fulfilment_from(self): return self.fields.get("fulfilment_from")
    @property
    def resolution_at(self): return self.fields.get("resolution_at")
    @property
    def resolution_from(self): return self.fields.get("resolution_from")


class RelationshipState(_SenderClassifiedView):
    @property
    def person_email(self): return self.fields.get("person_email")
    @property
    def last_interaction_at(self): return self.last_seen
    @property
    def directions(self): return self.lists.get("directions", [])
    @property
    def commitment_ids(self): return self.links.get("commitments", [])


class CommitmentState(_BaseView):
    @property
    def text(self): return self.fields.get("text")
    @property
    def speaker_email(self): return self.fields.get("speaker_email")
    @property
    def speaker_name(self): return self.fields.get("speaker_name")
    @property
    def thread_id(self): return self.fields.get("thread_id")
    @property
    def subject(self): return self.fields.get("subject")
    # ISO deadline, present ONLY when the promise stated one. None means "no
    # date given" — it is never a parse failure, and nothing may substitute a
    # guess for it.
    @property
    def due_at(self): return self.fields.get("due_at")
    @property
    def due_text(self): return self.fields.get("due_text")
    @property
    def promised_at(self): return self.fields.get("promised_at")
    @property
    def status(self): return self.fields.get("status", "open")  # fact-derived, no calc
    @property
    def owner_id(self): return (self.links.get("owner") or [None])[0]


class DecisionState(_BaseView):
    @property
    def text(self): return self.fields.get("text")
    @property
    def status(self): return self.fields.get("status", "open")
    # "record" (a decision the org made) or "request" (one the org owes). Older
    # facts predate the discriminator and are records by construction.
    @property
    def kind(self): return self.fields.get("kind", "record")
    @property
    def asked_by(self): return self.fields.get("asked_by")
    @property
    def asked_by_name(self): return self.fields.get("asked_by_name")
    @property
    def asked_at(self): return self.fields.get("asked_at")
    @property
    def thread_id(self): return self.fields.get("thread_id")
    @property
    def subject(self): return self.fields.get("subject")
    @property
    def rfc_message_id(self): return self.fields.get("rfc_message_id")


class ChannelState(_BaseView):
    @property
    def name(self): return self.fields.get("name")
    @property
    def visibility(self): return self.fields.get("visibility")
    @property
    def workspace_id(self): return self.fields.get("workspace_id")
    @property
    def person_ids(self): return self.links.get("people", [])


class DocumentState(_BaseView):
    @property
    def title(self): return self.fields.get("title")
    @property
    def status(self):
        """"active" | "trashed" — latest-wins, so a restore flips it back.
        Defaults to "active" for records written before the field existed."""
        return self.fields.get("status", "active")
    @property
    def is_trashed(self): return self.status == "trashed"
    @property
    def owner_emails(self): return self.fields.get("owners", [])     # raw emails from the payload
    @property
    def drive_id(self): return self.fields.get("drive_id")
    @property
    def project_ids(self): return self.links.get("projects", [])
    @property
    def owner_ids(self): return self.links.get("owners", [])         # resolved Person entity_ids


_VIEW_FOR = {
    EntityType.PERSON: PersonState,
    EntityType.COMPANY: CompanyState,
    EntityType.CUSTOMER: CustomerState,
    EntityType.PROJECT: ProjectState,
    EntityType.MEETING: MeetingState,
    EntityType.THREAD: ThreadState,
    EntityType.RELATIONSHIP: RelationshipState,
    EntityType.COMMITMENT: CommitmentState,
    EntityType.DECISION: DecisionState,
    EntityType.CHANNEL: ChannelState,
    EntityType.DOCUMENT: DocumentState,
}


def _view(entity_type: EntityType, raw: dict) -> _BaseView:
    # Fallback to the base view for any entity type without a typed class, so a new
    # Memory entity can never crash a read (the completeness test forces a typed view).
    cls = _VIEW_FOR.get(entity_type, _BaseView)
    return cls(
        entity_id=raw["entity_id"],
        first_seen=raw.get("first_seen"),
        last_seen=raw.get("last_seen"),
        fields=raw.get("fields", {}),
        lists=raw.get("lists", {}),
        links=raw.get("links", {}),
        provenance=raw.get("provenance", {}),
    )


# --- The aggregate read model ------------------------------------------------

@dataclass(slots=True)
class OrganizationalState:
    org_id: uuid.UUID
    as_of: datetime | None
    watermark: Watermark
    schema_version: int = STATE_SCHEMA_VERSION
    projection_version: int = PROJECTION_VERSION
    # {entity_type_value: {entity_id_str: raw_state_dict}}
    entities: dict[str, dict[str, dict]] = field(default_factory=dict)

    def _by_type(self, entity_type: EntityType) -> list[_BaseView]:
        bucket = self.entities.get(entity_type.value, {})
        return [_view(entity_type, bucket[k]) for k in sorted(bucket)]

    def people(self) -> list[PersonState]: return cast("list[PersonState]", self._by_type(EntityType.PERSON))
    def companies(self) -> list[CompanyState]: return cast("list[CompanyState]", self._by_type(EntityType.COMPANY))
    def customers(self) -> list[CustomerState]: return cast("list[CustomerState]", self._by_type(EntityType.CUSTOMER))
    def projects(self) -> list[ProjectState]: return cast("list[ProjectState]", self._by_type(EntityType.PROJECT))
    def meetings(self) -> list[MeetingState]: return cast("list[MeetingState]", self._by_type(EntityType.MEETING))
    def threads(self) -> list[ThreadState]: return cast("list[ThreadState]", self._by_type(EntityType.THREAD))
    def relationships(self) -> list[RelationshipState]: return cast("list[RelationshipState]", self._by_type(EntityType.RELATIONSHIP))
    def commitments(self) -> list[CommitmentState]: return cast("list[CommitmentState]", self._by_type(EntityType.COMMITMENT))
    def decisions(self) -> list[DecisionState]: return cast("list[DecisionState]", self._by_type(EntityType.DECISION))
    def channels(self) -> list[ChannelState]: return cast("list[ChannelState]", self._by_type(EntityType.CHANNEL))
    def documents(self) -> list[DocumentState]: return cast("list[DocumentState]", self._by_type(EntityType.DOCUMENT))

    def counts(self) -> dict[str, int]:
        """Pure facts only — counts, never interpretations."""
        return {et.value: len(self.entities.get(et.value, {})) for et in EntityType}

    def to_dict(self) -> dict[str, Any]:
        # Deterministic: sort entity buckets and ids.
        ents = {
            et: {eid: bucket[eid] for eid in sorted(bucket)}
            for et, bucket in sorted(self.entities.items())
        }
        return {
            "org_id": str(self.org_id),
            "as_of": self.as_of.isoformat() if self.as_of else None,
            "schema_version": self.schema_version,
            "projection_version": self.projection_version,
            "watermark": self.watermark.to_dict(),
            "entities": ents,
            "counts": self.counts(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "OrganizationalState":
        as_of = d.get("as_of")
        return cls(
            org_id=uuid.UUID(d["org_id"]),
            as_of=datetime.fromisoformat(as_of) if as_of else None,
            watermark=Watermark.from_dict(d.get("watermark")),
            schema_version=d.get("schema_version", STATE_SCHEMA_VERSION),
            projection_version=d.get("projection_version", PROJECTION_VERSION),
            entities=d.get("entities", {}),
        )
