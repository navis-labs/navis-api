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
        """True if (ts, memory_id) is strictly AFTER this watermark."""
        if self.last_timestamp is None:
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
        return cls(
            last_timestamp=datetime.fromisoformat(ts) if ts else None,
            last_memory_id=uuid.UUID(mid) if mid else None,
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


class PersonState(_BaseView):
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


class RelationshipState(_BaseView):
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
    def status(self): return self.fields.get("status", "open")  # fact-derived, no calc
    @property
    def owner_id(self): return (self.links.get("owner") or [None])[0]


class DecisionState(_BaseView):
    @property
    def text(self): return self.fields.get("text")
    @property
    def status(self): return self.fields.get("status", "open")


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
