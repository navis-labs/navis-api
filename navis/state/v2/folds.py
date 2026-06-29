"""Cross-entity link computation (the relational fold).

Pure and deterministic: given the entity sets, derive reference lists between
entities by the natural keys ALREADY present in Memory payloads (email, domain,
slack_user_id, channel_id, folder_id). Recomputed from scratch on every finalize,
so full and incremental projections produce identical links.

Determinism note: link maps resolve a join key to the SMALLEST entity_id on
collision (`_resolve`), never first-by-iteration. Two distinct entities can share
a join key — e.g. an email-keyed Person and a Slack-keyed Person whose snapshot
carries the same email are separate identities (Memory keeps slack_user_id and
email distinct on purpose). Dict iteration order differs between a full build
(first-appearance order) and incremental rehydration (snapshot is id-sorted), so a
first-wins resolve would diverge; min-entity_id is order-independent.

These are LINKS (references) derived from existing facts — no scoring, no
judgment, no inferred identity resolution. That stays a downstream concern.
"""
from __future__ import annotations

from navis.memory.envelope import EntityType


def _domain(email: str) -> str:
    return email.split("@", 1)[1] if email and "@" in email else ""


def _add(lst: list, value: str) -> None:
    """Append-unique, sorted — deterministic order, no duplicates."""
    if value not in lst:
        lst.append(value)
        lst.sort()


def _link(raw: dict, key: str, value: str) -> None:
    _add(raw["links"].setdefault(key, []), value)


def _resolve(mapping: dict[str, str], key: str, eid: str) -> None:
    """Map join key -> entity_id, smallest entity_id winning on collision.
    Iteration-order-independent so full and incremental rebuilds agree."""
    cur = mapping.get(key)
    if cur is None or eid < cur:
        mapping[key] = eid


def compute_links(entities: dict[str, dict[str, dict]]) -> None:
    people = entities.get(EntityType.PERSON.value, {})
    rels = entities.get(EntityType.RELATIONSHIP.value, {})
    commits = entities.get(EntityType.COMMITMENT.value, {})
    companies = entities.get(EntityType.COMPANY.value, {})
    meetings = entities.get(EntityType.MEETING.value, {})
    channels = entities.get(EntityType.CHANNEL.value, {})
    documents = entities.get(EntityType.DOCUMENT.value, {})
    projects = entities.get(EntityType.PROJECT.value, {})

    # Reset all link sets first (so stale links never linger across finalizes).
    for bucket in entities.values():
        for raw in bucket.values():
            raw["links"] = {}

    # --- Join indexes (all order-independent via _resolve) -------------------
    email_to_person: dict[str, str] = {}
    slack_uid_to_person: dict[str, str] = {}
    for eid, raw in people.items():
        email = (raw["fields"].get("email") or "").lower()
        if email:
            _resolve(email_to_person, email, eid)
        uid = raw["fields"].get("slack_user_id")
        if uid:
            _resolve(slack_uid_to_person, str(uid), eid)

    email_to_rel: dict[str, str] = {}
    for eid, raw in rels.items():
        email = (raw["fields"].get("person_email") or "").lower()
        if email:
            _resolve(email_to_rel, email, eid)

    channel_by_id: dict[str, str] = {}
    for eid, raw in channels.items():
        cid = raw["fields"].get("channel_id")
        if cid:
            _resolve(channel_by_id, str(cid), eid)

    folder_to_project: dict[str, str] = {}
    for eid, raw in projects.items():
        fid = raw["fields"].get("folder_id")
        if fid:
            _resolve(folder_to_project, str(fid), eid)

    # --- Person <-> Relationship (by email) ----------------------------------
    for email, person_eid in email_to_person.items():
        rel_eid = email_to_rel.get(email)
        if rel_eid:
            _link(people[person_eid], "relationships", rel_eid)

    # --- Company -> people (by email domain) ---------------------------------
    domain_to_company = {
        (raw["fields"].get("domain") or "").lower(): eid for eid, raw in companies.items()
    }
    for email, person_eid in email_to_person.items():
        company_eid = domain_to_company.get(_domain(email).lower())
        if company_eid:
            _link(companies[company_eid], "people", person_eid)

    # --- Commitment -> owner (speaker), Relationship -> commitments ----------
    for c_eid, raw in commits.items():
        speaker = (raw["fields"].get("speaker_email") or "").lower()
        if not speaker:
            continue
        owner = email_to_person.get(speaker)
        if owner:
            _link(raw, "owner", owner)
        rel_eid = email_to_rel.get(speaker)
        if rel_eid:
            _link(rels[rel_eid], "commitments", c_eid)

    # --- Relationship -> meetings (person appears in a meeting's attendees) ---
    for m_eid, raw in meetings.items():
        for attendee in raw["fields"].get("attendees", []) or []:
            rel_eid = email_to_rel.get((attendee or "").lower())
            if rel_eid:
                _link(rels[rel_eid], "meetings", m_eid)

    # --- Person <-> Channel (Slack: relationship carries user + channel) -----
    # The slack RELATIONSHIP fact ("user active in channel") joins a Slack person
    # (by slack_user_id) to a channel (by channel_id) — both ids exist in payloads.
    for raw in rels.values():
        uid = raw["fields"].get("slack_user_id")
        cid = raw["fields"].get("channel_id")
        if not uid or not cid:
            continue
        person_eid = slack_uid_to_person.get(str(uid))
        channel_eid = channel_by_id.get(str(cid))
        if person_eid and channel_eid:
            _link(people[person_eid], "channels", channel_eid)
            _link(channels[channel_eid], "people", person_eid)

    # --- Document <-> Project (document's parent folder == project folder) ----
    for doc_eid, raw in documents.items():
        for parent in raw["fields"].get("parents", []) or []:
            proj_eid = folder_to_project.get(str(parent))
            if proj_eid:
                _link(documents[doc_eid], "projects", proj_eid)
                _link(projects[proj_eid], "documents", doc_eid)

    # --- Document -> owners (people, by owner email) -------------------------
    for doc_eid, raw in documents.items():
        for owner_email in raw["fields"].get("owners", []) or []:
            person_eid = email_to_person.get((owner_email or "").lower())
            if person_eid:
                _link(documents[doc_eid], "owners", person_eid)
                _link(people[person_eid], "documents", doc_eid)
