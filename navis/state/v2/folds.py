"""Cross-entity link computation (the relational fold).

Pure and deterministic: given the entity sets, derive reference lists between
entities by their natural keys (email, domain). Recomputed from scratch on every
finalize, so full and incremental projections produce identical links.

These are LINKS (references), not intelligence — no scoring, no judgment.
"""
from __future__ import annotations

from navis.memory.envelope import EntityType


def _domain(email: str) -> str:
    return email.split("@", 1)[1] if email and "@" in email else ""


def compute_links(entities: dict[str, dict[str, dict]]) -> None:
    people = entities.get(EntityType.PERSON.value, {})
    rels = entities.get(EntityType.RELATIONSHIP.value, {})
    commits = entities.get(EntityType.COMMITMENT.value, {})
    companies = entities.get(EntityType.COMPANY.value, {})
    meetings = entities.get(EntityType.MEETING.value, {})

    # Reset all link sets first (so stale links never linger across finalizes).
    for bucket in entities.values():
        for raw in bucket.values():
            raw["links"] = {}

    # email -> person entity_id
    email_to_person: dict[str, str] = {}
    for eid, raw in people.items():
        email = (raw["fields"].get("email") or "").lower()
        if email:
            email_to_person.setdefault(email, eid)

    # email -> relationship entity_id
    email_to_rel: dict[str, str] = {}
    for eid, raw in rels.items():
        email = (raw["fields"].get("person_email") or "").lower()
        if email:
            email_to_rel.setdefault(email, eid)

    # Person -> relationships
    for email, person_eid in email_to_person.items():
        rel_eid = email_to_rel.get(email)
        if rel_eid:
            people[person_eid]["links"].setdefault("relationships", [])
            _add(people[person_eid]["links"]["relationships"], rel_eid)

    # Company -> people (by email domain)
    domain_to_company = {
        (raw["fields"].get("domain") or "").lower(): eid for eid, raw in companies.items()
    }
    for email, person_eid in email_to_person.items():
        company_eid = domain_to_company.get(_domain(email).lower())
        if company_eid:
            companies[company_eid]["links"].setdefault("people", [])
            _add(companies[company_eid]["links"]["people"], person_eid)

    # Commitment -> owner (speaker), Relationship -> commitments
    for c_eid, raw in commits.items():
        speaker = (raw["fields"].get("speaker_email") or "").lower()
        if not speaker:
            continue
        owner = email_to_person.get(speaker)
        if owner:
            raw["links"].setdefault("owner", [])
            _add(raw["links"]["owner"], owner)
        rel_eid = email_to_rel.get(speaker)
        if rel_eid:
            rels[rel_eid]["links"].setdefault("commitments", [])
            _add(rels[rel_eid]["links"]["commitments"], c_eid)

    # Relationship -> meetings (person appears in a meeting's attendees)
    for m_eid, raw in meetings.items():
        for attendee in raw["fields"].get("attendees", []) or []:
            rel_eid = email_to_rel.get((attendee or "").lower())
            if rel_eid:
                rels[rel_eid]["links"].setdefault("meetings", [])
                _add(rels[rel_eid]["links"]["meetings"], m_eid)


def _add(lst: list, value: str) -> None:
    if value not in lst:
        lst.append(value)
        lst.sort()
