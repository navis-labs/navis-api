"""Deterministic extractors over Gmail SignalEvents.

Email payloads carry from/to/cc, subject, body_text, thread_id (parsed by the
Gmail connector). Every extractor here is pure: same email → same facts.
"""
from __future__ import annotations

import hashlib
import re
from email.utils import parseaddr

from navis.memory.confidence import HEURISTIC, HIGH, LOW, MEDIUM, STRUCTURAL
from navis.memory.envelope import EntityType
from navis.memory.extractors.base import Extractor, MemoryDraft, SignalView
from navis.memory.schemas import current_version
from navis.signals.base import SignalSource

_GMAIL = {SignalSource.GMAIL}

# Free/consumer domains are not "companies".
_FREE_DOMAINS = {
    "gmail.com", "googlemail.com", "yahoo.com", "outlook.com", "hotmail.com",
    "icloud.com", "proton.me", "protonmail.com", "aol.com", "live.com",
}

_COMMITMENT_RE = re.compile(
    "|".join(
        [
            r"\bi'?ll\b", r"\bwe'?ll\b", r"\bi will\b", r"\bwe will\b",
            r"\bwe'?re going to\b", r"\bfollow(?:ing)? up\b", r"\bget back to you\b",
            r"\bsend (?:you|over|across)\b", r"\bcircle back\b",
            r"\bby (?:eod|end of day|tomorrow|monday|tuesday|wednesday|thursday|friday|next week)\b",
        ]
    ),
    re.IGNORECASE,
)
_DECISION_RE = re.compile(
    r"\b(we (?:have )?decided|decision:|we'?ve agreed|approved|we will go with|let'?s go with)\b",
    re.IGNORECASE,
)


def _addresses(payload: dict) -> list[tuple[str, str, str]]:
    """(role, name, email) for each address across from/to/cc, de-duplicated."""
    out: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str]] = set()
    for role in ("from", "to", "cc"):
        raw = payload.get(role)
        if not raw:
            continue
        for chunk in str(raw).split(","):
            name, email = parseaddr(chunk)
            email = email.strip().lower()
            if email and (role, email) not in seen:
                seen.add((role, email))
                out.append((role, name or "", email))
    return out


def _sentences(body: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?\n])\s+", body) if s.strip()]


def _fact_key(text: str) -> str:
    """CONTENT-derived key for a text fact (commitment/decision sentence). Stable across
    sentence re-ordering and any change to the splitter — unlike a positional index, which
    shifts and creates duplicate histories. Normalized (strip+lower+collapse-space) so trivial
    formatting differences don't fork the id."""
    norm = " ".join(text.strip().lower().split())
    return hashlib.sha256(norm.encode()).hexdigest()[:16]


class PersonExtractor(Extractor):
    name = "email_person"
    version = "1.0.0"
    sources = _GMAIL

    def extract(self, signal: SignalView) -> list[MemoryDraft]:
        ver = current_version(EntityType.PERSON)
        drafts = []
        for role, name, email in _addresses(signal.payload):
            drafts.append(
                MemoryDraft(
                    entity_type=EntityType.PERSON,
                    entity_natural_key=email,
                    dedup_key=email,  # one PERSON fact per address per signal
                    payload={"email": email, "name": name or None, "role": role},
                    confidence=HIGH,
                    schema_version=ver,
                )
            )
        return drafts


class CompanyExtractor(Extractor):
    name = "email_company"
    version = "1.0.0"
    sources = _GMAIL

    def extract(self, signal: SignalView) -> list[MemoryDraft]:
        ver = current_version(EntityType.COMPANY)
        domains: dict[str, None] = {}
        for _role, _name, email in _addresses(signal.payload):
            domain = email.split("@", 1)[1] if "@" in email else ""
            if domain and domain not in _FREE_DOMAINS:
                domains[domain] = None
        return [
            MemoryDraft(
                entity_type=EntityType.COMPANY,
                entity_natural_key=domain,
                dedup_key=domain,
                payload={"domain": domain, "name": None},
                confidence=MEDIUM,
                schema_version=ver,
            )
            for domain in domains
        ]


class ThreadExtractor(Extractor):
    name = "email_thread"
    version = "1.0.0"
    sources = _GMAIL

    def extract(self, signal: SignalView) -> list[MemoryDraft]:
        thread_id = signal.payload.get("thread_id")
        if not thread_id:
            return []
        return [
            MemoryDraft(
                entity_type=EntityType.THREAD,
                entity_natural_key=str(thread_id),
                dedup_key=str(thread_id),  # one thread-observation per message
                payload={
                    "thread_id": str(thread_id),
                    "subject": signal.payload.get("subject"),
                    "message_id": signal.external_id,
                    # Reply graph (KEEP): lets State/Reasoning reconstruct replied_to /
                    # references_message edges without re-parsing the raw email.
                    "rfc_message_id": signal.payload.get("rfc_message_id"),
                    "in_reply_to": signal.payload.get("in_reply_to"),
                    "references": signal.payload.get("references"),
                },
                confidence=STRUCTURAL,
                schema_version=current_version(EntityType.THREAD),
            )
        ]


class RelationshipExtractor(Extractor):
    name = "email_relationship"
    version = "1.0.0"
    sources = _GMAIL

    def extract(self, signal: SignalView) -> list[MemoryDraft]:
        ver = current_version(EntityType.RELATIONSHIP)
        drafts = []
        for role, _name, email in _addresses(signal.payload):
            direction = "inbound" if role == "from" else "outbound"
            drafts.append(
                MemoryDraft(
                    entity_type=EntityType.RELATIONSHIP,
                    entity_natural_key=email,
                    dedup_key=f"{direction}:{email}",
                    # An OBSERVED interaction — no health, no score.
                    payload={"person_email": email, "direction": direction, "via": "email"},
                    confidence=HIGH,
                    schema_version=ver,
                )
            )
        return drafts


class CommitmentExtractor(Extractor):
    name = "email_commitment"
    version = "1.0.0"
    sources = _GMAIL

    def extract(self, signal: SignalView) -> list[MemoryDraft]:
        body = signal.payload.get("body_text") or signal.payload.get("snippet") or ""
        if not body:
            return []
        _name, speaker = parseaddr(str(signal.payload.get("from", "")))
        ver = current_version(EntityType.COMMITMENT)
        drafts = []
        for idx, sentence in enumerate(_sentences(body)):
            if 8 <= len(sentence) <= 400 and _COMMITMENT_RE.search(sentence):
                drafts.append(
                    MemoryDraft(
                        entity_type=EntityType.COMMITMENT,
                        entity_natural_key=f"{signal.external_id}:{_fact_key(sentence)}",
                        dedup_key=_fact_key(sentence),
                        payload={
                            "text": sentence,
                            "speaker_email": speaker.lower() or None,
                            "subject": signal.payload.get("subject"),
                            "thread_id": signal.payload.get("thread_id"),
                        },
                        confidence=HEURISTIC,
                        schema_version=ver,
                    )
                )
                if len(drafts) >= 5:  # bound per email (deterministic: first 5)
                    break
        return drafts


class DecisionExtractor(Extractor):
    name = "email_decision"
    version = "1.0.0"
    sources = _GMAIL

    def extract(self, signal: SignalView) -> list[MemoryDraft]:
        body = signal.payload.get("body_text") or ""
        if not body:
            return []
        ver = current_version(EntityType.DECISION)
        drafts = []
        for idx, sentence in enumerate(_sentences(body)):
            if 8 <= len(sentence) <= 400 and _DECISION_RE.search(sentence):
                drafts.append(
                    MemoryDraft(
                        entity_type=EntityType.DECISION,
                        entity_natural_key=f"{signal.external_id}:{_fact_key(sentence)}",
                        dedup_key=f"d:{_fact_key(sentence)}",
                        payload={"text": sentence, "subject": signal.payload.get("subject")},
                        confidence=LOW,
                        schema_version=ver,
                    )
                )
                if len(drafts) >= 5:
                    break
        return drafts
