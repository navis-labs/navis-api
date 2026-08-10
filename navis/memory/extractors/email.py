"""Deterministic extractors over Gmail SignalEvents.

Email payloads carry from/to/cc, subject, body_text, thread_id (parsed by the
Gmail connector). Every extractor here is pure: same email → same facts.
"""
from __future__ import annotations

import hashlib
import re
from email.utils import parseaddr
from html import unescape
from html.parser import HTMLParser
from io import StringIO

from navis.memory.confidence import HEURISTIC, HIGH, LOW, MEDIUM, STRUCTURAL
from navis.memory.envelope import EntityType
from navis.memory.extractors.base import Extractor, MemoryDraft, SignalView
from navis.memory.schemas import current_version
from navis.signals.base import SignalSource
from navis.signals.due_dates import extract_due_at
from navis.signals.text_rules import (
    body_records_decision,
    body_shows_fulfilment,
    is_binding_commitment,
    # RE-EXPORT, not dead: extractors.slack and ml.embeddings.admission import
    # `is_commitment_sentence` FROM this module, so removing it here breaks
    # collection of the whole ml package. Not detectable by an unused-import
    # scan — it has no reference in this file at all.
    is_commitment_sentence,  # noqa: F401
    is_decision_request_sentence,
    is_decision_sentence,
    sentences,
)

_GMAIL = {SignalSource.GMAIL}

# Free/consumer domains are not "companies".
_FREE_DOMAINS = {
    "gmail.com", "googlemail.com", "yahoo.com", "outlook.com", "hotmail.com",
    "icloud.com", "proton.me", "protonmail.com", "aol.com", "live.com",
}


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


class _TextOnly(HTMLParser):
    """Collect visible text; drop <script>/<style> content entirely."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._out = StringIO()
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in ("script", "style", "head"):
            self._skip_depth += 1
        elif tag in ("br", "p", "div", "tr", "td", "li"):
            self._out.write(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style", "head") and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._out.write(data)

    def value(self) -> str:
        return self._out.getvalue()


def _html_to_text(html: str) -> str:
    """Strip HTML tags/entities to plain text. Permissive on malformed markup."""
    parser = _TextOnly()
    try:
        parser.feed(html)
        parser.close()
        text = parser.value()
    except Exception:
        # Fall back to a plain regex strip; HTMLParser is permissive but not
        # infallible on truly pathological input.
        text = re.sub(r"<[^>]+>", " ", html)
        text = unescape(text)
    return re.sub(r"\s+", " ", text).strip()


# Any signal that HTML is embedded in what was claimed to be text/plain.
# Newsletters and marketing mail routinely mis-declare the mime type.
_HTML_HEURISTIC_RE = re.compile(r"<(?:td|tr|table|div|p|span|a|br|img)\b", re.IGNORECASE)


# Ciphertext tag written by the field cipher (see signals/crypto.py). Bodies are
# sealed at rest and decrypted by the Memory read seam (SignalViewFactory) before
# extractors run. If a body still carries the tag here, decryption did not happen
# — treat it as no body rather than extracting from, and embedding, ciphertext.
_SEALED_PREFIX = "fenc:"


def _body_as_text(payload: dict) -> str:
    """Return the email body as plain text.

    The Gmail parser stores raw HTML in `body_text` when no text/plain part
    exists (see signals/gmail/parser.py — de-noising is Memory's job). Also
    catches senders that stuff HTML into a mis-declared text/plain part.
    """
    body = payload.get("body_text") or payload.get("snippet") or ""
    if not body or body.startswith(_SEALED_PREFIX):
        return ""
    mime = payload.get("body_mime")
    if mime == "text/html" or _HTML_HEURISTIC_RE.search(body):
        return _html_to_text(body)
    return body


# --- Bulk / automated mail classification -----------------------------------
# A commitment or a decision is something a PERSON states. Newsletters, receipts,
# OTP codes and marketing blasts are not organizational promises, yet they are the
# overwhelming majority of a founder's inbox and they use the same verbs
# ("we'll tackle them live", "we will send you offers"). We classify from the
# RFC bulk-mail headers and Gmail's own categories that the connector already
# captures — no content heuristics, so it stays deterministic.

_BULK_PRECEDENCE = {"bulk", "list", "junk"}
# CATEGORY_UPDATES covers receipts / OTPs / "your payment is due" — automated by
# construction. CATEGORY_PERSONAL and the absence of any category mean human mail.
_BULK_LABELS = {
    "CATEGORY_PROMOTIONS", "CATEGORY_SOCIAL", "CATEGORY_FORUMS",
    "CATEGORY_UPDATES", "SPAM",
}
_NOREPLY_LOCALPART_RE = re.compile(
    r"^(?:no[-_.]?reply|do[-_.]?not[-_.]?reply|donotreply|mailer[-_.]?daemon|postmaster"
    r"|bounces?|notifications?|newsletters?|alerts?|automated|mailer)(?:[-+._]|$)",
    re.IGNORECASE,
)
_LABEL_TO_CATEGORY = [
    ("SPAM", "spam"),
    ("CATEGORY_PROMOTIONS", "promotional"),
    ("CATEGORY_SOCIAL", "social"),
    ("CATEGORY_FORUMS", "forum"),
    ("CATEGORY_UPDATES", "update"),
]


def _labels(payload: dict) -> set[str]:
    """Gmail label ids. The connector writes `label_ids`; older/other captures
    used `labels` — read both so classification never depends on which."""
    out: set[str] = set()
    for key in ("label_ids", "labels"):
        raw = payload.get(key)
        if isinstance(raw, list):
            out.update(str(v) for v in raw if isinstance(v, (str, int)))
    return out


def is_automated_email(payload: dict) -> bool:
    """True when the message is bulk/automated rather than written by a person."""
    if str(payload.get("list_id") or "").strip():
        return True  # List-Id => mailing list
    if str(payload.get("precedence") or "").strip().lower() in _BULK_PRECEDENCE:
        return True
    auto = str(payload.get("auto_submitted") or "").strip().lower()
    if auto and auto != "no":
        return True  # RFC 3834
    if _labels(payload) & _BULK_LABELS:
        return True
    _name, sender = parseaddr(str(payload.get("from", "")))
    local = sender.split("@", 1)[0] if "@" in sender else ""
    return bool(local and _NOREPLY_LOCALPART_RE.match(local))


def classify_email(payload: dict) -> str:
    """Coarse, deterministic category recorded on the THREAD fact.

    Classification is kept as MEMORY (the observed fact "this arrived as a
    promotion"), never as suppression of the email itself — people, companies,
    threads and relationships are still extracted from every message.
    """
    labels = _labels(payload)
    for label, category in _LABEL_TO_CATEGORY:
        if label in labels:
            return category
    if is_automated_email(payload):
        return "automated"
    return "personal"


# --- Body cleanup ------------------------------------------------------------

# Quoted history + signature markers. Without this, every reply re-extracts the
# whole conversation and attributes the ORIGINAL sender's promises to whoever
# replied last — wrong speaker, duplicated facts.
_QUOTE_HEADER_RE = re.compile(
    r"^(?:"
    r"On .{0,120}\bwrote:"                     # "On Mon, 4 Aug 2026, Alice wrote:"
    r"|-{2,}\s*Original Message\s*-{2,}"
    r"|_{5,}"
    r"|From:\s.+"                               # Outlook-style quoted header block
    r"|Sent from my \w+"
    r")",
    re.IGNORECASE,
)
_SIGNATURE_RE = re.compile(r"^--\s*$")


# Tracking pixels / CTA links and invisible padding are pure noise in an
# embedding: newsletters pack hundreds of them and they crowd out real words.
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_INVISIBLE_RE = re.compile(
    # zero-width space/joiners, bidi marks, word-joiner, BOM, soft hyphen
    "[\u200b-\u200f\u2028\u2029\u202a-\u202e\u2060\ufeff\u00ad]"
)


def snippet_text(body: str, limit: int) -> str:
    """Bounded, whitespace-normalized opening of a message body for embedding."""
    text = _INVISIBLE_RE.sub("", body)
    text = _URL_RE.sub(" ", text)
    return " ".join(text.split())[:limit]


def strip_quoted_reply(body: str) -> str:
    """Keep only what THIS message adds: drop quoted history and the signature."""
    kept: list[str] = []
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith(">"):
            continue  # quoted line
        if _QUOTE_HEADER_RE.match(stripped) or _SIGNATURE_RE.match(stripped):
            break  # everything after a quote header/signature belongs to history
        kept.append(line)
    return "\n".join(kept).strip()


# Sentence splitting and the commitment/decision predicates live in
# navis.signals.text_rules so Memory extraction and Signal importance share ONE
# definition. Re-exported here because email.py has been their import site since
# before the move (slack.py and the extraction tests both rely on it).
_sentences = sentences


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


# A thread's embeddable snippet: enough to say what the conversation is ABOUT,
# short enough that one newsletter can't dominate the vector. Bodies average
# ~12KB; we keep the opening of the message's own text.
_SNIPPET_CHARS = 240
_MAX_PARTICIPANTS = 8


def _thread_resolution_markers(signal: SignalView, body: str) -> dict:
    """Sticky evidence that something in this thread was delivered or decided.

    Returns only the keys that APPLY — omitting a key (rather than setting it
    to None) is what makes the marker sticky under State's
    latest-non-null-wins fold.
    """
    out: dict = {}
    if not body:
        return out
    sender = (signal.payload.get("sender_email") or "").strip().lower() or None
    when = signal.event_timestamp.isoformat()
    if body_shows_fulfilment(body):
        out["fulfilment_at"] = when
        out["fulfilment_from"] = sender
    if body_records_decision(body):
        out["resolution_at"] = when
        out["resolution_from"] = sender
    return out


def _message_direction(signal: SignalView) -> str:
    """"outbound" if the org sent this message, else "inbound".

    Gmail's own `SENT` label is the source of truth — it is applied by Google at
    send time, is present on every message the connector captures (the sync
    lists ALL messages, not just INBOX), and requires no interpretation. Falling
    back to header comparison would need to know every alias the founder sends
    from; the label already encodes that.

    This is deliberately NOT an inference. "Did the user reply?" must be a fact,
    not a judgement — an LLM guessing wrong here means either nagging someone
    who already replied, or staying silent on a customer who is waiting.
    """
    labels = signal.payload.get("label_ids") or signal.metadata.get("label_ids") or []
    if isinstance(labels, (list, tuple)) and "SENT" in labels:
        return "outbound"
    return "inbound"


class ThreadExtractor(Extractor):
    # RENAMED from "email_thread" (see memory.ids EVOLUTION LAW). 1.2.0 added
    # `direction` and the sticky fulfilment/resolution markers, but the identity
    # tuple excludes `version` — so replaying an org that was ingested under
    # 1.1.0 re-derived the SAME memory_ids, hit ON CONFLICT DO NOTHING, and the
    # new keys never reached the database. Measured on a live workspace: 54/54
    # thread records skipped, 0 written. The name is the only part of the key we
    # may move, and moving it is what the law prescribes for a change that
    # SHOULD produce new facts.
    #
    # `entity_natural_key` is deliberately still the bare thread_id, so v2
    # records fold onto the SAME thread entity instead of forking a second one.
    # v1 and v2 records of one message share an event_timestamp, so the fold
    # falls back to memory_id order — which is safe here in both directions:
    # every key they share carries an identical value, and every key v2 adds is
    # absent from v1, and latest-NON-NULL-wins cannot be clobbered by an absent
    # key. The old rows stay exactly as written; nothing is deleted or mutated.
    name = "email_thread_v2"
    version = "1.2.0"
    sources = _GMAIL

    def extract(self, signal: SignalView) -> list[MemoryDraft]:
        payload = signal.payload
        thread_id = payload.get("thread_id")
        if not thread_id:
            return []
        # Participants + snippet + category are ADDITIVE payload keys (see
        # schemas/payloads.py): they give the thread fact something semantically
        # meaningful to embed. Previously the only embeddable text was the opaque
        # thread_id, because `subject` sat in the payload unread.
        participants = []
        for _role, _name, email in _addresses(payload):
            if email not in participants:
                participants.append(email)
        body = strip_quoted_reply(_body_as_text(payload))
        snippet = snippet_text(body, _SNIPPET_CHARS) or None
        return [
            MemoryDraft(
                entity_type=EntityType.THREAD,
                entity_natural_key=str(thread_id),
                dedup_key=str(thread_id),  # one thread-observation per message
                payload={
                    "thread_id": str(thread_id),
                    "subject": payload.get("subject"),
                    "message_id": signal.external_id,
                    "participants": participants[:_MAX_PARTICIPANTS],
                    "snippet": snippet,
                    "category": classify_email(payload),
                    "is_automated": is_automated_email(payload),
                    # Who spoke, and which way. Latest-wins folding turns these
                    # into "who spoke LAST in this thread" for free — which is
                    # the whole basis of the waiting-on-reply signal.
                    "direction": _message_direction(signal),
                    "from_email": (payload.get("sender_email") or "").strip().lower() or None,
                    "from_name": payload.get("sender_name"),
                    # STICKY markers. State folds latest-NON-NULL-wins, so a
                    # later message that shows neither of these leaves them
                    # standing — which is exactly right: "somebody said they
                    # delivered" and "somebody recorded a decision" are facts
                    # about the thread's history, not about its newest message.
                    #
                    # Both carry a timestamp rather than a bare flag so a
                    # reasoning engine can ask "did this happen AFTER the
                    # promise?" — a delivery that predates the promise resolves
                    # nothing.
                    **_thread_resolution_markers(signal, body),
                    # Reply graph (KEEP): lets State/Reasoning reconstruct replied_to /
                    # references_message edges without re-parsing the raw email.
                    "rfc_message_id": payload.get("rfc_message_id"),
                    "in_reply_to": payload.get("in_reply_to"),
                    "references": payload.get("references"),
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
    """Promises made TO this org, with a deadline when the text states one.

    1.3.0 tightened two things:
      * only BINDING promises are recorded — "maybe I'll send it Friday" carries
        promise vocabulary but commits nobody, and chasing someone over a hedge
        is the fastest way to make a Chief of Staff feel wrong;
      * a stated deadline is parsed off the sentence and resolved against the
        MESSAGE date, so a follow-up can say "the deck was due Friday" instead
        of the much weaker "this has been open a while".

    RENAMED from "email_commitment" (see memory.ids EVOLUTION LAW). 1.3.0
    originally kept the old name, reasoning that "existing histories are
    preserved and a replay converges". Preserved turned out to be the problem:
    the identity tuple excludes `version`, so a replay re-derives the same
    memory_ids, hits ON CONFLICT DO NOTHING, and `due_at` never lands on any org
    ingested under 1.2.0. Only the name is movable, and the law prescribes
    moving it for a change that SHOULD produce new facts.

    Old 1.2.0 rows are NOT deleted — they stay as history. Two consequences,
    both accepted: a promise recorded under the looser gate keeps its commitment
    entity (it simply gains no `due_at`, and it cannot clobber one, because the
    fold is latest-NON-NULL-wins and the old row has no such key); and undated
    commitments only ever feed the brief, never a proactive candidate, because
    the proposer maps `risk` to no action type.
    """

    name = "email_commitment_v2"
    version = "1.3.0"
    sources = _GMAIL

    def extract(self, signal: SignalView) -> list[MemoryDraft]:
        # Bulk/automated mail still becomes people, companies, threads and
        # relationships — only the COMMITMENT claim is suppressed, because a
        # newsletter saying "we'll tackle them live" is not a promise to this org.
        if is_automated_email(signal.payload):
            return []
        body = strip_quoted_reply(_body_as_text(signal.payload))
        if not body:
            return []
        _name, speaker = parseaddr(str(signal.payload.get("from", "")))
        ver = current_version(EntityType.COMMITMENT)
        drafts = []
        for idx, sentence in enumerate(_sentences(body)):
            if is_binding_commitment(sentence):
                # None when the sentence states no deadline — the common case,
                # and the honest one. An invented due date would make Navis
                # assert something false about a third party.
                due = extract_due_at(sentence, sent_at=signal.event_timestamp)
                drafts.append(
                    MemoryDraft(
                        entity_type=EntityType.COMMITMENT,
                        entity_natural_key=f"{signal.external_id}:{_fact_key(sentence)}",
                        dedup_key=_fact_key(sentence),
                        payload={
                            "text": sentence,
                            "speaker_email": speaker.lower() or None,
                            "speaker_name": signal.payload.get("sender_name"),
                            "subject": signal.payload.get("subject"),
                            "thread_id": signal.payload.get("thread_id"),
                            "due_at": due[0].isoformat() if due else None,
                            "due_text": due[1] if due else None,
                            "promised_at": signal.event_timestamp.isoformat(),
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
    version = "1.2.0"
    sources = _GMAIL

    def extract(self, signal: SignalView) -> list[MemoryDraft]:
        if is_automated_email(signal.payload):
            return []
        body = strip_quoted_reply(_body_as_text(signal.payload))
        if not body:
            return []
        ver = current_version(EntityType.DECISION)
        drafts = []
        for idx, sentence in enumerate(_sentences(body)):
            if is_decision_sentence(sentence):
                drafts.append(
                    MemoryDraft(
                        entity_type=EntityType.DECISION,
                        entity_natural_key=f"{signal.external_id}:{_fact_key(sentence)}",
                        dedup_key=f"d:{_fact_key(sentence)}",
                        payload={
                            "text": sentence,
                            "subject": signal.payload.get("subject"),
                            # Discriminator within EntityType.DECISION. A
                            # decision RECORDED and a decision REQUESTED are
                            # opposite states of the same real-world thing,
                            # so they share the entity type and are told
                            # apart by this field rather than by a second
                            # store.
                            "kind": "record",
                            "thread_id": signal.payload.get("thread_id"),
                        },
                        confidence=LOW,
                        schema_version=ver,
                    )
                )
                if len(drafts) >= 5:
                    break
        return drafts


class DecisionRequestExtractor(Extractor):
    """Someone asking THIS org to decide something.

    The mirror image of `DecisionExtractor`, which records decisions already
    made. Both write EntityType.DECISION facts and are told apart by `kind`
    ("request" vs "record") — one entity type, one history per decision, no
    second decision store.

    That pairing is what makes resolution free: when the org later records a
    decision in the same thread, `DecisionExtractor` writes the sticky
    `resolution_at` marker on the THREAD, and DecisionOwedEngine stops firing.
    """

    name = "email_decision_request"
    version = "1.0.0"
    sources = _GMAIL

    def extract(self, signal: SignalView) -> list[MemoryDraft]:
        # A newsletter asking "should we all switch to X?" is rhetorical, not a
        # decision this org owes anyone.
        if is_automated_email(signal.payload):
            return []
        body = strip_quoted_reply(_body_as_text(signal.payload))
        if not body:
            return []
        # A request the ORG SENT is not a decision the org owes — it is one the
        # org is waiting on. Only inbound asks count.
        if _message_direction(signal) == "outbound":
            return []
        _name, asker = parseaddr(str(signal.payload.get("from", "")))
        ver = current_version(EntityType.DECISION)
        drafts: list[MemoryDraft] = []
        for sentence in _sentences(body):
            if not is_decision_request_sentence(sentence):
                continue
            drafts.append(
                MemoryDraft(
                    entity_type=EntityType.DECISION,
                    entity_natural_key=f"{signal.external_id}:{_fact_key(sentence)}",
                    dedup_key=f"dr:{_fact_key(sentence)}",
                    payload={
                        "text": sentence,
                        "kind": "request",
                        "asked_by": asker.lower() or None,
                        "asked_by_name": signal.payload.get("sender_name"),
                        "asked_at": signal.event_timestamp.isoformat(),
                        "subject": signal.payload.get("subject"),
                        "thread_id": signal.payload.get("thread_id"),
                        "rfc_message_id": signal.payload.get("rfc_message_id"),
                    },
                    confidence=HEURISTIC,
                    schema_version=ver,
                )
            )
            if len(drafts) >= 5:  # bound per email (deterministic: first 5)
                break
        return drafts
