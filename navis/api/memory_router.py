"""Memory read edge — surfaces People & Decisions to the dashboard.

Reads immutable facts from `memory_records` and derives lightweight, RULE-BASED
metrics (impactScore, outcomeStatus, escalationRisk, ...) at request time. No LLM.
Same auth + shape discipline as brief/context routers so the dashboard can
consume this alongside them without special-casing.

Determinism: the raw records are event-sourced; the derivations here are pure
functions of those records (regex + counts + timestamps), so replay is stable.

Noise gating: the extractors are permissive by design (rules fire on any signal
match), so newsletter/marketing traffic ends up in memory as PERSON+DECISION
records. This edge filters that noise OUT for the dashboard surfaces — the raw
records stay in memory (still queryable by /chat's retrieval) so we never lose
data, we only hide low-signal rows from founder-facing panels.
"""
from __future__ import annotations

import re
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from navis.identity.authz import Principal
from navis.memory.envelope import EntityType
from navis.memory.repository import MemoryRepository
from navis.organization.contract import Capability
from navis.organization.guards import require_capability
from navis.platform.database import get_session

router = APIRouter(tags=["memory"])

# Free/consumer domains (mirrors extractor list) — used to classify roleHint.
_FREE_DOMAINS = {
    "gmail.com", "googlemail.com", "yahoo.com", "outlook.com", "hotmail.com",
    "icloud.com", "proton.me", "protonmail.com", "aol.com", "live.com",
}

# ── Noise filters (Fix 1 + Fix 2) ────────────────────────────────────────────
# A valid email is roughly `local@domain.tld` with no quotes/braces/spaces. The
# extractor sometimes stores broken input verbatim (parser edge cases + probe
# rows leaking from tests); a strict shape check drops those rows before we
# ever try to reason about them.
_VALID_EMAIL_RE = re.compile(
    r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"
)

# `noreply` / `no-reply` / `donotreply` / `do-not-reply` can appear ANYWHERE in
# the local part (`workspace-noreply@`, `analytics-noreply@`, `google-noreply@`),
# not just at the start — so match as a substring.
_NOREPLY_SUBSTRING_RE = re.compile(
    r"no[-_]?reply|do[-_]?not[-_]?reply|donotreply|no[-_]?response",
    re.IGNORECASE,
)

# Local-part markers of automated / bulk senders. Anchored at the start of the
# local part so we don't kill a real person named "info" (`info.chandra@…`)
# — but the `-|_|.` separator means `hello-team@`, `team.marketing@` etc. still
# match. Trailing `-` inside a token is optional so `beca` matches `beca-joven`.
_AUTOMATED_LOCAL_RE = re.compile(
    r"^("
    r"newsletter|marketing|notification|notifications|alerts?|updates?|"
    r"support|hello|team|community|invite|opinion|"
    r"recommendations?|pageupdates|friendsuggestion|"
    r"digest|news|info|mail|mails|contact|welcome|"
    r"security|billing|receipts?|gogreen|wecare|beca|"
    # 'the…team', 'the…news', 'the…updates' — common bulk-sender construction
    # (thehubspotteam, the-verge-team, etc.). Anchored at start.
    r"the[a-z]+(team|news|updates?|digest|weekly)"
    r")([-_.].*)?$",
    re.IGNORECASE,
)

# Domain markers of marketing / notification subdomains. Catches `mail.foo.com`,
# `updates.foo.com`, `notification.priority.foo.com`, AND hyphenated marketing
# variants like `web-spicejet.com` — token boundary is `.` OR `-`, so we catch
# both `mail.foo.com` and `mail-foo.com`.
_AUTOMATED_DOMAIN_RE = re.compile(
    r"(^|[.-])("
    r"mail|mails|email|emails|updates|notification|notifications|"
    r"newsletter|digest|marketing|alerts?|inbox|"
    r"engage|connect|discover|inspire|explore|priority|panel|"
    r"rmp|ses|smailer\d*|tm|web|"
    r"finance|shop|rewards"
    r")[.-]",
    re.IGNORECASE,
)
# Whole domains that are pure newsletter/marketing platforms — matched against
# the FULL registrable domain, not a subdomain. `substack.com` is the canonical
# example (every substack@substack.com sender is a newsletter).
_AUTOMATED_WHOLE_DOMAIN = frozenset({
    "substack.com",
    "mailchimp.com",
    "mailchi.mp",
    "beehiiv.com",
    "convertkit.com",
})

# Subject markers of newsletters / round-ups — bulk content, not real decisions.
_NEWSLETTER_SUBJECT_RE = re.compile(
    r"digest|newsletter|weekly|daily|briefing|"
    r"round[- ]?up|&\s*more\s*$|\-\s*Groww\b|"
    r"morning brief|evening brief|today[' ]s (?:news|update)",
    re.IGNORECASE,
)

# Third-party news-report structure — "The government approved", "Board approved",
# "Aurobindo Pharma: ...", "Sun Pharma: shareholders approved ..." — these are
# NEWS about someone else's decision, not a decision made by the user.
_THIRD_PARTY_TEXT_RE = re.compile(
    r"^\s*(the\s+(government|board|company|committee|council|indian\s+railways|"
    r"defen[cs]e\s+acquisition\s+council|dac|centre|state|ministry|cabinet|"
    r"reserve\s+bank|rbi|sebi)\b|"
    r"[A-Z][\w&]*\s+(Pharma|Bank|Motors?|Steel|Industries|Corp|Ltd|Group|"
    r"Capital|Cements?|Power|Energy|Digital|Neu|Finance|Technologies|Health)\b|"
    r"\bshareholders?\s+approved\b|\bboard\s+(?:also\s+)?approved\b)",
    re.IGNORECASE,
)

# High-impact subject markers → bump the decision impactScore.
_URGENT_SUBJECT_RE = re.compile(
    r"\b(urgent|critical|priority|important|blocker|asap|escalat)", re.IGNORECASE
)

# Later-mention keywords used to derive a decision's outcomeStatus from the
# text of related decision/commitment records in the same subject/thread.
_REVERSED_RE = re.compile(
    r"\b(revert|reversed|changed (?:our|my) mind|on hold|cancell?ed|not going ahead|scrap(?:ping|ped))\b",
    re.IGNORECASE,
)
_SUCCESS_RE = re.compile(
    r"\b(done|shipped|completed|launched|delivered|closed|resolved|approved)\b",
    re.IGNORECASE,
)


def _norm_subject(s: str | None) -> str:
    """Strip Re:/Fwd:/whitespace so a thread's subject variants collapse to one key."""
    if not s:
        return ""
    return re.sub(r"^\s*(re|fwd|fw)\s*:\s*", "", s.strip(), flags=re.IGNORECASE).lower()


def _bucket(value: float, low: float, high: float) -> str:
    """Rule-based 3-way bucketization used for risk / likelihood fields."""
    if value >= high:
        return "high"
    if value >= low:
        return "medium"
    return "low"


def _is_automated_email(email: str | None) -> bool:
    """True for noreply/marketing/newsletter senders AND for malformed rows
    (parser leftovers with quotes/braces). Malformed drops here — the dashboard
    should never surface `'probe` or `'ajitesh@...'}`; those are ingestion bugs
    upstream, not people the founder should see.

    Layers of match (any hit → automated):
      1. Not a syntactically valid email (parser bug, probe data, garbage).
      2. `noreply`/`donotreply` substring in the local part (also catches
         middle-of-string like `analytics-noreply@`, `workspace-noreply@`).
      3. Anchored bulk-sender token in the local part (newsletter, marketing,
         support, hello, team, `the*team`, `beca`, ...).
      4. Marketing/notification token in the DOMAIN (`updates.foo.com`,
         `mail.foo.com`, `web-spicejet.com`).
      5. Whole-domain newsletter platforms (substack.com, mailchimp.com, ...).
    """
    # Empty / None is malformed too → drop. `""` and `None` never belong on the
    # dashboard.
    if not email:
        return True
    # Guard 1: malformed shape → treat as automated (drop). Any real person's
    # email will parse cleanly; anything else is upstream noise.
    if not _VALID_EMAIL_RE.match(email):
        return True
    local, _, domain = email.partition("@")
    domain = domain.lower()
    # Guard 2: noreply substring anywhere in local part.
    if _NOREPLY_SUBSTRING_RE.search(local):
        return True
    # Guard 3: anchored bulk-sender token in local part.
    if _AUTOMATED_LOCAL_RE.match(local):
        return True
    # Guard 4: marketing/notification token in domain (subdomain match).
    if _AUTOMATED_DOMAIN_RE.search(domain):
        return True
    # Guard 5: whole-domain newsletter platforms.
    if domain in _AUTOMATED_WHOLE_DOMAIN:
        return True
    return False


def _is_newsletter_subject(subject: str | None) -> bool:
    if not subject:
        return False
    return bool(_NEWSLETTER_SUBJECT_RE.search(subject))


def _is_third_party_decision_text(text: str | None) -> bool:
    """News-report style prose ('The government approved X') is a decision ABOUT
    someone else, not one the founder made. Distinct from newsletter subject —
    a third-party phrasing can leak in from a forwarded article on any subject."""
    if not text:
        return False
    return bool(_THIRD_PARTY_TEXT_RE.match(text))


def _viewer_email(p: Principal) -> str | None:
    """Best-effort read of the requesting user's email — used to filter the
    viewer out of their OWN People panel. API-key principals have no user, so
    self-detection silently no-ops for them (they see everyone)."""
    user = getattr(p, "user", None)
    email = getattr(user, "email", None) if user is not None else None
    return email.lower() if isinstance(email, str) else None


# ─────────────────────────── /decision-memory ────────────────────────────────


def _derive_decision(
    entity_id: uuid.UUID,
    records: list[Any],
    related_texts_by_subject: dict[str, list[str]],
) -> dict[str, Any]:
    """Roll a decision entity's fact history into one dashboard row.

    `records` is oldest-first for the same entity_id. `related_texts_by_subject`
    provides commitment/decision text that shares this decision's subject — that
    is our evidence pool for outcome derivation.
    """
    latest = records[-1]
    payload = latest.payload or {}
    subject_key = _norm_subject(payload.get("subject"))
    related = related_texts_by_subject.get(subject_key, [])
    evidence_count = len(records) + len(related)

    # Impact — bounded 0..1. Baseline 0.3, additive bumps.
    impact = 0.3
    if _URGENT_SUBJECT_RE.search(payload.get("subject") or ""):
        impact += 0.2
    if latest.confidence >= 0.7:
        impact += 0.2
    if len(records) >= 3:
        impact += 0.2  # repeated mention = more consequential
    if evidence_count >= 5:
        impact += 0.1
    impact = min(1.0, round(impact, 2))

    # Outcome — scan LATER text (records after this decision's first mention +
    # any related commitments/decisions in the same subject) for reversal or
    # success markers. Reversal wins ties (safer to surface a rollback).
    later_texts = [r.payload.get("text", "") for r in records[1:]] + related
    reversed_hit = any(_REVERSED_RE.search(t) for t in later_texts if t)
    success_hit = any(_SUCCESS_RE.search(t) for t in later_texts if t)
    outcome = "reversed" if reversed_hit else ("success" if success_hit else "pending")

    return {
        "id": str(entity_id),
        "text": payload.get("text", ""),
        "subject": payload.get("subject"),
        "via": payload.get("via") or "email",
        "capturedAt": records[0].event_timestamp.isoformat(),
        "lastMentionedAt": latest.event_timestamp.isoformat(),
        "confidence": round(latest.confidence, 3),
        "impactScore": impact,
        "outcomeStatus": outcome,
        "evidenceCount": evidence_count,
    }


@router.get("/decision-memory")
async def list_decisions(
    user_id: str | None = None,  # accepted for FE compat, ignored (org-scoped)
    limit: int = 200,
    p: Principal = Depends(require_capability(Capability.DECISION)),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Decision memory feed: each unique decision entity with derived impact
    + outcome. Frontend renders these on /dashboard/decisions and in the
    'What Navis learned' cards."""
    repo = MemoryRepository(db)
    # Overfetch — noise filters will drop most rows on a typical inbox.
    decision_rows = await repo.list_by_entity_type(
        p.org_id, EntityType.DECISION, limit=limit * 8
    )
    commitment_rows = await repo.list_by_entity_type(
        p.org_id, EntityType.COMMITMENT, limit=limit * 8
    )
    # Cross-reference: build the set of signal_event_ids whose sender is an
    # automated address (newsletter/noreply). Anything extracted from those
    # signals is noise — third-party news reporting, not a founder decision.
    person_rows = await repo.list_by_entity_type(
        p.org_id, EntityType.PERSON, limit=limit * 8
    )
    automated_signal_ids: set[uuid.UUID] = {
        r.source_signal_event_id
        for r in person_rows
        if (r.payload or {}).get("role") == "from"
        and _is_automated_email((r.payload or {}).get("email"))
    }

    def _is_noise(row) -> bool:
        payload = row.payload or {}
        # Extracted from a newsletter/automated sender → drop.
        if row.source_signal_event_id in automated_signal_ids:
            return True
        # Subject looks like a newsletter/digest → drop.
        if _is_newsletter_subject(payload.get("subject")):
            return True
        # Third-person news-report phrasing → drop.
        if _is_third_party_decision_text(payload.get("text")):
            return True
        return False

    # Fold: history per unique decision entity (oldest-first). Skip noise.
    by_decision: dict[uuid.UUID, list[Any]] = defaultdict(list)
    for row in decision_rows:
        if _is_noise(row):
            continue
        by_decision[row.entity_id].append(row)
    for records in by_decision.values():
        records.sort(key=lambda r: r.event_timestamp)

    # Evidence pool per subject — commitments AND older decisions share here.
    # Also filter commitments from automated senders so a newsletter's "we'll
    # be sending you..." doesn't fabricate evidence for a real decision.
    related_by_subject: dict[str, list[str]] = defaultdict(list)
    for row in commitment_rows:
        if row.source_signal_event_id in automated_signal_ids:
            continue
        text = (row.payload or {}).get("text")
        subj = _norm_subject((row.payload or {}).get("subject"))
        if text and subj:
            related_by_subject[subj].append(text)

    decisions = [
        _derive_decision(entity_id, records, related_by_subject)
        for entity_id, records in by_decision.items()
    ]
    # Newest last-mention first — matches "recent decisions" UX.
    decisions.sort(key=lambda d: d["lastMentionedAt"], reverse=True)
    decisions = decisions[:limit]

    return {
        "decisions": decisions,
        "total": len(decisions),
        "asOf": datetime.now(timezone.utc).isoformat(),
    }


# ────────────────────────────── /memory/people ───────────────────────────────


def _role_hint(email: str) -> str:
    """Cheap classification — domain-based. Team detection would need org
    domain config (not modeled yet) so we return 'external' vs 'personal'
    and let the FE style the badge from there."""
    if "@" not in email:
        return "unknown"
    domain = email.rsplit("@", 1)[1].lower()
    if domain in _FREE_DOMAINS:
        return "personal"
    return "external"


def _derive_person(
    entity_id: uuid.UUID,
    person_records: list[Any],
    relationship_records: list[Any],
    slipping_by_speaker: dict[str, int],
) -> dict[str, Any]:
    """Roll one person's PERSON + RELATIONSHIP facts into a dashboard card.

    `person_records` is newest-first for the same entity_id (kept dense — we
    intentionally count each fact as a signal). `relationship_records` are the
    RELATIONSHIP rows for this person's email. `slipping_by_speaker` counts
    commitment records where the person spoke and the subject has aged past
    the reasoning threshold — proxy for escalation risk.
    """
    latest = person_records[0]  # newest first
    oldest = person_records[-1]
    payload = latest.payload or {}
    email = payload.get("email") or ""

    inbound = sum(1 for r in relationship_records
                  if (r.payload or {}).get("direction") == "inbound")
    outbound = sum(1 for r in relationship_records
                   if (r.payload or {}).get("direction") == "outbound")
    total_contact = inbound + outbound

    # Response likelihood: how often the org has replied to this person.
    # outbound / (inbound+1) as a smoothed ratio; low means "they email us and
    # we mostly don't reply", high means active back-and-forth.
    response_ratio = outbound / (inbound + 1) if total_contact else 0.0
    response_likelihood = _bucket(response_ratio, low=0.25, high=0.75) if total_contact else "unknown"

    # Escalation risk: slipping commitments spoken by this person.
    slipping = slipping_by_speaker.get(email, 0)
    escalation_risk = _bucket(float(slipping), low=1, high=3)

    return {
        "id": str(entity_id),
        "email": email,
        "name": payload.get("name") or None,
        "roleHint": _role_hint(email),
        "signalCount": len(person_records) + len(relationship_records),
        "firstSeenAt": oldest.event_timestamp.isoformat(),
        "lastSeenAt": latest.event_timestamp.isoformat(),
        "inboundCount": inbound,
        "outboundCount": outbound,
        "responseLikelihood": response_likelihood,
        "escalationRisk": escalation_risk,
        "replyLatencyHours": None,  # requires signal-level thread timing; deferred
    }


@router.get("/memory/people")
async def list_people(
    user_id: str | None = None,  # accepted for FE compat, ignored (org-scoped)
    limit: int = 200,
    p: Principal = Depends(require_capability(Capability.DECISION)),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """People memory feed: one card per unique person we've observed, with
    derived escalation + response signals. Frontend renders these on the
    Memory panels and the Organization pulse tiles."""
    repo = MemoryRepository(db)
    person_rows = await repo.list_by_entity_type(
        p.org_id, EntityType.PERSON, limit=limit * 8
    )
    relationship_rows = await repo.list_by_entity_type(
        p.org_id, EntityType.RELATIONSHIP, limit=limit * 8
    )
    commitment_rows = await repo.list_by_entity_type(
        p.org_id, EntityType.COMMITMENT, limit=limit * 8
    )

    viewer = _viewer_email(p)

    # Fold PERSON records by entity_id (newest-first per group for direct read).
    by_person: dict[uuid.UUID, list[Any]] = defaultdict(list)
    for row in person_rows:
        by_person[row.entity_id].append(row)
    # list_by_entity_type already sorts DESC by event_timestamp — preserved.

    # Fold RELATIONSHIP records by email (relationship rows dedup on email + direction).
    rel_by_email: dict[str, list[Any]] = defaultdict(list)
    for row in relationship_rows:
        email = (row.payload or {}).get("person_email")
        if email:
            rel_by_email[email].append(row)

    # Count slipping commitments per speaker. This is a lightweight proxy for
    # what the RiskEngine emits; we compute it here to avoid running the full
    # reasoning pass on every people request. Threshold matches risk.py (>=5d).
    # Skip commitments spoken by AUTOMATED addresses so newsletters can't
    # push themselves onto the escalation ladder (Fix 4).
    slipping_by_speaker: dict[str, int] = defaultdict(int)
    now = datetime.now(timezone.utc)
    for row in commitment_rows:
        speaker = (row.payload or {}).get("speaker_email")
        if not speaker or _is_automated_email(speaker):
            continue
        age_days = (now - row.event_timestamp).total_seconds() / 86400.0
        if age_days >= 5.0:
            slipping_by_speaker[speaker] += 1

    people = []
    for entity_id, records in by_person.items():
        email = (records[0].payload or {}).get("email") or ""
        # Fix 1: hide automated/no-reply senders (they aren't people).
        if _is_automated_email(email):
            continue
        # Fix 3: hide the viewer's own address so they don't sit at the top
        # of their own People panel.
        if viewer and email.lower() == viewer:
            continue
        people.append(
            _derive_person(
                entity_id,
                records,
                rel_by_email.get(email, []),
                slipping_by_speaker,
            )
        )

    # Most recently seen first — matches the "Recently active" strip UX.
    people.sort(key=lambda x: x["lastSeenAt"], reverse=True)
    people = people[:limit]

    return {
        "people": people,
        "total": len(people),
        "asOf": datetime.now(timezone.utc).isoformat(),
    }
