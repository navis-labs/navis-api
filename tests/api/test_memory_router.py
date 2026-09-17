"""Locks the /memory/people and /decision-memory read endpoints.

These endpoints ship the 'People / Customers / Investors' and 'What Navis learned'
surfaces on the dashboard. They MUST:
  * Return org-scoped rows only (no cross-tenant leakage).
  * Be capability-gated on DECISION (same class as brief/context).
  * Fold memory_records into ONE row per unique entity (not one per fact).
  * Derive impactScore / outcomeStatus / escalationRisk deterministically
    (pure functions of the records) so replay is stable and there is no LLM
    hidden cost in the path.
"""
from __future__ import annotations

import inspect
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from navis.api import memory_router as R
from navis.memory.envelope import EntityType
from navis.organization.contract import Capability


ORG = uuid.UUID("11111111-1111-1111-1111-111111111111")


# ─────────────────────── minimal record + repo stubs ─────────────────────────

@dataclass
class _Rec:
    """Stands in for MemoryRecordRow — the handlers only read these attributes.

    `source_signal_event_id` defaults to a FRESH random UUID per row so noise
    cross-referencing (drop-decisions-from-automated-senders) is naturally OFF
    unless a test opts in by pinning matching ids across PERSON+DECISION rows.
    """
    entity_id: uuid.UUID
    event_timestamp: datetime
    payload: dict
    confidence: float = 0.5
    source_signal_event_id: uuid.UUID = field(default_factory=uuid.uuid4)


class _Repo:
    """Feeds canned record lists per entity_type. Records are returned newest-first
    (matches list_by_entity_type's ORDER BY DESC contract)."""

    def __init__(self, by_type: dict[EntityType, list[_Rec]]) -> None:
        self._by_type = by_type

    async def list_by_entity_type(
        self, org_id: uuid.UUID, entity_type: EntityType, *, limit: int = 500
    ):
        rows = list(self._by_type.get(entity_type, []))
        rows.sort(key=lambda r: r.event_timestamp, reverse=True)
        return rows[:limit]


@dataclass
class _Principal:
    org_id: uuid.UUID = ORG


def _install_repo(monkeypatch, records_by_type):
    """Route MemoryRepository() to our stub — the handler still constructs it."""
    monkeypatch.setattr(R, "MemoryRepository", lambda db: _Repo(records_by_type))


def _uuid(seed: str) -> uuid.UUID:
    return uuid.uuid5(uuid.NAMESPACE_DNS, seed)


# ────────────────────────── /decision-memory ─────────────────────────────────

async def test_decisions_empty_org_returns_empty_list(monkeypatch):
    """No decisions → the endpoint MUST still return the envelope (not 404),
    so the frontend renders an empty state instead of a broken card."""
    _install_repo(monkeypatch, {})
    resp = await R.list_decisions(p=_Principal(), db=None)  # type: ignore[arg-type]
    assert resp["decisions"] == []
    assert resp["total"] == 0
    assert "asOf" in resp


async def test_decisions_folds_history_into_one_row_per_entity(monkeypatch):
    """Two facts about the SAME decision entity MUST collapse to one row —
    otherwise the 'What Navis learned' list would double-count repeated mentions."""
    eid = _uuid("decision:q3-launch")
    t0 = datetime(2026, 7, 1, 10, tzinfo=timezone.utc)
    _install_repo(monkeypatch, {
        EntityType.DECISION: [
            _Rec(eid, t0,                 {"text": "we decided to postpone", "subject": "Q3 launch"}),
            _Rec(eid, t0 + timedelta(days=1), {"text": "confirming: we decided", "subject": "Re: Q3 launch"}),
        ]
    })
    resp = await R.list_decisions(p=_Principal(), db=None)  # type: ignore[arg-type]
    assert len(resp["decisions"]) == 1
    row = resp["decisions"][0]
    assert row["id"] == str(eid)
    # oldest = capturedAt, newest = lastMentionedAt — that ordering is what the
    # timeline surface groups by.
    assert row["capturedAt"] < row["lastMentionedAt"]


async def test_decisions_impact_score_bumps_on_urgent_subject(monkeypatch):
    """A decision whose subject flags urgency should surface higher in the
    dashboard rank. Impact is bounded 0..1 and monotonic in the bumps."""
    plain = _uuid("plain")
    urgent = _uuid("urgent")
    t = datetime(2026, 7, 1, tzinfo=timezone.utc)
    _install_repo(monkeypatch, {
        EntityType.DECISION: [
            _Rec(plain,  t, {"text": "we decided X",       "subject": "roadmap sync"}),
            _Rec(urgent, t, {"text": "we decided Y",       "subject": "URGENT: legal ask"}),
        ]
    })
    resp = await R.list_decisions(p=_Principal(), db=None)  # type: ignore[arg-type]
    scores = {d["id"]: d["impactScore"] for d in resp["decisions"]}
    assert scores[str(urgent)] > scores[str(plain)]
    assert 0.0 <= scores[str(plain)] <= 1.0
    assert 0.0 <= scores[str(urgent)] <= 1.0


async def test_decisions_outcome_reversed_beats_success_in_same_subject(monkeypatch):
    """If a later commitment in the SAME subject signals a reversal, outcome
    MUST land on 'reversed' — surfacing a rollback is safer than a stale 'success'."""
    eid = _uuid("decision:launch")
    t = datetime(2026, 7, 1, tzinfo=timezone.utc)
    _install_repo(monkeypatch, {
        EntityType.DECISION: [
            _Rec(eid, t, {"text": "we decided to launch", "subject": "Launch plan"}),
        ],
        EntityType.COMMITMENT: [
            _Rec(_uuid("c1"), t + timedelta(hours=2),
                 {"text": "shipping tomorrow", "subject": "Launch plan"}),
            _Rec(_uuid("c2"), t + timedelta(days=1),
                 {"text": "actually reverting — on hold", "subject": "Re: Launch plan"}),
        ],
    })
    resp = await R.list_decisions(p=_Principal(), db=None)  # type: ignore[arg-type]
    assert resp["decisions"][0]["outcomeStatus"] == "reversed"


async def test_decisions_outcome_pending_when_no_followup(monkeypatch):
    """No later evidence → pending. This is the honest default — the previous
    hardcoded '88%' fallback lied to founders about follow-through."""
    eid = _uuid("d")
    t = datetime(2026, 7, 1, tzinfo=timezone.utc)
    _install_repo(monkeypatch, {
        EntityType.DECISION: [_Rec(eid, t, {"text": "we decided X", "subject": "S"})],
    })
    resp = await R.list_decisions(p=_Principal(), db=None)  # type: ignore[arg-type]
    assert resp["decisions"][0]["outcomeStatus"] == "pending"


# ─────────────────────────── /memory/people ─────────────────────────────────

async def test_people_empty_org_returns_empty_list(monkeypatch):
    _install_repo(monkeypatch, {})
    resp = await R.list_people(p=_Principal(), db=None)  # type: ignore[arg-type]
    assert resp["people"] == []
    assert resp["total"] == 0


async def test_people_folds_by_entity_id_and_counts_signals(monkeypatch):
    """Multiple PERSON facts for one email MUST collapse to one card; signalCount
    counts total observations (PERSON + RELATIONSHIP) so the dashboard 'activity'
    heat maps stay honest."""
    sarah = _uuid("sarah@acme.com")
    t = datetime(2026, 7, 1, tzinfo=timezone.utc)
    _install_repo(monkeypatch, {
        EntityType.PERSON: [
            _Rec(sarah, t + timedelta(days=1), {"email": "sarah@acme.com", "name": "Sarah"}),
            _Rec(sarah, t,                     {"email": "sarah@acme.com", "name": "Sarah"}),
        ],
        EntityType.RELATIONSHIP: [
            _Rec(_uuid("r1"), t, {"person_email": "sarah@acme.com", "direction": "inbound"}),
            _Rec(_uuid("r2"), t, {"person_email": "sarah@acme.com", "direction": "outbound"}),
        ],
    })
    resp = await R.list_people(p=_Principal(), db=None)  # type: ignore[arg-type]
    assert len(resp["people"]) == 1
    row = resp["people"][0]
    assert row["email"] == "sarah@acme.com"
    assert row["signalCount"] == 4  # 2 PERSON + 2 RELATIONSHIP
    assert row["inboundCount"] == 1
    assert row["outboundCount"] == 1


async def test_people_role_hint_classifies_by_domain(monkeypatch):
    """Free-domain emails are 'personal' (unknown intent), everything else is
    'external' (likely customer/partner). Team detection needs org config so
    we intentionally do NOT guess it here."""
    biz = _uuid("ceo@acme.com")
    consumer = _uuid("me@gmail.com")
    t = datetime(2026, 7, 1, tzinfo=timezone.utc)
    _install_repo(monkeypatch, {
        EntityType.PERSON: [
            _Rec(biz, t,      {"email": "ceo@acme.com"}),
            _Rec(consumer, t, {"email": "me@gmail.com"}),
        ],
    })
    resp = await R.list_people(p=_Principal(), db=None)  # type: ignore[arg-type]
    by_email = {p["email"]: p for p in resp["people"]}
    assert by_email["ceo@acme.com"]["roleHint"] == "external"
    assert by_email["me@gmail.com"]["roleHint"] == "personal"


async def test_people_escalation_risk_bumps_on_aged_commitments(monkeypatch):
    """Person with multiple aged commitments (past the reasoning threshold of
    5 days) MUST land on 'high' escalation. This is the proxy the frontend uses
    for the 'red' escalation badge."""
    sarah = _uuid("sarah@acme.com")
    now = datetime.now(timezone.utc)
    _install_repo(monkeypatch, {
        EntityType.PERSON: [
            _Rec(sarah, now, {"email": "sarah@acme.com", "name": "Sarah"}),
        ],
        EntityType.COMMITMENT: [
            _Rec(_uuid("c1"), now - timedelta(days=10),
                 {"speaker_email": "sarah@acme.com", "text": "I'll send it"}),
            _Rec(_uuid("c2"), now - timedelta(days=8),
                 {"speaker_email": "sarah@acme.com", "text": "we'll follow up"}),
            _Rec(_uuid("c3"), now - timedelta(days=7),
                 {"speaker_email": "sarah@acme.com", "text": "I will circle back"}),
        ],
    })
    resp = await R.list_people(p=_Principal(), db=None)  # type: ignore[arg-type]
    assert resp["people"][0]["escalationRisk"] == "high"


async def test_people_response_likelihood_high_on_active_outbound(monkeypatch):
    """When the org replies more than they're contacted, likelihood is 'high'.
    This drives the 'you're likely to hear back' hint in the People panel."""
    sarah = _uuid("sarah@acme.com")
    t = datetime(2026, 7, 1, tzinfo=timezone.utc)
    _install_repo(monkeypatch, {
        EntityType.PERSON: [
            _Rec(sarah, t, {"email": "sarah@acme.com"}),
        ],
        EntityType.RELATIONSHIP: [
            _Rec(_uuid(f"o{i}"), t,
                 {"person_email": "sarah@acme.com", "direction": "outbound"}) for i in range(4)
        ] + [
            _Rec(_uuid("i1"), t, {"person_email": "sarah@acme.com", "direction": "inbound"}),
        ],
    })
    resp = await R.list_people(p=_Principal(), db=None)  # type: ignore[arg-type]
    assert resp["people"][0]["responseLikelihood"] == "high"


# ─────────────────────────── helpers: pure derivations ───────────────────────

def test_role_hint_returns_unknown_for_malformed_email():
    """Guard against a payload where the extractor mis-parses — we should still
    return a card with 'unknown' role rather than crash."""
    assert R._role_hint("no-at-sign") == "unknown"


def test_bucket_boundary_values():
    """Boundary is inclusive of low → 'medium' at low, 'high' at high.
    Downstream frontend renders the badge color directly from this string."""
    assert R._bucket(0.0,  low=0.25, high=0.75) == "low"
    assert R._bucket(0.25, low=0.25, high=0.75) == "medium"
    assert R._bucket(0.75, low=0.25, high=0.75) == "high"


# ─────────────────────────── noise filters (Fix 1-4) ────────────────────────

async def test_people_filters_automated_local_part(monkeypatch):
    """noreply@ / notification@ / hello@ etc. are automated senders — they MUST
    NOT surface as 'people'. Otherwise a founder's People panel is 90% mailing
    lists and the panel becomes useless."""
    real = _uuid("harsha@slashy.com")
    t = datetime(2026, 7, 1, tzinfo=timezone.utc)
    _install_repo(monkeypatch, {
        EntityType.PERSON: [
            _Rec(real,               t, {"email": "harsha@slashy.com"}),
            _Rec(_uuid("noreply"),   t, {"email": "noreply@digest.groww.in"}),
            _Rec(_uuid("marketing"), t, {"email": "marketing@engage.canva.com"}),
            _Rec(_uuid("hello"),     t, {"email": "hello@bolt.new"}),
            _Rec(_uuid("security"),  t, {"email": "security@facebookmail.com"}),
        ],
    })
    resp = await R.list_people(p=_Principal(), db=None)  # type: ignore[arg-type]
    emails = [p["email"] for p in resp["people"]]
    assert emails == ["harsha@slashy.com"]


async def test_people_filters_marketing_subdomain(monkeypatch):
    """`sender@updates.foo.com` / `sender@mail.foo.com` are bulk-marketing
    domains — filter them by the DOMAIN pattern even when the local part
    looks human ('hello', 'info' also caught but a real-looking local should
    still be dropped when the subdomain is a bulk mailer)."""
    real = _uuid("ceo@acme.com")
    t = datetime(2026, 7, 1, tzinfo=timezone.utc)
    _install_repo(monkeypatch, {
        EntityType.PERSON: [
            _Rec(real,             t, {"email": "ceo@acme.com"}),
            _Rec(_uuid("m1"),      t, {"email": "founders@updates.foo.com"}),
            _Rec(_uuid("m2"),      t, {"email": "connect@mail.reliancedigital.in"}),
            _Rec(_uuid("m3"),      t, {"email": "team@notification.priority.acme.com"}),
        ],
    })
    resp = await R.list_people(p=_Principal(), db=None)  # type: ignore[arg-type]
    emails = [p["email"] for p in resp["people"]]
    assert emails == ["ceo@acme.com"]


async def test_people_hides_viewer_own_email(monkeypatch):
    """The founder MUST NOT appear in their own People panel — that seat is
    reserved for the OTHER side of every conversation. Fix 3."""
    me = _uuid("me@acme.com")
    other = _uuid("harsha@slashy.com")
    t = datetime(2026, 7, 1, tzinfo=timezone.utc)

    # Principal with a user whose email matches one of the person entities.
    @dataclass
    class _User:
        email: str = "me@acme.com"

    p = _Principal()
    p.user = _User()  # type: ignore[attr-defined]

    _install_repo(monkeypatch, {
        EntityType.PERSON: [
            _Rec(me,    t, {"email": "me@acme.com"}),
            _Rec(other, t, {"email": "harsha@slashy.com"}),
        ],
    })
    resp = await R.list_people(p=p, db=None)  # type: ignore[arg-type]
    emails = [row["email"] for row in resp["people"]]
    assert emails == ["harsha@slashy.com"]


async def test_decisions_drops_newsletter_subjects(monkeypatch):
    """A decision whose subject smells like a digest/newsletter is (a) not the
    founder's decision and (b) usually third-party news being reported. Drop it."""
    t = datetime(2026, 7, 1, tzinfo=timezone.utc)
    _install_repo(monkeypatch, {
        EntityType.DECISION: [
            _Rec(_uuid("real"),  t, {"text": "we decided to postpone", "subject": "Q3 planning"}),
            _Rec(_uuid("news1"), t, {"text": "we decided X", "subject": "ICICI Bank raises $1 billion, ... & more - Groww Digest"}),
            _Rec(_uuid("news2"), t, {"text": "we approved Y", "subject": "Monday Briefing"}),
            _Rec(_uuid("news3"), t, {"text": "we approved Z", "subject": "Weekly roundup"}),
        ],
    })
    resp = await R.list_decisions(p=_Principal(), db=None)  # type: ignore[arg-type]
    ids = [d["id"] for d in resp["decisions"]]
    assert ids == [str(_uuid("real"))]


async def test_decisions_drops_third_party_news_text(monkeypatch):
    """Prose that starts with 'The government approved' or 'Aurobindo Pharma:'
    is a NEWS REPORT — it made it into memory via a forwarded article. Drop it
    even if the subject looks clean."""
    t = datetime(2026, 7, 1, tzinfo=timezone.utc)
    _install_repo(monkeypatch, {
        EntityType.DECISION: [
            _Rec(_uuid("real"), t,   {"text": "we've agreed to hire", "subject": "team plan"}),
            _Rec(_uuid("news1"), t,  {"text": "The government approved two railway lines.", "subject": "notes"}),
            _Rec(_uuid("news2"), t,  {"text": "Aurobindo Pharma: subsidiary approved acquisition.", "subject": "notes"}),
            _Rec(_uuid("news3"), t,  {"text": "Board also approved raising Rs 20,000 crore.", "subject": "notes"}),
        ],
    })
    resp = await R.list_decisions(p=_Principal(), db=None)  # type: ignore[arg-type]
    ids = [d["id"] for d in resp["decisions"]]
    assert ids == [str(_uuid("real"))]


async def test_decisions_drops_when_source_signal_sender_is_automated(monkeypatch):
    """If the SOURCE email was sent by a noreply address, every fact extracted
    from it is bulk noise. Cross-reference via source_signal_event_id and drop
    all facts sharing that signal id — even if their own subject/text look clean."""
    t = datetime(2026, 7, 1, tzinfo=timezone.utc)
    news_signal = uuid.uuid4()
    real_signal = uuid.uuid4()
    _install_repo(monkeypatch, {
        # The PERSON record with role='from' is how we know the sender.
        EntityType.PERSON: [
            _Rec(_uuid("sender-news"), t,
                 {"email": "noreply@digest.groww.in", "role": "from"},
                 source_signal_event_id=news_signal),
            _Rec(_uuid("sender-real"), t,
                 {"email": "sarah@acme.com", "role": "from"},
                 source_signal_event_id=real_signal),
        ],
        EntityType.DECISION: [
            _Rec(_uuid("news"), t, {"text": "we approved X", "subject": "clean subject"},
                 source_signal_event_id=news_signal),
            _Rec(_uuid("real"), t, {"text": "we've agreed to ship", "subject": "clean subject"},
                 source_signal_event_id=real_signal),
        ],
    })
    resp = await R.list_decisions(p=_Principal(), db=None)  # type: ignore[arg-type]
    ids = [d["id"] for d in resp["decisions"]]
    assert ids == [str(_uuid("real"))]


async def test_escalation_ignores_commitments_from_automated_speakers(monkeypatch):
    """A newsletter body contains phrases like 'we'll be launching next week'
    which the commitment extractor legitimately picks up. Fix 4 asserts that
    those 'commitments' (a) never appear in the slipping-by-speaker index at
    all and (b) never surface as an escalated PERSON row on the dashboard.

    Concretely: even if we DIDN'T filter the automated PERSON out (Fix 1),
    they'd have zero slipping count because the loop skips automated
    speaker_emails — so escalation would still be 'low'. This test proves
    both guarantees hold together: the newsletter person is gone, the human's
    real (own) slipping count is not polluted by the newsletter's 5 spurious
    commitments (which share NO speaker with the human anyway)."""
    news = _uuid("noreply@digest.groww.in")
    real = _uuid("harsha@slashy.com")
    now = datetime.now(timezone.utc)
    _install_repo(monkeypatch, {
        EntityType.PERSON: [
            _Rec(news, now, {"email": "noreply@digest.groww.in"}),
            _Rec(real, now, {"email": "harsha@slashy.com"}),
        ],
        # 5 aged commitments 'spoken' by the newsletter — pre-Fix-4 these
        # would inflate slipping_by_speaker['noreply@...'] to 5 (high risk).
        EntityType.COMMITMENT: [
            _Rec(_uuid(f"nc{i}"), now - timedelta(days=10),
                 {"speaker_email": "noreply@digest.groww.in", "text": "we'll be sending"})
            for i in range(5)
        ],
    })
    resp = await R.list_people(p=_Principal(), db=None)  # type: ignore[arg-type]
    # Newsletter is filtered by Fix 1 — so it never appears on the dashboard,
    # regardless of what its aged-commitment count would have been.
    assert [p["email"] for p in resp["people"]] == ["harsha@slashy.com"]
    # And the human's escalation is not falsely bumped by any of the newsletter's
    # commitments (they're indexed under a different speaker email anyway;
    # zero slipping for harsha → 'low').
    assert resp["people"][0]["escalationRisk"] == "low"


def test_is_automated_covers_documented_noise_patterns():
    """Lock the specific patterns we observed in the founder's real inbox
    (screenshot / demo audit) — if the regex is loosened later, this fails
    loudly and we re-audit."""
    must_flag = [
        # Round 1 (initial anchor-based filter)
        "noreply@digest.groww.in",
        "notification@priority.facebookmail.com",
        "hello@updates.pepperfry.com",
        "newsletter@faucetcrypto.com",
        "donotreply@usebraintrust.com",
        "friendsuggestion@facebookmail.com",
        "security@facebookmail.com",
        "recommendations@discover.pinterest.com",
        "marketing@engage.canva.com",
        "info@connect.netmeds.com",
        "mail@mail.adobe.com",
        # Round 2 (added after real-data audit)
        # noreply in the middle of the local part:
        "workspace-noreply@google.com",
        "analytics-noreply@google.com",
        "google-noreply@google.com",
        # 'the*team' bulk construction:
        "thehubspotteam@hubspot.com",
        # substack whole-domain:
        "speedrun@substack.com",
        # `web-` subdomain (bulk mailer for airlines/retailers):
        "spicejet@web-spicejet.com",
        # `beca` bulk mailer:
        "beca-joven@cefoto.es",
        # Malformed rows (parser bug / probe leakage) MUST drop:
        "'probe",
        "'ajiteshkr533@gmail.com'}",
        "'probe-sender@example.com'}",
    ]
    for e in must_flag:
        assert R._is_automated_email(e), f"MUST be flagged: {e}"

    # And these humans MUST slip through unchanged.
    for e in [
        "harsha@slashy.com",
        "sarah@acme.com",
        "thilak.jk@abmrc.com",
        "ericsimons@stackblitz.com",
        "mulayamabmrc@gmail.com",
        "founders@slashy.com",  # 'founders' is not a bulk token — real address
        "mail@fundizr.com",     # 'mail' local part gone, but this pattern:
                                # a person choosing 'mail' as their mailbox
                                # — trade-off: we drop this too, treat as automated.
                                # NOTE: if this changes, update _AUTOMATED_LOCAL_RE.
    ]:
        # `mail@fundizr.com` is expected to be flagged today because `mail`
        # is a bulk token in the local part. If we ever whitelist real
        # single-word local parts, remove it from this list.
        if e == "mail@fundizr.com":
            assert R._is_automated_email(e), (
                "`mail@` currently drops as bulk — remove from list if that changes"
            )
        else:
            assert not R._is_automated_email(e), f"MUST NOT be flagged: {e}"


def test_valid_email_shape_gate_drops_parser_bugs():
    """Rows with quotes, braces, or missing structure are parser leftovers
    from a fixed bug (`'probe`, `'sender@example.com'}`, etc.). They MUST NOT
    surface as people — they aren't real addresses."""
    for bad in ["'probe", "'ajitesh@gmail.com'}", "no-at-sign", "", "@only.com", "no.tld@x"]:
        assert R._is_automated_email(bad), f"malformed MUST drop: {bad!r}"


# ─────────────────────────── auth wiring + registration ──────────────────────

def test_endpoints_gated_on_decision_capability():
    """A refactor that drops or downgrades the guard would silently expose
    org memory to anyone who can hit the API. Assert the Depends is present
    on both handlers."""
    for handler in (R.list_decisions, R.list_people):
        sig = inspect.signature(handler)
        dep = sig.parameters["p"].default
        # `require_capability(Capability.DECISION)` closes over the capability;
        # the closure carries the enum value we care about.
        closed = [
            c.cell_contents for c in (dep.dependency.__closure__ or ())
        ]
        assert any(c is Capability.DECISION for c in closed), (
            f"{handler.__name__} not gated on Capability.DECISION"
        )


def test_router_registers_expected_paths():
    """These paths are the frontend contract — a rename would resurrect the
    404s this endpoint was built to fix."""
    paths = {r.path for r in R.router.routes}
    assert "/decision-memory" in paths
    assert "/memory/people" in paths


