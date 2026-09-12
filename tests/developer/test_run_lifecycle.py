"""The agent execution lifecycle — vocabulary, state machine, and scrubbing.

These are the parts that need no database, and they are the parts most worth
pinning: a state machine that admits one wrong edge lets a finished run be
rewritten, and a scrubber that misses one shape puts a customer's third-party
credential in Navis's database and on Navis's screens.

Schema parity between the ORM models and migration 0048 is asserted in
`tests/integration/test_agent_lifecycle_postgres.py`, which needs a disposable
Postgres and is skipped without one.
"""
from __future__ import annotations

import json

import pytest

from navis.developer import lifecycle as L
from navis.developer.sanitize import (
    REDACTED,
    PayloadTooLarge,
    sanitize_metadata,
    sanitize_text,
)


# ── Vocabulary ──────────────────────────────────────────────────────────────


def test_the_vocabularies_are_closed_and_match_their_enums():
    """The frozensets feed database CHECK constraints; drift between them and
    the enums would mean the edge accepts what the database rejects."""
    assert L.TASK_STATUSES == {s.value for s in L.TaskStatus}
    assert L.RUN_STATUSES == {s.value for s in L.RunStatus}
    assert L.STEP_TYPES == {s.value for s in L.StepType}
    assert L.STEP_STATUSES == {s.value for s in L.StepStatus}
    assert L.TOOL_KINDS == {k.value for k in L.ToolKind}
    assert L.ERROR_CATEGORIES == {c.value for c in L.ErrorCategory}
    assert L.OUTCOME_STATUSES == {s.value for s in L.OutcomeStatus}


def test_run_and_outcome_statuses_are_different_vocabularies():
    """A run's status answers "did the process finish"; an outcome's answers
    "was the result any good". Sharing a vocabulary is how the second question
    stops being askable."""
    assert L.RUN_STATUSES != L.OUTCOME_STATUSES
    assert "success" not in L.RUN_STATUSES
    assert "completed" not in L.OUTCOME_STATUSES


def test_an_unjudged_outcome_has_a_word_of_its_own():
    """`unknown` must exist, because nothing evaluates agent outcomes yet and
    recording an unjudged result as a success would poison the first metric
    anyone builds on this table."""
    assert "unknown" in L.OUTCOME_STATUSES


# ── The run state machine ───────────────────────────────────────────────────


def test_every_status_has_a_transition_entry():
    """A status missing from the table resolves to "no edges", which is a silent
    denial rather than a decided one."""
    assert set(L.ALLOWED_RUN_TRANSITIONS) == L.RUN_STATUSES


def test_terminal_runs_have_no_outgoing_edges():
    """THE LEDGER PROPERTY. A finished run cannot move again, which is what
    stops its timeline being rewritten after the fact."""
    for status in L.TERMINAL_RUN_STATUSES:
        assert L.ALLOWED_RUN_TRANSITIONS[status] == frozenset()
        for target in L.RUN_STATUSES:
            assert not L.may_transition(status, target)


def test_the_happy_path_is_permitted():
    assert L.may_transition("queued", "running")
    assert L.may_transition("running", "completed")


@pytest.mark.parametrize(
    "current,target",
    [
        ("completed", "running"),   # re-opening a finished run
        ("failed", "completed"),    # laundering a failure into a success
        ("cancelled", "running"),
        ("running", "queued"),      # moving backwards
        ("completed", "completed"), # idempotent-looking, still a rewrite
    ],
)
def test_forbidden_transitions_are_refused(current: str, target: str):
    assert not L.may_transition(current, target)


def test_an_unknown_status_cannot_transition_anywhere():
    """A row written before a vocabulary change must not become movable by
    accident; refusing is the safe answer."""
    assert not L.may_transition("not-a-status", "running")


def test_a_queued_run_can_fail_without_ever_running():
    """A run rejected by its own environment before it starts is a real
    situation — the model has to hold it rather than force a fake `running`."""
    assert L.may_transition("queued", "failed")
    assert L.may_transition("queued", "cancelled")


# ── Scrubbing: the security-critical half ───────────────────────────────────


@pytest.mark.parametrize(
    "key",
    [
        # Every separator spelling, because the first version of the
        # deny-list wrote its stems with underscores and let `x-api-key`
        # through untouched.
        "authorization", "Authorization", "AUTHORIZATION",
        "x-api-key", "X-Api-Key", "apiKey", "api_key", "API_KEY",
        "provider.api.key", "access_token", "accessToken", "refresh-token",
        "password", "client_secret", "cookie", "private_key", "privateKey",
        "signature", "bearer", "otp", "auth", "sessionId",
    ],
)
def test_a_sensitive_key_has_its_value_destroyed(key: str):
    """An agent reporting an authenticated tool call WILL eventually include the
    header it sent. Nobody decides to do that; it is what happens when a model
    serializes a request object."""
    out = sanitize_metadata({key: "super-secret-value"}, field="request_metadata")

    assert out[key] == REDACTED
    assert "super-secret-value" not in json.dumps(out)


@pytest.mark.parametrize(
    "value",
    [
        "navis_0123456789abcdef_thesecretpart",          # Navis's own format
        "Bearer eyJhbGciOiJIUzI1NiJ9.abcdefghij.sig",
        "sk-abcdefghijklmnopqrstuvwx",
        "ghp_abcdefghijklmnopqrstuvwxyz012345",
        "xoxb-1234567890-abcdefghij",
        "AKIAIOSFODNN7EXAMPLE",
        "AIzaSyA0000000000000000000000000000000",
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.abcdefg",
        "-----BEGIN RSA PRIVATE KEY-----",
    ],
)
def test_a_credential_shaped_value_is_destroyed_whatever_it_is_called(value: str):
    """The deny-list on KEY NAMES cannot be complete — the field is frequently
    called `h`, `opts` or `arg2`. This is the half that catches those."""
    out = sanitize_metadata({"h": value}, field="request_metadata")

    assert out["h"] == REDACTED
    assert value not in json.dumps(out)


def test_navis_never_leaks_its_own_credential_format():
    """The platform disclosing its own keys through its own run ledger would be
    the worst version of this bug."""
    leaked = "navis_deadbeefdeadbeef_supersecretsecret"
    out = sanitize_metadata({"note": f"called with {leaked}"}, field="request_metadata")

    assert leaked not in json.dumps(out)


def test_scrubbing_reaches_into_nested_structures():
    """A header dict inside a request object is the common shape, not the
    exotic one."""
    out = sanitize_metadata(
        {"request": {"url": "https://api.example.com", "headers": {"authorization": "Bearer xyz"}}},
        field="request_metadata",
    )

    assert out["request"]["headers"]["authorization"] == REDACTED
    assert "Bearer xyz" not in json.dumps(out)
    # The non-secret parts survive: a redacted record still has to be useful.
    assert out["request"]["url"] == "https://api.example.com"


def test_scrubbing_reaches_into_lists():
    out = sanitize_metadata(
        {"calls": [{"token": "abc"}, {"safe": "value"}]}, field="request_metadata"
    )

    assert out["calls"][0]["token"] == REDACTED
    assert out["calls"][1]["safe"] == "value"


def test_harmless_metadata_survives_untouched():
    """A scrubber that mangles ordinary data is a scrubber people route around."""
    payload = {"status_code": 200, "rows": 42, "ok": True, "name": "billing_search", "n": None}
    assert sanitize_metadata(payload, field="response_metadata") == payload


def test_depth_is_bounded():
    """Depth is a denial-of-service axis of its own — it costs recursion on every
    read as well as every write."""
    deep: dict = {"a": {"b": {"c": {"d": {"e": {"f": "too deep"}}}}}}
    out = sanitize_metadata(deep, field="request_metadata")

    assert json.dumps(out).count("too deep") == 0


def test_an_oversized_payload_is_refused_not_truncated():
    """Truncating produces a record that looks complete and is not."""
    with pytest.raises(PayloadTooLarge):
        sanitize_metadata({"blob": "x" * (L.MAX_METADATA_BYTES + 1)}, field="request_metadata")


def test_too_many_keys_is_refused():
    """Breadth as well as size: four thousand one-byte keys would otherwise
    slip under a bytes-only bound."""
    with pytest.raises(PayloadTooLarge):
        sanitize_metadata(
            {f"k{i}": i for i in range(L.MAX_METADATA_KEYS + 1)}, field="request_metadata"
        )


def test_a_non_object_metadata_value_is_refused():
    with pytest.raises(PayloadTooLarge):
        sanitize_metadata(["not", "an", "object"], field="request_metadata")


def test_absent_metadata_becomes_an_empty_object():
    """The column is NOT NULL, so `None` has to mean `{}` rather than fail."""
    assert sanitize_metadata(None, field="request_metadata") == {}


def test_free_text_is_bounded_and_scrubbed():
    """A task description is exactly where somebody pastes a curl command."""
    assert sanitize_text(None, field="description", limit=10) == ""

    with pytest.raises(PayloadTooLarge):
        sanitize_text("x" * 11, field="description", limit=10)

    pasted = "curl -H 'Authorization: Bearer abcdefghijklmnop' https://x"
    assert sanitize_text(pasted, field="description", limit=200) == REDACTED


def test_the_redaction_marker_is_not_a_hash_or_a_prefix():
    """A hash of a secret is still a credential oracle, and a prefix is what
    makes a leaked key findable."""
    out = sanitize_metadata({"token": "abcdefghijklmnop"}, field="request_metadata")

    assert out["token"] == REDACTED
    assert "abcd" not in out["token"]


# ── Bounds ──────────────────────────────────────────────────────────────────


def test_page_sizes_are_bounded():
    """An unbounded list endpoint is how a shared table becomes an outage."""
    assert L.DEFAULT_PAGE_SIZE <= L.MAX_PAGE_SIZE
    assert L.MAX_PAGE_SIZE <= 200


def test_a_run_cannot_hold_unlimited_steps():
    assert L.MAX_STEPS_PER_RUN > 0
