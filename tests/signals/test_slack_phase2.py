"""Slack Phase 2: client rate-limit retryability, failure-monitor threshold, channel_id column."""
from __future__ import annotations

from navis.signals.slack.client import SlackAPIError
from navis.signals.slack.oauth import install_fields
from navis.signals.reliability.slack_health import is_unhealthy


def test_slack_rate_limit_is_retryable_other_errors_not():
    assert SlackAPIError("ratelimited", status=429).retryable is True
    assert SlackAPIError("server_error", status=503).retryable is True
    assert SlackAPIError("not_in_channel").retryable is False     # terminal -> skip, don't retry
    assert SlackAPIError("invalid_auth").retryable is False


def test_retry_after_is_honored_by_the_framework():
    # P0 proof (already-solved): the retry wait reads our exception's retry_after and returns
    # it (capped at max_backoff), so a 429 sleeps exactly that long before retrying.
    from types import SimpleNamespace

    from navis.signals.reliability.retry import RetryPolicy, _retry_after_aware_wait

    wait = _retry_after_aware_wait(RetryPolicy(max_backoff=30.0))
    exc = SlackAPIError("ratelimited", status=429, retry_after=7.0)
    state = SimpleNamespace(outcome=SimpleNamespace(exception=lambda: exc))
    assert wait(state) == 7.0


def _client(handler, **kw):
    import httpx

    from navis.signals.slack.client import SlackClient

    return SlackClient("tok", client=httpx.AsyncClient(transport=httpx.MockTransport(handler)), **kw)


async def test_invalid_auth_and_token_revoked_become_reconsent_required():
    import httpx
    import pytest

    from navis.signals.slack.client import SlackReconsentRequired

    for err in ("invalid_auth", "token_revoked", "account_inactive"):
        client = _client(lambda req, e=err: httpx.Response(200, json={"ok": False, "error": e}))
        with pytest.raises(SlackReconsentRequired) as exc:
            await client.users_list()
        assert exc.value.error == err and exc.value.retryable is False  # terminal, never retried
        await client.aclose()


async def test_malformed_json_becomes_typed_error_with_status():
    import httpx
    import pytest

    from navis.signals.reliability.retry import RetryPolicy

    # max_attempts=1 so the (retryable) invalid_json doesn't loop/sleep in the test.
    client = _client(lambda req: httpx.Response(200, text="<html>cloudflare</html>"),
                     policy=RetryPolicy(max_attempts=1))
    with pytest.raises(SlackAPIError) as exc:
        await client.users_list()
    assert exc.value.error == "invalid_json" and exc.value.status == 200
    await client.aclose()


async def test_metrics_emitted_on_success_and_rate_limit(monkeypatch):
    import httpx

    from navis.platform.monitoring.metrics import metrics

    seen: list[str] = []
    monkeypatch.setattr(metrics, "incr", lambda name, **k: seen.append(name))

    ok = _client(lambda req: httpx.Response(200, json={"ok": True, "members": []}))
    await ok.users_list()
    await ok.aclose()
    assert "slack.api.success" in seen and "slack.api.request" in seen

    seen.clear()
    limited = _client(lambda req: httpx.Response(429, headers={"Retry-After": "1"}, text="rate"),
                      policy=__import__("navis.signals.reliability.retry", fromlist=["RetryPolicy"]).RetryPolicy(max_attempts=1))
    try:
        await limited.users_list()
    except SlackAPIError:
        pass
    await limited.aclose()
    assert "slack.api.rate_limited" in seen


def test_failure_monitor_threshold():
    # below the min sample -> never alert (Slack also exempts tiny apps)
    assert is_unhealthy(1.0, 10) is False
    # high failure rate with enough volume -> alert (before Slack's 0.95 auto-disable)
    assert is_unhealthy(0.85, 100) is True
    assert is_unhealthy(0.50, 100) is False


def test_oauth_install_captures_both_bot_identifiers():
    # CR-1: persist bot_user_id (U…) AND bot_id (B…) so the echo-loop guard matches either.
    access = {
        "ok": True, "access_token": "xoxb-secret",
        "team": {"id": "T1"}, "bot_user_id": "U_BOT", "app_id": "A1", "scope": "channels:history",
    }
    fields = install_fields(access, bot_id="B_BOT")
    assert fields["team_id"] == "T1" and fields["bot_token"] == "xoxb-secret"
    assert fields["slack_state"]["bot_id"] == "B_BOT"
    assert fields["slack_state"]["bot_user_id"] == "U_BOT"


# ---- oauth.py hardening ----
import pytest  # noqa: E402

from navis.signals.slack.oauth import (  # noqa: E402
    SLACK_SCOPES,
    SlackOAuthError,
    exchange_code,
    fetch_bot_identity,
    install_fields,
)

_ALL_SCOPES = ",".join(SLACK_SCOPES)


class _Resp:
    def __init__(self, status=200, payload=None, raise_json=False):
        self.status_code = status
        self._p = payload or {}
        self._raise = raise_json

    def json(self):
        if self._raise:
            raise ValueError("not json")
        return self._p


def _patch_httpx(monkeypatch, resp=None, raise_http=False):
    import httpx

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, *a, **k):
            if raise_http:
                raise httpx.HTTPError("boom")
            return resp

    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: _Client())


def test_install_rejects_missing_team_and_token():
    with pytest.raises(SlackOAuthError):
        install_fields({"ok": True, "access_token": "x", "team": {}}, bot_id="B")  # no team.id
    with pytest.raises(SlackOAuthError):
        install_fields({"ok": True, "team": {"id": "T1"}}, bot_id="B")             # no access_token


async def test_exchange_rejects_missing_scope(monkeypatch):
    _patch_httpx(monkeypatch, _Resp(200, {"ok": True, "scope": "channels:history"}))  # partial
    with pytest.raises(SlackOAuthError, match="missing required scopes"):
        await exchange_code("code")


async def test_exchange_invalid_json(monkeypatch):
    _patch_httpx(monkeypatch, _Resp(200, raise_json=True))
    with pytest.raises(SlackOAuthError, match="invalid oauth response"):
        await exchange_code("code")


async def test_exchange_http_500(monkeypatch):
    _patch_httpx(monkeypatch, _Resp(500))
    with pytest.raises(SlackOAuthError, match="oauth server error"):
        await exchange_code("code")


async def test_exchange_network_failure(monkeypatch):
    _patch_httpx(monkeypatch, raise_http=True)
    with pytest.raises(SlackOAuthError, match="oauth request failed"):
        await exchange_code("code")


async def test_exchange_metrics_on_success_and_failure(monkeypatch):
    from navis.platform.monitoring.metrics import metrics

    seen: list[str] = []
    monkeypatch.setattr(metrics, "incr", lambda name, **k: seen.append(name))

    _patch_httpx(monkeypatch, _Resp(200, {"ok": True, "scope": _ALL_SCOPES, "access_token": "x"}))
    await exchange_code("code")
    assert "slack.oauth.exchange.success" in seen

    seen.clear()
    _patch_httpx(monkeypatch, _Resp(500))
    with pytest.raises(SlackOAuthError):
        await exchange_code("code")
    assert "slack.oauth.exchange.failure" in seen


async def test_fetch_bot_identity_parses_all_ids(monkeypatch):
    _patch_httpx(monkeypatch, _Resp(200, {"ok": True, "bot_id": "B1", "team_id": "T1", "user_id": "U1"}))
    ident = await fetch_bot_identity("xoxb-tok")
    assert ident == {"bot_id": "B1", "team_id": "T1", "user_id": "U1"}


def test_channel_id_is_denormalized_into_the_row():
    # CR-3: the repository copies metadata.channel_id into the indexed column.
    from navis.signals.repository import SignalRepository
    from navis.signals.slack.events import event_from_slack
    from tests.signals.fakes import make_ctx

    ev = event_from_slack(make_ctx(), "T1", {
        "type": "message", "channel": "C42", "user": "U1", "ts": "1700000000.001", "text": "x",
    })
    assert ev is not None
    row = SignalRepository._row(ev)
    assert row["channel_id"] == "C42"
