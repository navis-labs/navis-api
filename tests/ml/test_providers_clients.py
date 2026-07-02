"""Regression: Azure + Claude SDK clients must disable their built-in retries (max_retries=0) so
the ResilientProvider (registry) is the SOLE retry owner. The SDK default of 2 would double-retry
— up to 9 provider calls per logical call during an incident (retry amplification). The vendor
SDKs are faked in sys.modules, so this runs without the real packages installed (lazy imports).
"""
from __future__ import annotations

import sys
import types


async def test_azure_client_disables_sdk_retries(monkeypatch):
    captured: dict = {}

    class _FakeAzure:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def close(self):
            return None

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(AsyncAzureOpenAI=_FakeAzure))
    from navis.ml.providers.azure import AzureProvider

    AzureProvider()._client()  # inside the test's running loop
    assert captured.get("max_retries") == 0


async def test_claude_client_disables_sdk_retries(monkeypatch):
    captured: dict = {}

    class _FakeAnthropic:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def close(self):
            return None

    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(AsyncAnthropic=_FakeAnthropic))
    from navis.ml.providers.claude import ClaudeProvider

    ClaudeProvider()._client()
    assert captured.get("max_retries") == 0
