"""Regression test for navis/ml/providers/gemini.py.

GeminiProvider.embed calls the SYNCHRONOUS google-generativeai SDK; it must run OFF the event
loop (asyncio.to_thread) so a Gemini embed never blocks other concurrent coroutines. The SDK is
faked in sys.modules, so this runs without the real package installed (matching the lazy import).
"""
from __future__ import annotations

import sys
import threading
import types


async def test_gemini_embed_runs_off_the_event_loop(monkeypatch):
    loop_thread = threading.current_thread()
    seen: list[threading.Thread] = []

    def embed_content(model, content):
        seen.append(threading.current_thread())
        return {"embedding": [0.1, 0.2, 0.3]}

    fake_genai = types.SimpleNamespace(configure=lambda **k: None, embed_content=embed_content)
    monkeypatch.setitem(sys.modules, "google.generativeai", fake_genai)

    from navis.ml.providers.gemini import GeminiProvider

    result = await GeminiProvider().embed(["a", "b"])

    # correct normalization to the vendor-agnostic DTO
    assert len(result.vectors) == 2 and result.dim == 3
    assert result.model == "text-embedding-004"
    # ran on a worker thread, NOT the event-loop thread (no loop blocking)
    assert seen and all(t is not loop_thread for t in seen)
