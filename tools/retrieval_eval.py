"""Retrieval validation harness — founder questions against real Memory.

Run BEFORE and AFTER an embedding migration; the numbers are directly comparable
because the golden set is derived from the corpus itself rather than hand-written
ids that rot when memory changes.

    python tools/retrieval_eval.py <org_id> [--llm] [--runs N]

Measures, per query:
  * retrieved hits (pre-rerank) with cosine scores
  * survivors after rerank
  * what ContextBuilder actually admits
  * optional: the real LLM answer over that context
And overall: recall@k, retrieval latency, duplicate rate.

Read-only. Writes nothing, embeds nothing.
"""
from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
# Runnable as `python tools/retrieval_eval.py` from the repo root without install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from sqlalchemy import select  # noqa: E402

import navis.registry  # noqa: F401,E402  (model + subscriber registration)
from navis.memory.models import MemoryRecord  # noqa: E402
from navis.memory.importance import importance_by_memory_id  # noqa: E402
from navis.memory.lineage import lineage_by_memory_id  # noqa: E402
from navis.ml.embeddings import admits, read_pin  # noqa: E402
from navis.ml.rag.context_builder import ContextBuilder  # noqa: E402
from navis.ml.rag.time_window import parse_time_window  # noqa: E402
from navis.ml.retrieval.retriever import MemoryRetriever  # noqa: E402
from navis.platform.database import session_scope  # noqa: E402

K = 8

QUESTIONS = [
    "What should I focus on today?",
    "What am I waiting on?",
    "What did we decide?",
    "What changed since yesterday?",
    "Prepare me for my next meeting.",
]


async def golden_set(db, org_id: uuid.UUID, now: datetime) -> dict[str, set[str]]:
    """Derive relevance labels from the corpus so the set survives re-extraction.

    Deliberately conservative: only facts a founder would unambiguously expect
    back for that question count as relevant.
    """
    records = (
        await db.scalars(select(MemoryRecord).where(MemoryRecord.organization_id == org_id))
    ).all()
    live = [r for r in records if admits(r)]  # facts the current rules still stand behind

    commitments = {str(r.memory_id) for r in live if r.entity_type == "commitment"}
    decisions = {str(r.memory_id) for r in live if r.entity_type == "decision"}
    meetings = {str(r.memory_id) for r in live if r.entity_type == "meeting"}
    yesterday = now - timedelta(days=2)
    recent = {
        str(r.memory_id)
        for r in live
        if r.event_timestamp and r.event_timestamp >= yesterday
        and r.entity_type in ("commitment", "decision", "meeting", "thread")
    }
    return {
        "What should I focus on today?": commitments | decisions,
        "What am I waiting on?": commitments,
        "What did we decide?": decisions,
        "What changed since yesterday?": recent,
        "Prepare me for my next meeting.": meetings,
    }


def recall_at_k(retrieved: list[str], relevant: set[str], k: int = K) -> float | None:
    if not relevant:
        return None  # nothing to find — reporting 0 or 1 would both mislead
    return round(len(set(retrieved[:k]) & relevant) / len(relevant), 3)


def duplicate_rate(texts: list[str]) -> float:
    """Share of returned slots wasted on a fact already present in the answer."""
    if not texts:
        return 0.0
    return round(1 - len(set(texts)) / len(texts), 3)


async def run(
    org_id: uuid.UUID,
    use_llm: bool,
    runs: int,
    model: str | None = None,
    version: int | None = None,
) -> None:
    # --model/--version evaluate a candidate index WITHOUT moving the production
    # read pin: this process reads the override, every other process is unaffected.
    # That is how a migration is validated before cutover rather than after.
    if model:
        from navis.platform.config import settings

        settings.embedding_read_model = model
        if version is not None:
            settings.embedding_read_version = version
    pin = read_pin()
    print(f"read pin: model={pin.model} version={pin.version} provider={pin.provider.info.name}")
    now = datetime.now(timezone.utc)
    recalls: list[float] = []
    dups: list[float] = []
    latencies: list[float] = []
    importances: list[float] = []
    event_ratios: list[float] = []

    async with session_scope() as db:
        golden = await golden_set(db, org_id, now)
        for q in QUESTIONS:
            window = parse_time_window(q, now=now)
            since, until = window if window else (None, None)

            t0 = time.perf_counter()
            hits = await MemoryRetriever(db).search(org_id, q, k=K, since=since, until=until)
            elapsed = (time.perf_counter() - t0) * 1000
            for _ in range(max(0, runs - 1)):
                t = time.perf_counter()
                await MemoryRetriever(db).search(org_id, q, k=K, since=since, until=until)
                latencies.append((time.perf_counter() - t) * 1000)
            latencies.append(elapsed)

            ctx = await ContextBuilder(db).build(
                org_id, q, k=K, token_budget=1500, since=since, until=until, now=now
            )
            texts = [f.text for f in ctx.facts]
            r = recall_at_k([f.source for f in ctx.facts], golden.get(q, set()))
            d = duplicate_rate(texts)
            if r is not None:
                recalls.append(r)
            dups.append(d)

            print(f"\n{'='*94}\nQ: {q}")
            print(f"  window={'yes' if window else 'none'}  latency={elapsed:.0f}ms  "
                  f"relevant_in_corpus={len(golden.get(q, set()))}")
            print(f"  reranked hits ({len(hits)}):")
            for h in hits[:5]:
                print(f"      cos={h.score:+.4f} [{h.entity_type}] {h.source_text[:66]!r}")
            # Mean importance of what was actually returned — the metric Phase 2
            # moves. Absent = unscored, reported separately rather than as zero.
            imp_map = await importance_by_memory_id(
                db, [uuid.UUID(f.source) for f in ctx.facts]
            )
            imps = [imp_map[uuid.UUID(f.source)] for f in ctx.facts
                    if uuid.UUID(f.source) in imp_map]
            mean_imp = round(sum(imps) / len(imps), 3) if imps else None
            if mean_imp is not None:
                importances.append(mean_imp)
            # Distinct EVENTS per answer — the metric dedup targets. Text-level
            # duplicate_rate misses it: "Person Mulayam" and "Received email from
            # mulayam@…" are different strings describing the same meeting.
            lin = await lineage_by_memory_id(db, [uuid.UUID(f.source) for f in ctx.facts])
            events = {lin[uuid.UUID(f.source)].source_signal_event_id
                      for f in ctx.facts if uuid.UUID(f.source) in lin}
            ratio = round(len(events) / len(ctx.facts), 3) if ctx.facts else 1.0
            event_ratios.append(ratio)
            high = sum(1 for i in imps if i >= 0.6)
            print(f"  ContextBuilder: facts={len(ctx.facts)} tokens={ctx.token_estimate} "
                  f"recall@{K}={r} duplicate_rate={d} mean_importance={mean_imp} "
                  f"high_importance={high}/{len(imps)} "
                  f"events={len(events)}/{len(ctx.facts)} (ratio={ratio})")
            for f in ctx.facts[:8]:
                print(f"        imp={imp_map.get(uuid.UUID(f.source), 'n/a')!s:>5} {f.text[:66]!r}")
            if not ctx.facts:
                print("      >>> ZERO FACTS -> hard-coded 'not enough memory'; LLM not called")
            if use_llm and ctx.facts:
                from navis.ml.generation.narratives import explanation, recommendation
                gen = recommendation if "focus on" in q.lower() else explanation
                t = time.perf_counter()
                narrative = await gen(org_id, ctx)
                print(f"  LLM ({narrative.model}, {(time.perf_counter()-t)*1000:.0f}ms):")
                print(f"      {narrative.text[:400]}")

    print(f"\n{'='*94}\nSUMMARY")
    scored = [r for r in recalls]
    print(f"  mean recall@{K}     : {round(sum(scored)/len(scored), 3) if scored else 'n/a'}")
    print(f"  mean duplicate rate : {round(sum(dups)/len(dups), 3)}")
    print(f"  mean events/facts   : "
          f"{round(sum(event_ratios)/len(event_ratios), 3) if event_ratios else 'n/a'}"
          f"   (1.0 = one event per slot)")
    print(f"  mean importance     : "
          f"{round(sum(importances)/len(importances), 3) if importances else 'n/a'}")
    print(f"  latency p50 / max   : {statistics.median(latencies):.0f}ms / {max(latencies):.0f}ms")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("org_id")
    ap.add_argument("--llm", action="store_true", help="also call the real LLM")
    ap.add_argument("--runs", type=int, default=3, help="latency samples per query")
    ap.add_argument("--model", help="evaluate this index instead of the configured read pin")
    ap.add_argument("--version", type=int, help="index version to evaluate")
    args = ap.parse_args()
    asyncio.run(run(uuid.UUID(args.org_id), args.llm, args.runs, args.model, args.version))


if __name__ == "__main__":
    main()
