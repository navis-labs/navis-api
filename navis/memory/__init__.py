"""Memory — Layer 2. The append-only, event-sourced Organizational History.

Memory = Historical Truth. It consumes ONLY SignalEvents, writes immutable,
deterministically-keyed, fully-provenanced `memory_records`, and computes NO
intelligence (no status/health/risk/priority/score). State is built FROM Memory;
Memory never reads State.

    signal_events -> extractors (pure) -> MemoryRecord (append-only) -> replayable
"""
from navis.memory.envelope import EntityType, MemoryRecord
from navis.memory.ids import entity_id, memory_record_id
from navis.memory import governance as _governance  # noqa: F401  registers the erasure cascade

__all__ = ["EntityType", "MemoryRecord", "entity_id", "memory_record_id"]
