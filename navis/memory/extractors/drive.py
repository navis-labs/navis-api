"""Deterministic extractor over Drive SignalEvents.

A project is approximated by a Drive folder (the file's parent). We record the
observed file-in-folder fact — no status, no health. Project status/health are
State/Reasoning, not Memory.
"""
from __future__ import annotations

from navis.memory.envelope import EntityType
from navis.memory.extractors.base import Extractor, MemoryDraft, SignalView
from navis.memory.schemas import current_version
from navis.signals.base import SignalSource

_DRIVE = {SignalSource.DRIVE}


class DriveProjectExtractor(Extractor):
    name = "drive_project"
    version = "1.0.0"
    sources = _DRIVE

    def extract(self, signal: SignalView) -> list[MemoryDraft]:
        parents = signal.payload.get("parents") or []
        if not parents:
            return []
        ver = current_version(EntityType.PROJECT)
        drafts = []
        for parent in parents:
            drafts.append(
                MemoryDraft(
                    entity_type=EntityType.PROJECT,
                    entity_natural_key=str(parent),
                    dedup_key=f"{parent}:{signal.external_id}:{signal.external_version}",
                    payload={"folder_id": str(parent), "name": signal.payload.get("name")},
                    confidence=0.6,
                    schema_version=ver,
                )
            )
        return drafts
