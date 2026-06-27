"""Deterministic extractor over Drive SignalEvents.

A project is approximated by a Drive folder (the file's parent). We record the
observed file-in-folder fact — no status, no health. Project status/health are
State/Reasoning, not Memory.
"""
from __future__ import annotations

from navis.memory.envelope import EntityType
from navis.memory.extractors.base import Extractor, MemoryDraft, SignalView
from navis.memory.schemas import current_version
from navis.memory.confidence import MEDIUM
from navis.signals.base import SignalSource

_DRIVE = {SignalSource.DRIVE}


class DriveProjectExtractor(Extractor):
    name = "drive_project"
    version = "1.0.0"
    sources = _DRIVE

    def extract(self, signal: SignalView) -> list[MemoryDraft]:
        p = signal.payload
        parents = p.get("parents") or []
        if not parents:
            return []
        owners = [o.get("email") for o in (p.get("owners") or []) if o.get("email")]
        ver = current_version(EntityType.PROJECT)
        drafts = []
        for parent in parents:
            drafts.append(
                MemoryDraft(
                    entity_type=EntityType.PROJECT,
                    entity_natural_key=str(parent),
                    dedup_key=f"{parent}:{signal.external_id}:{signal.external_version}",
                    payload={
                        "folder_id": str(parent),
                        "name": p.get("name"),
                        # KEEP: ownership + collaboration + type (future person↔file edges).
                        "owners": owners,
                        "shared": p.get("shared"),
                        "mime_type": p.get("mime_type"),
                        "drive_id": p.get("drive_id"),
                    },
                    confidence=MEDIUM,
                    schema_version=ver,
                )
            )
        return drafts
