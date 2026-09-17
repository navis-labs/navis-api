"""Deterministic extractor over Drive SignalEvents.

INERT BY DESIGN — this extractor currently emits nothing. See below.

It used to mint one `EntityType.PROJECT` per parent folder of every Drive file,
which asserted an organizational claim the evidence does not support: a Drive
folder is a storage container, not a project. "Receipts", "Untitled folder",
"Screenshots" and a founder's personal folders all became Projects, and because
`DriveProjectExtractor` was the ONLY producer of PROJECT, everything State
reported through `OrganizationalState.projects()` was a folder name.

Why not just lower the confidence instead: `MEDIUM` already documents this exact
case ("a Drive folder treated as a project"), but confidence is a formation-time
annotation that nothing downstream gates on — `state/v2/builders.py` folds every
record regardless — so a low number never stopped the folder from surfacing as a
project. Qualifying the claim required a filter that does not exist; the honest
alternative is to stop making the claim.

Nothing is lost from the record:
  * the Drive `SignalEvent` still captures name / parents / owners / shared /
    mime_type / drive_id in full — the raw observation is untouched;
  * folder lineage for Google Docs survives on the DOCUMENT entity, which
    carries `parents` (see `extractors/docs.py`);
  * `EntityType.PROJECT` itself is untouched — the type, its payload schema and
    State's fold all remain, so a genuine project signal (a real PM tool, or an
    explicit user designation) can populate it without any of this being rebuilt.

Restoring folder→project inference means first answering what evidence makes a
folder a project. Do not simply revert this file.
"""
from __future__ import annotations

from navis.memory.extractors.base import Extractor, MemoryDraft, SignalView
from navis.signals.base import SignalSource

_DRIVE = {SignalSource.DRIVE}


class DriveProjectExtractor(Extractor):
    name = "drive_project"
    # 2.0.0: stopped emitting PROJECT drafts (see module docstring). Extractor
    # version is part of `memory_id`, so previously-written records keep their
    # identity and replay stays deterministic.
    version = "2.0.0"
    sources = _DRIVE

    def extract(self, signal: SignalView) -> list[MemoryDraft]:
        """Emit nothing: a Drive folder is not evidence of an organizational project."""
        return []
