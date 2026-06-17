"""Meet signals — derived from Calendar (phase 1). No attendance tracking yet."""
from navis.signals.meet.deriver import MeetDeriver, derive_meeting_events

__all__ = ["MeetDeriver", "derive_meeting_events"]
