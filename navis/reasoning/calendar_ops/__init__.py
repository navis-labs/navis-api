"""Mutations of EXISTING calendar events — reschedule and cancel.

Sibling of `reasoning.scheduling` (which CREATES events) and
`reasoning.calendar_query` (which READS them). Kept separate because the
targeting problem ("which meeting do you mean?") is specific to mutation and has
no analogue in the other two.
"""
from navis.reasoning.calendar_ops.agent import CalendarOpsAgent
from navis.reasoning.calendar_ops.contract import (
    CalendarOpsClarification,
    CalendarOpsContext,
    CalendarOpsResult,
    MeetingMutationProposal,
    TargetEvent,
)

__all__ = [
    "CalendarOpsAgent",
    "CalendarOpsClarification",
    "CalendarOpsContext",
    "CalendarOpsResult",
    "MeetingMutationProposal",
    "TargetEvent",
]
