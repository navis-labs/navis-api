"""Datetime handling at the Graph boundary.

Two conversions, both of which have a wrong-looking obvious implementation:

`graph_iso` — Graph's OData `$filter` grammar takes an ISO-8601 datetime as a
BARE literal (`receivedDateTime ge 2026-08-07T00:00:00Z`). It rejects the
`+00:00` offset spelling that `datetime.isoformat()` produces, and it rejects
the quoted form that most other query languages require. A filter Graph cannot
parse comes back as HTTP 400, which the transport treats as non-retryable, so
the whole backfill fails — loudly, at least, but for a reason that reads like a
permissions problem rather than a formatting one.

`parse_graph_datetime` — Graph emits the trailing-`Z` spelling, which
`datetime.fromisoformat` only learned to accept in Python 3.11. Normalising here
rather than relying on the runtime version keeps capture correct if this ever
runs somewhere older, and returning None instead of raising keeps one
unparseable timestamp on one item from failing an entire sync pass.

Shared across the three Graph connectors because all three hit both cases, and
a per-connector copy is how two of them end up with a subtly different one.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def graph_iso(value: datetime) -> str:
    """Render a datetime as the bare UTC literal an OData $filter expects."""
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_graph_datetime(value: Any) -> datetime | None:
    """Parse a Graph timestamp, or return None if it is absent/unparseable."""
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def parse_graph_datetime_field(field: Any) -> datetime | None:
    """Parse Graph's `{dateTime, timeZone}` composite, used by calendar events.

    Graph returns event times in this shape rather than as a plain ISO string,
    and the `dateTime` half carries NO offset — the zone lives in the sibling
    field. Reading `dateTime` alone therefore yields a naive datetime that looks
    valid and is silently wrong by up to a day, which for a calendar connector
    means meetings landing on the wrong date in a founder's brief.

    Events are requested with `Prefer: outlook.timezone="UTC"` so the zone is
    UTC in practice; this attaches that explicitly rather than trusting a naive
    value to be interpreted correctly downstream.
    """
    if not isinstance(field, dict):
        return None
    parsed = parse_graph_datetime(field.get("dateTime"))
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        # timeZone is a Windows or IANA zone name. We request UTC, so anything
        # else means the Prefer header did not take effect — assume UTC rather
        # than dropping the event, and let the value be visibly one zone off
        # instead of missing entirely.
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed
