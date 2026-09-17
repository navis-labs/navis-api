"""Graph delta enumeration — the one pagination contract all three connectors share.

Every Microsoft resource Navis captures (mail, calendar, drive) exposes the same
change-tracking mechanism: a `/delta` endpoint that returns pages of items, each
page carrying either an `@odata.nextLink` (more of THIS pass) or, on the final
page, an `@odata.deltaLink` (the cursor for the NEXT pass).

Written once here rather than three times in the connectors because the two ways
this goes wrong are subtle, identical across resources, and both silent:

  1. **Following a link by rebuilding it.** Both links are fully-qualified URLs
     carrying an opaque server-side token in the query string. They must be
     followed VERBATIM. Reconstructing one from a base path and parameters drops
     the token, and Graph answers by restarting the enumeration from scratch —
     so the sync appears to work, returns plausible data, and never advances.

  2. **Advancing the cursor on a partial drain.** The deltaLink only appears on
     the final page. A drain that stops early and persists what it has skips
     every item on the pages it did not read, permanently, because the next pass
     resumes after them. So a truncated drain raises instead of returning; the
     sync engine dead-letters it and the cursor stays where it was, which is
     recoverable. Losing a founder's mail silently is not.

`full_resync_required` is returned for HTTP 410 Gone, which is what Graph
answers when a delta token has aged out. That is a normal lifecycle event
rather than a failure — the data is intact, only our cursor into it is stale.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from navis.platform.config import settings
from navis.platform.monitoring import get_logger
from navis.platform.monitoring.metrics import metrics
from navis.signals.microsoft.transport import GraphAPIError

logger = get_logger("signals.microsoft.delta")


class DeltaSyncIncomplete(Exception):
    """A delta drain hit the page-safety cap before reaching @odata.deltaLink.

    Raised, never swallowed, so the sync fails LOUD and retryable instead of
    persisting a cursor that would skip the undrained pages. Mirrors
    `DriveSyncIncomplete` on the Google side, for the same reason.
    """


@dataclass(slots=True)
class DeltaPage:
    """One drained delta enumeration."""

    items: list[dict[str, Any]] = field(default_factory=list)
    # The @odata.deltaLink to persist as the next cursor. None only when the
    # drain was cut short, which never reaches the caller — see the docstring.
    delta_link: str | None = None
    # True when Graph reported the supplied delta token as expired (410 Gone).
    full_resync_required: bool = False
    pages: int = 0


async def drain_delta(
    transport,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    resource: str,
    item_cap: int | None = None,
) -> DeltaPage:
    """Follow a delta enumeration to completion and return every item plus the
    next cursor.

    `url` is either the initial delta path ("/me/mailFolders/inbox/messages/delta")
    or a stored `@odata.deltaLink` from a previous pass. `params` are applied to
    the FIRST request only — a nextLink/deltaLink already carries its own query
    string, and re-appending parameters to one is how a drain silently loses its
    place.

    `item_cap` bounds an onboarding backfill. Hitting it is legitimate and
    returns normally WITH the delta link, because a capped first pass still has
    a valid go-forward cursor: the remainder is captured incrementally. That is
    the opposite of hitting the page cap, which has no valid cursor and raises.
    """
    max_pages = max(1, settings.microsoft_max_sync_pages)
    items: list[dict[str, Any]] = []
    next_url: str | None = url
    # Parameters ride the first request only; every subsequent URL is opaque.
    next_params: dict[str, Any] | None = params
    delta_link: str | None = None
    pages = 0

    while next_url:
        try:
            body = await transport.request("GET", next_url, params=next_params)
        except GraphAPIError as exc:
            if exc.is_gone:
                # The stored delta token aged out. Not an error — ask the caller
                # to start a fresh full sync rather than dead-lettering a pass
                # that was never going to succeed with this cursor.
                metrics.incr("graph.delta.expired", labels={"resource": resource})
                logger.info("graph delta token expired resource=%s; full resync", resource)
                return DeltaPage(full_resync_required=True, pages=pages)
            raise

        pages += 1
        page_items = body.get("value") or []
        items.extend(page_items)

        delta_link = body.get("@odata.deltaLink") or delta_link
        next_url = body.get("@odata.nextLink")
        next_params = None  # opaque from here on

        if item_cap is not None and len(items) >= item_cap:
            # Bounded backfill. If Graph already handed us the deltaLink we keep
            # it; otherwise we still need one, so keep draining pages but stop
            # collecting items — the cursor is what makes the next pass correct.
            if delta_link:
                metrics.incr("graph.delta.capped", labels={"resource": resource})
                logger.warning(
                    "graph %s backfill capped at %d items; remainder captured incrementally",
                    resource,
                    len(items),
                )
                break

        if next_url and pages >= max_pages:
            metrics.incr("graph.delta.truncated", labels={"resource": resource})
            logger.error(
                "graph %s delta drain exceeded %d pages without a deltaLink",
                resource,
                max_pages,
            )
            raise DeltaSyncIncomplete(
                f"graph {resource} delta drain exceeded {max_pages} pages "
                "without reaching @odata.deltaLink"
            )

    if delta_link is None:
        # Reached the end of the enumeration with no cursor to persist. Graph
        # always terminates a delta run with a deltaLink, so this means the
        # response shape was not what we think it is — refuse rather than
        # persisting an empty cursor, which would silently restart every pass.
        metrics.incr("graph.delta.no_cursor", labels={"resource": resource})
        raise DeltaSyncIncomplete(
            f"graph {resource} delta drain finished without @odata.deltaLink"
        )

    metrics.incr("graph.delta.pages", labels={"resource": resource}, value=pages)
    return DeltaPage(items=items, delta_link=delta_link, pages=pages)
