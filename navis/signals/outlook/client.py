"""OutlookClient — thin Graph wrapper for mail capture.

Holds the URLs, the `$select` lists and nothing else: no retries (the transport
owns those), no parsing (the parser owns that), no persistence (the sync engine
owns that). Kept separate from the connector so the connector can be tested
against a fake client, which is how every Google connector is tested.

The `$select` lists are not incidental. Graph returns a large default projection
per message; asking for exactly the fields the parser reads cuts payload size by
roughly an order of magnitude on a backfill, and — more importantly — makes the
set of fields Navis pulls out of somebody's mailbox reviewable in one place.
"""
from __future__ import annotations

from typing import Any

# Everything `outlook.parser.parse_message` reads, and nothing else.
#
# `internetMessageHeaders` is requested explicitly because Graph omits it from
# the default projection; without it the reply graph loses In-Reply-To and
# References, so threads reconstruct less accurately.
MESSAGE_FIELDS = ",".join((
    "id",
    "changeKey",
    "conversationId",
    "internetMessageId",
    "internetMessageHeaders",
    "subject",
    "from",
    "sender",
    "toRecipients",
    "ccRecipients",
    "bccRecipients",
    "receivedDateTime",
    "sentDateTime",
    "bodyPreview",
    "body",
    "hasAttachments",
    "categories",
    "isRead",
    "importance",
    "webLink",
))


class OutlookClient:
    def __init__(self, transport) -> None:
        self._transport = transport

    @property
    def transport(self):
        """The underlying GraphTransport.

        Exposed because delta pagination is driven by `microsoft.delta.drain_delta`,
        which must follow Graph's opaque nextLink/deltaLink URLs verbatim and so
        cannot go through a method that builds a path. Read-only on purpose: the
        client still owns every URL and projection this connector uses.
        """
        return self._transport

    async def messages_delta(
        self,
        *,
        url: str | None = None,
        since_iso: str | None = None,
    ) -> dict[str, Any]:
        """One page of the inbox delta enumeration.

        `url` follows a stored deltaLink/nextLink verbatim. When absent this
        starts a fresh enumeration, optionally filtered to messages received
        since `since_iso` — which is what bounds an onboarding backfill to a
        recent window instead of pulling a decade of mail into the signal log.

        Scoped to the INBOX rather than the whole mailbox. Sent mail, archive
        and every user-created folder are deliberately out of scope: the
        product reasons about what is arriving and going unanswered, and
        enumerating a full mailbox is both far slower and a much larger claim
        on somebody's data than the feature needs.
        """
        if url:
            return await self._transport.request("GET", url)
        params: dict[str, Any] = {"$select": MESSAGE_FIELDS}
        if since_iso:
            params["$filter"] = f"receivedDateTime ge {since_iso}"
        return await self._transport.request(
            "GET", "/me/mailFolders/inbox/messages/delta", params=params
        )

    async def get_profile(self) -> dict[str, Any]:
        """The signed-in user, used to identify the connected account."""
        return await self._transport.request(
            "GET", "/me", params={"$select": "id,mail,userPrincipalName,displayName"}
        )
