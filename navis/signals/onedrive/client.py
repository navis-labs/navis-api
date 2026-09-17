"""OneDriveClient — thin Graph wrapper for file-activity capture."""
from __future__ import annotations

from typing import Any

# Metadata only. There is deliberately no field here that returns file CONTENT
# or a working download URL: the product reasons about document activity
# ("this deck changed, these people can see it"), never about what a document
# says, and a `@microsoft.graph.downloadUrl` in the signal log would be a
# pre-authenticated link to somebody's file sitting in an append-only table.
#
# The Google Drive connector drops `webViewLink` for exactly this reason.
ITEM_FIELDS = ",".join((
    "id",
    "eTag",
    "cTag",
    "name",
    "file",
    "folder",
    "size",
    "createdDateTime",
    "lastModifiedDateTime",
    "createdBy",
    "lastModifiedBy",
    "parentReference",
    "shared",
    "deleted",
))


class OneDriveClient:
    def __init__(self, transport) -> None:
        self._transport = transport

    @property
    def transport(self):
        """The underlying GraphTransport — see OutlookClient.transport."""
        return self._transport

    async def root_delta(self, *, url: str | None = None) -> dict[str, Any]:
        """One page of the drive-root delta enumeration.

        Rooted at `/me/drive/root` — the user's own OneDrive. Files shared WITH
        the user live in other drives and are reachable via `/me/drive/sharedWithMe`,
        which has no delta endpoint and therefore cannot participate in this
        change-tracking model. Capturing them would mean polling a full listing
        on every pass and diffing it ourselves, which is a different mechanism
        with different failure modes; it is a deliberate follow-up rather than
        something to bolt onto this drain.
        """
        if url:
            return await self._transport.request("GET", url)
        return await self._transport.request(
            "GET", "/me/drive/root/delta", params={"$select": ITEM_FIELDS}
        )
