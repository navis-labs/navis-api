"""Google Drive API client over GoogleTransport.

The change feed is the heart of Drive capture: getStartPageToken establishes a
cursor, changes.list(pageToken) returns everything since. We request rich file
fields so a single pass carries ownership/permission/parent state in the payload.
"""
from __future__ import annotations

from typing import Any

from navis.signals.google.transport import GoogleTransport

_BASE = "https://www.googleapis.com/drive/v3"

# Fields kept deliberately rich so consumers never need a second round-trip.
_FILE_FIELDS = (
    "id,name,mimeType,parents,trashed,createdTime,modifiedTime,"
    "owners(emailAddress,displayName),shared,webViewLink"
)
_CHANGE_FIELDS = f"newStartPageToken,nextPageToken,changes(changeType,removed,fileId,file({_FILE_FIELDS}))"


class DriveClient:
    def __init__(self, transport: GoogleTransport) -> None:
        self._t = transport

    async def get_start_page_token(self) -> dict[str, Any]:
        return await self._t.request("GET", f"{_BASE}/changes/startPageToken")

    async def list_changes(
        self, page_token: str, *, page_size: int = 100
    ) -> dict[str, Any]:
        return await self._t.request(
            "GET",
            f"{_BASE}/changes",
            params={
                "pageToken": page_token,
                "pageSize": page_size,
                "includeRemoved": True,
                "fields": _CHANGE_FIELDS,
            },
        )

    async def list_files(self, page_token: str | None = None, page_size: int = 100) -> dict[str, Any]:
        params: dict[str, Any] = {
            "pageSize": page_size,
            "fields": f"nextPageToken,files({_FILE_FIELDS})",
            "q": "trashed = false",
        }
        if page_token:
            params["pageToken"] = page_token
        return await self._t.request("GET", f"{_BASE}/files", params=params)

    async def watch(self, page_token: str, channel_id: str, address: str) -> dict[str, Any]:
        return await self._t.request(
            "POST",
            f"{_BASE}/changes/watch",
            params={"pageToken": page_token},
            json={"id": channel_id, "type": "web_hook", "address": address},
        )

    async def stop_channel(self, channel_id: str, resource_id: str) -> dict[str, Any]:
        return await self._t.request(
            "POST", f"{_BASE}/channels/stop", json={"id": channel_id, "resourceId": resource_id}
        )
