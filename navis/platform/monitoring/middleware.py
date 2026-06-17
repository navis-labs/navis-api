"""Request-context ASGI middleware.

Assigns/propagates a correlation id per request: reads the inbound
`X-Request-ID` header if present (so a gateway/client id is preserved), else
generates one. Binds it to the logging contextvar and echoes it on the response.
"""
from __future__ import annotations

import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from navis.platform.config import settings
from navis.platform.monitoring.logging import set_request_id


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        header = settings.request_id_header
        request_id = request.headers.get(header) or uuid.uuid4().hex
        set_request_id(request_id)
        response = await call_next(request)
        response.headers[header] = request_id
        return response
