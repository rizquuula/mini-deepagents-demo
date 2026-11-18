"""A pure ASGI middleware that gives each HTTP request an ID and logs its lifetime.

It is pure ASGI, not ``BaseHTTPMiddleware``, so the ``request completed`` line
of a streamed response comes after the last byte of the stream.
"""

from __future__ import annotations

import logging
import time
from typing import Final
from uuid import uuid4

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from mini_deepagents_demo.observability.logging import request_id_var

__all__ = ["RequestContextMiddleware"]

logger = logging.getLogger(__name__)

REQUEST_ID_HEADER: Final = "x-request-id"
_REQUEST_ID_LENGTH: Final = 12
_INTERNAL_ERROR: Final = "internal server error"


class RequestContextMiddleware:
    """Set the request ID, add the ``x-request-id`` header, and log the request."""

    def __init__(self, app: ASGIApp) -> None:
        self._app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        request_id = uuid4().hex[:_REQUEST_ID_LENGTH]
        token = request_id_var.set(request_id)
        fields = {"method": scope["method"], "path": scope["path"]}
        started = time.perf_counter()
        status: int | None = None

        async def send_with_request_id(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                headers = list(message.get("headers", []))
                headers.append((REQUEST_ID_HEADER.encode("latin-1"), request_id.encode("latin-1")))
                message = {**message, "headers": headers}
            await send(message)

        try:
            logger.info("request received", extra=fields)
            try:
                await self._app(scope, receive, send_with_request_id)
            except Exception:
                logger.exception("unhandled error", extra=fields)
                if status is None:
                    response = JSONResponse({"detail": _INTERNAL_ERROR}, status_code=500)
                    await response(scope, receive, send_with_request_id)
        finally:
            logger.info(
                "request completed",
                extra={**fields, "status": status, "duration_ms": _elapsed_ms(started)},
            )
            request_id_var.reset(token)


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)
