"""The v1 HTTP API of the chat: routes, request and response models, error mapping."""

from __future__ import annotations

import logging
from typing import Annotated, Final

from fastapi import APIRouter, FastAPI, Request, status
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from mini_deepagents import (
    ChatService,
    ContextOverflowError,
    Message,
    Session,
    SessionNotFoundError,
)

from mini_deepagents_demo.adapters.http.sse import chat_event_stream
from mini_deepagents_demo.observability.logging import request_id_var

__all__ = ["create_api_router", "register_error_handlers"]

logger = logging.getLogger(__name__)

API_PREFIX: Final = "/api/v1"
_SESSION_NOT_FOUND: Final = "session not found"
_TOO_LONG: Final = "message is too long for the context window"
_SSE_HEADERS: Final = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


class SendMessageRequest(BaseModel):
    content: Annotated[str, Field(min_length=1)]


class SessionResponse(BaseModel):
    id: int
    created_at: str

    @classmethod
    def of(cls, session: Session) -> SessionResponse:
        return cls(id=session.id, created_at=session.created_at)


class MessageResponse(BaseModel):
    role: str
    content: str

    @classmethod
    def of(cls, message: Message) -> MessageResponse:
        return cls(role=message.role, content=message.content)


def create_api_router(service: ChatService) -> APIRouter:
    """Build the v1 routes on top of ``service``."""
    router = APIRouter(prefix=API_PREFIX)

    @router.get("/sessions")
    async def list_sessions() -> list[SessionResponse]:
        return [SessionResponse.of(s) for s in await service.list_sessions()]

    @router.post("/sessions", status_code=status.HTTP_201_CREATED)
    async def create_session() -> SessionResponse:
        session = await service.create_session()
        logger.info("session created", extra={"session_id": session.id})
        return SessionResponse.of(session)

    @router.get("/sessions/{session_id}/messages")
    async def get_messages(session_id: int) -> list[MessageResponse]:
        return [MessageResponse.of(m) for m in await service.get_messages(session_id)]

    @router.post("/sessions/{session_id}/messages")
    async def send_message(session_id: int, body: SendMessageRequest) -> StreamingResponse:
        turn = await service.prepare(session_id, body.content)
        logger.info(
            "context fitted",
            extra={
                "session_id": session_id,
                "history_messages": turn.history_messages,
                "sent_messages": len(turn.window),
                "dropped_messages": turn.history_messages + 1 - len(turn.window),
                "estimated_tokens": turn.estimated_tokens,
                "budget_tokens": turn.budget_tokens,
            },
        )
        return StreamingResponse(
            chat_event_stream(
                service.stream_reply(turn),
                session_id=session_id,
                request_id=request_id_var.get(),
            ),
            media_type="text/event-stream",
            headers=_SSE_HEADERS,
        )

    return router


def register_error_handlers(app: FastAPI) -> None:
    """Map the domain errors that happen before a stream starts to JSON responses."""
    app.add_exception_handler(SessionNotFoundError, _session_not_found)
    app.add_exception_handler(ContextOverflowError, _context_overflow)


async def _session_not_found(request: Request, _exc: Exception) -> JSONResponse:
    logger.info("session not found", extra=_session_fields(request))
    return JSONResponse(
        {"detail": _SESSION_NOT_FOUND}, status_code=status.HTTP_404_NOT_FOUND
    )


async def _context_overflow(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, ContextOverflowError)  # registered for this type only
    logger.warning(
        "message rejected",
        extra={
            **_session_fields(request),
            "reason": "too_long",
            "estimated_tokens": exc.estimated_tokens,
            "budget_tokens": exc.budget_tokens,
        },
    )
    return JSONResponse(
        {"detail": _TOO_LONG}, status_code=status.HTTP_413_CONTENT_TOO_LARGE
    )


def _session_fields(request: Request) -> dict[str, int]:
    """Return ``{"session_id": <int>}`` when the request path has a valid one."""
    try:
        return {"session_id": int(request.path_params["session_id"])}
    except (KeyError, ValueError):
        return {}
