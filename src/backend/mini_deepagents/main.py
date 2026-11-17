"""FastAPI app: the chat API and the one-page web UI.

Run with: uv run --env-file .env uvicorn mini_deepagents.main:create_app --factory
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from mini_deepagents.config import Settings
from mini_deepagents.chat.context import ContextOverflowError, estimate_tokens, fit_to_budget
from mini_deepagents.adapters.llm.opencode import ChatModel, LLMError, OpenCodeClient
from mini_deepagents.observability.logging import request_id_var, setup_logging
from mini_deepagents.adapters.storage.sqlite import ChatStore, SessionNotFoundError

logger = logging.getLogger(__name__)

_INDEX_HTML = Path(__file__).resolve().parents[2] / "frontend" / "index.html"
_SESSION_NOT_FOUND = "session not found"
_TOO_LONG = "message is too long for the context window"
_MODEL_FAILED = "model request failed"
_INTERNAL_ERROR = "internal server error"
_MODEL_SESSION_PREFIX = "mini-deepagents-"
_REQUEST_ID_HEADER = "X-Request-ID"
_REQUEST_ID_LENGTH = 12


class SendMessageRequest(BaseModel):
    content: Annotated[str, Field(min_length=1)]


class SendMessageResponse(BaseModel):
    reply: str


class SessionResponse(BaseModel):
    id: int
    created_at: str


class MessageResponse(BaseModel):
    role: str
    content: str


def create_app(settings: Settings | None = None, model: ChatModel | None = None) -> FastAPI:
    """Build the app. Load settings from the environment when ``settings`` is None.

    When ``model`` is None, the app owns an ``OpenCodeClient`` and closes it on
    shutdown. The app never closes a model that the caller passes in.
    """
    if settings is None:
        settings = Settings.from_env()
        setup_logging(settings.log_level, settings.log_file)
        _log_startup(settings)

    owned_client: OpenCodeClient | None = None
    if model is None:
        owned_client = OpenCodeClient(settings.base_url, settings.api_key, settings.model)
        model = owned_client

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            if owned_client is not None:
                owned_client.close()

    store = ChatStore(settings.db_path)
    app = FastAPI(title="mini-deepagents", lifespan=lifespan)
    _add_request_logging(app)
    _add_routes(app, settings, store, model)
    return app


def _log_startup(settings: Settings) -> None:
    logger.info(
        "app started",
        extra={
            "model": settings.model,
            "base_url": settings.base_url,
            "context_window": settings.context_window,
            "max_output_tokens": settings.max_output_tokens,
            "input_budget": settings.input_budget,
            "db_path": str(settings.db_path),
            "log_file": str(settings.log_file),
        },
    )


def _add_request_logging(app: FastAPI) -> None:
    @app.middleware("http")
    async def log_request(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = uuid4().hex[:_REQUEST_ID_LENGTH]
        token = request_id_var.set(request_id)
        fields = {"method": request.method, "path": request.url.path}
        started = time.perf_counter()
        try:
            logger.info("request received", extra=fields)
            try:
                response = await call_next(request)
            except Exception:
                logger.exception("unhandled error", extra=fields)
                response = JSONResponse(
                    {"detail": _INTERNAL_ERROR},
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                )
            duration_ms = int((time.perf_counter() - started) * 1000)
            logger.info(
                "request completed",
                extra={**fields, "status": response.status_code, "duration_ms": duration_ms},
            )
            response.headers[_REQUEST_ID_HEADER] = request_id
            return response
        finally:
            request_id_var.reset(token)


def _add_routes(app: FastAPI, settings: Settings, store: ChatStore, model: ChatModel) -> None:
    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(_INDEX_HTML, media_type="text/html")

    @app.get("/api/sessions")
    def list_sessions() -> list[SessionResponse]:
        return [SessionResponse.model_validate(s) for s in store.list_sessions()]

    @app.post("/api/sessions", status_code=status.HTTP_201_CREATED)
    def create_session() -> SessionResponse:
        session = store.create_session()
        logger.info("session created", extra={"session_id": session["id"]})
        return SessionResponse.model_validate(session)

    @app.get("/api/sessions/{session_id}/messages")
    def get_messages(session_id: int) -> list[MessageResponse]:
        messages = _load_history(store, session_id)
        return [MessageResponse.model_validate(m) for m in messages]

    @app.post("/api/sessions/{session_id}/messages")
    def send_message(session_id: int, request: SendMessageRequest) -> SendMessageResponse:
        content = request.content
        history = _load_history(store, session_id)
        window = _fit_window(session_id, history, content, settings.input_budget)

        try:
            reply = model.chat(
                window,
                settings.max_output_tokens,
                session_id=f"{_MODEL_SESSION_PREFIX}{session_id}",
            )
        except LLMError as exc:
            logger.error(
                "model call failed", extra={"session_id": session_id, "error": str(exc)}
            )
            raise HTTPException(status.HTTP_502_BAD_GATEWAY, _MODEL_FAILED) from exc

        store.add_exchange(session_id, content, reply)
        logger.info("exchange saved", extra={"session_id": session_id, "reply_chars": len(reply)})
        return SendMessageResponse(reply=reply)


def _load_history(store: ChatStore, session_id: int) -> list[dict[str, str]]:
    try:
        return store.get_messages(session_id)
    except SessionNotFoundError as exc:
        logger.info("session not found", extra={"session_id": session_id})
        raise HTTPException(status.HTTP_404_NOT_FOUND, _SESSION_NOT_FOUND) from exc


def _fit_window(
    session_id: int, history: list[dict[str, str]], content: str, budget: int
) -> list[Mapping[str, str]]:
    new_message = {"role": "user", "content": content}
    try:
        window = fit_to_budget([*history, new_message], budget)
    except ContextOverflowError as exc:
        logger.warning(
            "message rejected",
            extra={
                "session_id": session_id,
                "reason": "too_long",
                "estimated_tokens": estimate_tokens(new_message),
                "budget_tokens": budget,
            },
        )
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, _TOO_LONG) from exc

    logger.info(
        "context fitted",
        extra={
            "session_id": session_id,
            "history_messages": len(history),
            "sent_messages": len(window),
            "dropped_messages": len(history) + 1 - len(window),
            "estimated_tokens": sum(estimate_tokens(m) for m in window),
            "budget_tokens": budget,
        },
    )
    return window
