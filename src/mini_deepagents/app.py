"""FastAPI app: the chat API and the one-page web UI.

Run with: uv run --env-file .env uvicorn mini_deepagents.app:create_app --factory
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from mini_deepagents.config import Settings
from mini_deepagents.context import ContextOverflowError, fit_to_budget
from mini_deepagents.llm import ChatModel, LLMError, OpenCodeClient
from mini_deepagents.store import ChatStore, SessionNotFoundError

logger = logging.getLogger(__name__)

_INDEX_HTML = Path(__file__).parent / "static" / "index.html"
_SESSION_NOT_FOUND = "session not found"
_TOO_LONG = "message is too long for the context window"
_MODEL_FAILED = "model request failed"


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
        logging.basicConfig(level=logging.INFO)
        settings = Settings.from_env()

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
    _add_routes(app, settings, store, model)
    return app


def _add_routes(app: FastAPI, settings: Settings, store: ChatStore, model: ChatModel) -> None:
    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(_INDEX_HTML, media_type="text/html")

    @app.get("/api/sessions")
    def list_sessions() -> list[SessionResponse]:
        return [SessionResponse.model_validate(s) for s in store.list_sessions()]

    @app.post("/api/sessions", status_code=status.HTTP_201_CREATED)
    def create_session() -> SessionResponse:
        return SessionResponse.model_validate(store.create_session())

    @app.get("/api/sessions/{session_id}/messages")
    def get_messages(session_id: int) -> list[MessageResponse]:
        try:
            messages = store.get_messages(session_id)
        except SessionNotFoundError as exc:
            raise HTTPException(status.HTTP_404_NOT_FOUND, _SESSION_NOT_FOUND) from exc
        return [MessageResponse.model_validate(m) for m in messages]

    @app.post("/api/sessions/{session_id}/messages")
    def send_message(session_id: int, request: SendMessageRequest) -> SendMessageResponse:
        content = request.content
        try:
            history = store.get_messages(session_id)
        except SessionNotFoundError as exc:
            raise HTTPException(status.HTTP_404_NOT_FOUND, _SESSION_NOT_FOUND) from exc

        candidate = [*history, {"role": "user", "content": content}]
        try:
            window = fit_to_budget(candidate, settings.input_budget)
        except ContextOverflowError as exc:
            raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, _TOO_LONG) from exc

        try:
            reply = model.chat(window, settings.max_output_tokens)
        except LLMError as exc:
            logger.warning("model call failed session_id=%s error=%s", session_id, exc)
            raise HTTPException(status.HTTP_502_BAD_GATEWAY, _MODEL_FAILED) from exc

        store.add_exchange(session_id, content, reply)
        return SendMessageResponse(reply=reply)
