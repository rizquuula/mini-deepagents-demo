"""The composition root: build every adapter and wire the app.

This is the only module that reads the environment (through
``Settings.from_env``) and the only module that knows every adapter.

Run with: uv run --env-file .env uvicorn mini_deepagents.main:create_app --factory
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Final

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from mini_deepagents.adapters.http import (
    RequestContextMiddleware,
    create_api_router,
    register_error_handlers,
)
from mini_deepagents.adapters.llm import create_chat_model
from mini_deepagents.adapters.storage import create_repository
from mini_deepagents.chat import ByteTokenEstimator, ChatModel, ChatRepository, ChatService
from mini_deepagents.config import Settings
from mini_deepagents.observability.logging import setup_logging

__all__ = ["FRONTEND_DIR", "create_app"]

logger = logging.getLogger(__name__)

FRONTEND_DIR: Final = Path(__file__).resolve().parents[2] / "frontend"


def create_app(
    settings: Settings | None = None,
    *,
    model: ChatModel | None = None,
    repository: ChatRepository | None = None,
) -> FastAPI:
    """Build the app. Load settings from the environment when ``settings`` is None.

    The app builds the model and the repository that the caller does not pass.
    On shutdown, the app closes only a model that it built.
    """
    if settings is None:
        settings = Settings.from_env()
        setup_logging(settings.log_level, settings.log_file)
        _log_startup(settings)

    if repository is None:
        repository = create_repository(settings.storage_backend, settings)
    owned_model: ChatModel | None = None
    if model is None:
        owned_model = model = create_chat_model(settings.llm_provider, settings)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            if owned_model is not None:
                await owned_model.aclose()

    service = ChatService(
        repository,
        model,
        ByteTokenEstimator(),
        input_budget=settings.input_budget,
        max_output_tokens=settings.max_output_tokens,
    )

    app = FastAPI(title="mini-deepagents", lifespan=lifespan)
    app.add_middleware(RequestContextMiddleware)
    register_error_handlers(app)
    app.include_router(create_api_router(service))

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(FRONTEND_DIR / "index.html", media_type="text/html")

    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
    return app


def _log_startup(settings: Settings) -> None:
    logger.info(
        "app started",
        extra={
            "llm_provider": settings.llm_provider,
            "storage_backend": settings.storage_backend,
            "model": settings.model,
            "base_url": settings.base_url,
            "context_window": settings.context_window,
            "max_output_tokens": settings.max_output_tokens,
            "input_budget": settings.input_budget,
            "db_path": str(settings.db_path),
            "log_file": str(settings.log_file),
        },
    )
