"""The v1 server-sent event wire contract for a streamed chat reply.

Each frame is ``event: <name>\\ndata: <json>\\n\\n``. The JSON is compact and
its first key is ``"v"``, the contract version:

    event: delta   data: {"v":1,"text":"..."}
    event: done    data: {"v":1,"reply_chars":13}
    event: error   data: {"v":1,"code":"model_failed","message":"model request failed","request_id":"..."}

A stream ends with exactly one ``done`` or one ``error`` frame.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncGenerator, Mapping
from contextlib import aclosing
from typing import Final

from mini_deepagents.chat import ChatEvent, ReplyDone, TextDelta
from mini_deepagents.errors import ModelError

__all__ = ["SSE_VERSION", "chat_event_stream", "encode_event"]

logger = logging.getLogger(__name__)

SSE_VERSION: Final = 1

_EVENT_DELTA: Final = "delta"
_EVENT_DONE: Final = "done"
_EVENT_ERROR: Final = "error"
_CODE_MODEL_FAILED: Final = "model_failed"
_MESSAGE_MODEL_FAILED: Final = "model request failed"
_CODE_INTERNAL_ERROR: Final = "internal_error"
_MESSAGE_INTERNAL_ERROR: Final = "internal server error"


def encode_event(name: str, payload: Mapping[str, object]) -> bytes:
    """Return one SSE frame. ``"v"`` is the first key of the JSON data."""
    data = json.dumps(
        {"v": SSE_VERSION, **payload}, separators=(",", ":"), ensure_ascii=False
    )
    return f"event: {name}\ndata: {data}\n\n".encode()


async def chat_event_stream(
    events: AsyncGenerator[ChatEvent, None], *, session_id: int, request_id: str
) -> AsyncGenerator[bytes, None]:
    """Encode chat events as SSE frames. Log the outcome; never log message content.

    A failure after the stream starts becomes an ``error`` frame, because the
    HTTP status is already sent. A cancellation or an early close closes
    ``events`` and re-raises.
    """
    started = time.perf_counter()
    first_delta_ms: int | None = None
    try:
        async with aclosing(events):
            async for event in events:
                match event:
                    case TextDelta(text=text):
                        if first_delta_ms is None:
                            first_delta_ms = _elapsed_ms(started)
                        yield encode_event(_EVENT_DELTA, {"text": text})
                    case ReplyDone(reply_chars=reply_chars, usage=usage):
                        yield encode_event(_EVENT_DONE, {"reply_chars": reply_chars})
                        logger.info(
                            "stream completed",
                            extra={
                                "session_id": session_id,
                                "reply_chars": reply_chars,
                                "prompt_tokens": usage.prompt_tokens,
                                "completion_tokens": usage.completion_tokens,
                                "total_tokens": usage.total_tokens,
                                "first_delta_ms": first_delta_ms,
                                "duration_ms": _elapsed_ms(started),
                            },
                        )
    except ModelError as exc:
        logger.error("stream failed", extra={"session_id": session_id, "error": str(exc)})
        yield _error_frame(_CODE_MODEL_FAILED, _MESSAGE_MODEL_FAILED, request_id)
    except Exception:
        logger.exception("stream failed", extra={"session_id": session_id})
        yield _error_frame(_CODE_INTERNAL_ERROR, _MESSAGE_INTERNAL_ERROR, request_id)
    except (asyncio.CancelledError, GeneratorExit):
        logger.info(
            "stream cancelled",
            extra={"session_id": session_id, "duration_ms": _elapsed_ms(started)},
        )
        raise


def _error_frame(code: str, message: str, request_id: str) -> bytes:
    return encode_event(
        _EVENT_ERROR, {"code": code, "message": message, "request_id": request_id}
    )


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)
