"""The v1 server-sent event wire contract for a streamed agent reply.

Each frame is ``event: <name>\\ndata: <json>\\n\\n``. The JSON is compact and
its first key is ``"v"``, the contract version:

    event: tool_call    data: {"v":1,"call_id":"...","name":"calculate"}
    event: tool_result  data: {"v":1,"call_id":"...","ok":true}
    event: delta        data: {"v":1,"text":"..."}
    event: done         data: {"v":1,"reply_chars":13}
    event: error        data: {"v":1,"code":"model_failed","message":"model request failed","request_id":"..."}

The ``error`` codes are ``context_full``, ``model_failed``, and
``internal_error``. Frames never carry tool arguments or tool results.
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

from mini_deepagents import (
    AgentEvent,
    ContextOverflowError,
    ModelError,
    ReplyDone,
    TextDelta,
    ToolFinished,
    ToolStarted,
)

__all__ = ["SSE_VERSION", "chat_event_stream", "encode_event"]

logger = logging.getLogger(__name__)

SSE_VERSION: Final = 1

_EVENT_TOOL_CALL: Final = "tool_call"
_EVENT_TOOL_RESULT: Final = "tool_result"
_EVENT_DELTA: Final = "delta"
_EVENT_DONE: Final = "done"
_EVENT_ERROR: Final = "error"
_CODE_CONTEXT_FULL: Final = "context_full"
_MESSAGE_CONTEXT_FULL: Final = "the conversation is too long for the context window"
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
    events: AsyncGenerator[AgentEvent, None], *, session_id: int, request_id: str
) -> AsyncGenerator[bytes, None]:
    """Encode agent events as SSE frames. Log the outcome; never log message content.

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
                    case ToolStarted(call=call):
                        yield encode_event(
                            _EVENT_TOOL_CALL, {"call_id": call.id, "name": call.name}
                        )
                    case ToolFinished():
                        yield encode_event(
                            _EVENT_TOOL_RESULT, {"call_id": event.call_id, "ok": event.ok}
                        )
                        _log_tool_finished(event, session_id)
                    case TextDelta(text=text):
                        if first_delta_ms is None:
                            first_delta_ms = _elapsed_ms(started)
                        yield encode_event(_EVENT_DELTA, {"text": text})
                    case ReplyDone():
                        yield encode_event(_EVENT_DONE, {"reply_chars": event.reply_chars})
                        _log_completed(
                            event,
                            session_id=session_id,
                            first_delta_ms=first_delta_ms,
                            duration_ms=_elapsed_ms(started),
                        )
    except ContextOverflowError as exc:
        logger.warning(
            "stream failed",
            extra={
                "session_id": session_id,
                "reason": _CODE_CONTEXT_FULL,
                "estimated_tokens": exc.estimated_tokens,
                "budget_tokens": exc.budget_tokens,
            },
        )
        yield _error_frame(_CODE_CONTEXT_FULL, _MESSAGE_CONTEXT_FULL, request_id)
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


def _log_tool_finished(event: ToolFinished, session_id: int) -> None:
    # "name" is a reserved LogRecord attribute, so the field is "tool_name".
    logger.info(
        "tool finished",
        extra={
            "session_id": session_id,
            "tool_name": event.name,
            "ok": event.ok,
            "result_chars": event.result_chars,
            "duration_ms": event.duration_ms,
        },
    )


def _log_completed(
    event: ReplyDone, *, session_id: int, first_delta_ms: int | None, duration_ms: int
) -> None:
    usage = event.usage
    logger.info(
        "stream completed",
        extra={
            "session_id": session_id,
            "reply_chars": event.reply_chars,
            "steps": event.steps,
            "peak_estimated_tokens": event.peak_estimated_tokens,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
            "first_delta_ms": first_delta_ms,
            "duration_ms": duration_ms,
        },
    )


def _error_frame(code: str, message: str, request_id: str) -> bytes:
    return encode_event(
        _EVENT_ERROR, {"code": code, "message": message, "request_id": request_id}
    )


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)
