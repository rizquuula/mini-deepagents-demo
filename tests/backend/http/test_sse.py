from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncGenerator, Sequence

import pytest

from mini_deepagents import (
    AgentEvent,
    ContextOverflowError,
    ModelError,
    ReplyDone,
    TextDelta,
    ToolCall,
    ToolFinished,
    ToolStarted,
    Usage,
)

from mini_deepagents_demo.adapters.http.sse import chat_event_stream, encode_event

SECRET_TEXT = "secret-user-text"
SECRET_ARGUMENT = "secret-argument-text"
ONE_TOOL_TURN: tuple[AgentEvent, ...] = (
    ToolStarted(ToolCall("call-1", "calculate", f'{{"expression":"{SECRET_ARGUMENT}"}}')),
    ToolFinished("call-1", "calculate", ok=True, result_chars=2, duration_ms=3),
    TextDelta("42"),
    ReplyDone(2, Usage(10, 2, 12), steps=2, peak_estimated_tokens=900),
)


def find_record(caplog: pytest.LogCaptureFixture, message: str) -> logging.LogRecord:
    [record] = [r for r in caplog.records if r.getMessage() == message]
    return record


def parse_frames(chunks: Sequence[bytes]) -> list[tuple[str, dict[str, object]]]:
    frames = []
    for chunk in chunks:
        event_line, data_line = chunk.decode("utf-8").rstrip("\n").split("\n")
        frames.append(
            (event_line.removeprefix("event: "), json.loads(data_line.removeprefix("data: ")))
        )
    return frames


async def source(
    events: Sequence[AgentEvent], *, error: Exception | None = None
) -> AsyncGenerator[AgentEvent, None]:
    for event in events:
        yield event
    if error is not None:
        raise error


async def collect(stream: AsyncGenerator[bytes, None]) -> list[bytes]:
    return [chunk async for chunk in stream]


def stream_one_tool_turn() -> list[bytes]:
    return asyncio.run(
        collect(chat_event_stream(source(ONE_TOOL_TURN), session_id=7, request_id="rid"))
    )


def test_encode_event_writes_compact_json_with_the_version_first() -> None:
    frame = encode_event("delta", {"text": "héllo 👋"})

    assert frame == 'event: delta\ndata: {"v":1,"text":"héllo 👋"}\n\n'.encode()


def test_stream_maps_deltas_and_done_and_logs_the_usage(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    events = [
        TextDelta("Hel"),
        TextDelta("lo"),
        ReplyDone(5, Usage(10, 2, 12), steps=2, peak_estimated_tokens=900),
    ]

    chunks = asyncio.run(
        collect(chat_event_stream(source(events), session_id=7, request_id="rid"))
    )

    assert parse_frames(chunks) == [
        ("delta", {"v": 1, "text": "Hel"}),
        ("delta", {"v": 1, "text": "lo"}),
        ("done", {"v": 1, "reply_chars": 5}),
    ]
    record = find_record(caplog, "stream completed")
    assert record.session_id == 7
    assert record.reply_chars == 5
    assert (record.prompt_tokens, record.completion_tokens, record.total_tokens) == (10, 2, 12)
    assert (record.steps, record.peak_estimated_tokens) == (2, 900)
    assert isinstance(record.first_delta_ms, int)
    assert isinstance(record.duration_ms, int)


def test_one_tool_turn_sends_tool_call_and_tool_result_frames_before_the_reply() -> None:
    assert parse_frames(stream_one_tool_turn()) == [
        ("tool_call", {"v": 1, "call_id": "call-1", "name": "calculate"}),
        ("tool_result", {"v": 1, "call_id": "call-1", "ok": True}),
        ("delta", {"v": 1, "text": "42"}),
        ("done", {"v": 1, "reply_chars": 2}),
    ]


def test_tool_arguments_reach_neither_frames_nor_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)

    chunks = stream_one_tool_turn()

    assert SECRET_ARGUMENT.encode() not in b"".join(chunks)
    assert all(SECRET_ARGUMENT not in str(vars(record)) for record in caplog.records)


def test_finished_tool_is_logged_with_its_outcome_and_cost(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)

    stream_one_tool_turn()

    record = find_record(caplog, "tool finished")
    assert record.levelno == logging.INFO
    assert (record.session_id, record.tool_name, record.ok) == (7, "calculate", True)
    assert (record.result_chars, record.duration_ms) == (2, 3)


def test_context_overflow_in_a_tool_step_sends_a_context_full_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    overflow = ContextOverflowError(
        "tool steps need 7000 tokens", estimated_tokens=7000, budget_tokens=6000
    )
    events = source(ONE_TOOL_TURN[:2], error=overflow)

    chunks = asyncio.run(collect(chat_event_stream(events, session_id=4, request_id="rid")))

    assert parse_frames(chunks)[-1] == (
        "error",
        {
            "v": 1,
            "code": "context_full",
            "message": "the conversation is too long for the context window",
            "request_id": "rid",
        },
    )
    record = find_record(caplog, "stream failed")
    assert record.levelno == logging.WARNING
    assert (record.session_id, record.reason) == (4, "context_full")
    assert (record.estimated_tokens, record.budget_tokens) == (7000, 6000)


def test_model_error_after_a_delta_sends_an_error_event_without_content_in_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    events = source([TextDelta(SECRET_TEXT)], error=ModelError("status=502"))

    chunks = asyncio.run(collect(chat_event_stream(events, session_id=3, request_id="rid")))

    assert parse_frames(chunks) == [
        ("delta", {"v": 1, "text": SECRET_TEXT}),
        (
            "error",
            {
                "v": 1,
                "code": "model_failed",
                "message": "model request failed",
                "request_id": "rid",
            },
        ),
    ]
    record = find_record(caplog, "stream failed")
    assert record.levelno == logging.ERROR
    assert record.session_id == 3
    assert record.error == "status=502"
    assert SECRET_TEXT not in caplog.text


def test_closing_the_byte_stream_early_logs_cancel_and_closes_the_source(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    closed: list[bool] = []

    async def endless() -> AsyncGenerator[AgentEvent, None]:
        try:
            while True:
                yield TextDelta("x")
        finally:
            closed.append(True)

    async def read_one_then_close() -> None:
        stream = chat_event_stream(endless(), session_id=5, request_id="rid")
        await anext(stream)
        await stream.aclose()

    asyncio.run(read_one_then_close())

    assert closed == [True]
    record = find_record(caplog, "stream cancelled")
    assert record.session_id == 5
    assert isinstance(record.duration_ms, int)
